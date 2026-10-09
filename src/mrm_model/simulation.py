"""Coupled multirate plant, shared controller interface and truthful evaluation.

Slow optics averages two quasi-static NRZ rails. It does not resolve ISI or
predict BER. Short waveform experiments use waveform.py's independent fast mode.
"""

import csv
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from warnings import warn

import numpy as np
from scipy.linalg import expm

from .config import ExperimentConfig
from .controllers import ControllerContext, Observation, make_controller
from .hardware import HeaterDriver, MonitorADC
from .optics import optical_response, resonance_frequency
from .thermal import ThermalNetwork


@dataclass(frozen=True)
class BusResponse:
    monitor_power_w: np.ndarray
    absorbed_power_w: np.ndarray
    output_low_w: np.ndarray
    output_high_w: np.ndarray
    drop_by_ring_and_carrier_w: np.ndarray


def bus_response(config, temperature_k, laser_shift_hz=0.0, power_fraction=1.0):
    """All carriers encounter every ring in order, including periodic aliases.

    Different lasers are treated as incoherent over the monitor bandwidth. Their
    hundreds-of-GHz beats are averaged. Fast WDM data is approximated by independent
    rail duty averages through upstream rings. It is NOT joint bit-pattern simulation.
    """
    r = config.run
    n = len(config.optical)
    frequencies = np.asarray(r.laser_frequency_hz) + laser_shift_hz
    carrier_power = np.full(n, r.input_power_w * power_fraction)
    monitor = np.zeros(n)
    absorbed = np.zeros(n)
    dropped = np.zeros((n, n))
    assigned_rail_transmission = np.ones((2, n))
    average_transmissions = []
    attenuation = 10 ** (-r.bus_loss_db_between_rings / 10)
    for i, p in enumerate(config.optical):
        low = optical_response(p, frequencies, temperature_k[i], r.junction_low_v, 1)
        high = optical_response(p, frequencies, temperature_k[i], r.junction_high_v, 1)
        drop = carrier_power * (
            (1 - r.data_one_fraction) * low.drop_power_w + r.data_one_fraction * high.drop_power_w
        )
        monitor[i] = np.sum(drop)
        dropped[i] = drop
        absorbed[i] = np.sum(
            carrier_power
            * ((1 - r.data_one_fraction) * low.absorbed_power_w + r.data_one_fraction * high.absorbed_power_w)
        )
        averaged = (
            1 - r.data_one_fraction
        ) * low.through_power_w + r.data_one_fraction * high.through_power_w
        average_transmissions.append(averaged)
        assigned_rail_transmission[:, i] = [low.through_power_w[i], high.through_power_w[i]]
        carrier_power = carrier_power * averaged
        if i < n - 1:
            carrier_power *= attenuation
    low_out = np.full(n, r.input_power_w * power_fraction * attenuation ** (n - 1))
    high_out = low_out.copy()
    for transmission in average_transmissions:
        low_out *= transmission
        high_out *= transmission
    for i in range(n):
        # Replace the assigned ring's duty-averaged factor with its bit-specific rail.
        factor = average_transmissions[i][i]
        if factor > 0:
            low_out[i] *= assigned_rail_transmission[0, i] / factor
            high_out[i] *= assigned_rail_transmission[1, i] / factor
    return BusResponse(monitor, absorbed, low_out, high_out, dropped)


def _steady_state(config):
    t = config.thermal
    r = config.run
    driver = HeaterDriver(
        replace(
            config.actuator,
            latency_s=0,
            settling_time_s=0,
            slew_rate_v_per_s=float("inf"),
            voltage_noise_std_v=0,
        ),
        len(config.optical),
    )
    network = ThermalNetwork(t, config.disturbance.ambient_temperature_k)
    ambient, external, shift, power_factor = config.disturbance.at(0)
    if r.initial_temperature_k is not None:
        network = ThermalNetwork(t, r.initial_temperature_k)
    elif r.initial_state_rise_k is not None:
        network.state_rise_k = np.asarray(r.initial_state_rise_k, dtype=float).copy()
        if np.any(network.ring_temperatures_k <= 0) or np.any(network.heater_temperatures_k <= 0):
            raise ValueError("initial generalized state reconstructs nonpositive temperatures")
    else:
        for _ in range(150):
            driver.step(1e-9, r.initial_codes, network.heater_temperatures_k)
            bus = bus_response(config, network.ring_temperatures_k, shift, power_factor)
            heat = np.zeros(t.n_states) if external is None else external.copy()
            if r.optical_heating:
                heat += t.ring_map.T @ bus.absorbed_power_w
            forcing = (
                t.heater_map @ driver.power_w
                + t.boundary_map_w_per_k
                @ np.full(t.boundary_map_w_per_k.shape[1], ambient - t.reference_temperature_k)
                + heat
            )
            target = np.linalg.solve(t.conductance_w_per_k, forcing)
            difference = np.max(abs(target - network.state_rise_k))
            network.state_rise_k = 0.5 * (target + network.state_rise_k)
            if difference < 1e-8:
                break
        else:
            raise ValueError(
                "Initial nonlinear electrothermal equilibrium failed; provide initial states "
                "or lower absorbed optical power. This may indicate bistability."
            )
    # Explicit thermal states skip the equilibrium iteration, but the offline
    # algebraic actuator must still represent the accepted initial command.
    # Refresh resistance at the finalized temperature in every initialization
    # path before nominal_context differentiates heater power with respect to code.
    driver.step(0, r.initial_codes, network.heater_temperatures_k)
    return network, driver


def _actuator_operating_derivatives(params, codes, heater_temperature_k):
    """Continuous accepted-DAC-code derivatives on a fixed local active set.

    Quantization is intentionally not differentiated.  At a strict compliance
    limit the DAC has no local authority.  Temperature still changes a current
    limit's voltage and a power limit's voltage, so fixed-voltage TCR and total
    settled-power TCR derivatives must be distinguished.  Thresholds and the
    driver's instantaneous output projection are nonsmooth; this local model
    cannot globally predict a trajectory that crosses those constraints.
    """
    codes, temperature = np.broadcast_arrays(
        np.asarray(codes, dtype=float), np.asarray(heater_temperature_k, dtype=float)
    )
    resistance = params.heater_resistance_ohm * (
        1 + params.resistance_tcr_per_k * (temperature - params.reference_temperature_k)
    )
    resistance_derivative = params.heater_resistance_ohm * params.resistance_tcr_per_k
    dac_scale = params.dac_gain * params.dac_reference_v / (2**params.dac_bits - 1)
    raw = dac_scale * codes + params.dac_offset_v
    limits = np.stack(
        [
            np.full_like(resistance, params.max_voltage_v),
            params.max_current_a * resistance,
            np.sqrt(params.max_power_w * resistance),
        ]
    )
    active_upper = np.argmin(limits, axis=0)
    upper = np.min(limits, axis=0)
    lower = np.maximum(raw, params.min_voltage_v)
    upper_active = lower >= upper
    voltage = np.minimum(lower, upper)
    upper_temperature_derivatives = np.stack(
        [
            np.zeros_like(resistance),
            np.full_like(resistance, params.max_current_a * resistance_derivative),
            0.5 * limits[2] * resistance_derivative / resistance,
        ]
    )
    dv_du = np.where((raw > params.min_voltage_v) & ~upper_active, dac_scale, 0.0)
    dv_dt = np.where(
        upper_active,
        np.take_along_axis(upper_temperature_derivatives, active_upper[None, :], axis=0)[0],
        0.0,
    )
    dp_dt_fixed_voltage = -((voltage / resistance) ** 2) * resistance_derivative
    dp_du = 2 * voltage / resistance * dv_du
    dp_dt = dp_dt_fixed_voltage + 2 * voltage / resistance * dv_dt
    # Use a deterministic upper-active one-sided derivative at a threshold, but
    # do not present it as a unique differentiable calibration.
    tolerance = 1e-11 * np.maximum(1.0, np.maximum(abs(lower), upper))
    lower_threshold = (abs(raw - params.min_voltage_v) <= tolerance) & ~upper_active
    upper_threshold = abs(lower - upper) <= tolerance
    upper_tie = (np.sum(abs(limits - upper) <= tolerance, axis=0) > 1) & upper_active
    if np.any(lower_threshold | upper_threshold | upper_tie):
        warn(
            "Actuator operating point is on a nonsmooth compliance threshold; nominal calibration "
            "uses a one-sided active-set derivative, not a unique global clipping model.",
            RuntimeWarning,
            stacklevel=2,
        )
    if np.any(upper_active) and params.settling_time_s > 0:
        warn(
            "Nominal actuator calibration uses the active compliance target and its local settling; "
            "instantaneous output clipping remains a nonlinear plant constraint.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "voltage_v": voltage,
        "dv_du": dv_du,
        "dv_dt": dv_dt,
        "dp_du": dp_du,
        "dp_dt": dp_dt,
        "dp_dt_fixed_voltage": dp_dt_fixed_voltage,
    }


def nominal_context(config):
    """Offline local plant calibration, not measurements from the active run.

    Caller may replace this context with an independently identified model to
    test sim-to-real model mismatch. All controllers see the same nominal data.
    """
    network, driver = _steady_state(config)
    r, t, s, p = config.run, config.thermal, config.sensor, config.actuator
    temperatures = network.ring_temperatures_k
    monitor = bus_response(config, temperatures).monitor_power_w
    target = (
        s.adc_gain * (s.responsivity_a_per_w * monitor + s.dark_current_a) + s.adc_offset_a
    ) / s.adc_full_scale_a
    adc_observable = (target > 0) & (target < 1)
    if not np.all(adc_observable):
        warn(
            "Initial ADC operating point is clipped or on a clipping threshold; nominal calibration "
            "uses a clipped target and zero local output slope. Change optical power, gain or ADC "
            "range before using a local flank controller.",
            RuntimeWarning,
            stacklevel=2,
        )
    target = np.clip(target, 0, 1)
    n = len(config.optical)
    dmonitor_dt = np.zeros((n, n))
    dabsorption_dt = np.zeros((n, n))
    eps = 1e-3
    for j in range(n):
        delta = np.eye(n)[j] * eps
        plus = bus_response(config, temperatures + delta)
        minus = bus_response(config, temperatures - delta)
        dmonitor_dt[:, j] = (plus.monitor_power_w - minus.monitor_power_w) / (2 * eps)
        dabsorption_dt[:, j] = (plus.absorbed_power_w - minus.absorbed_power_w) / (2 * eps)
    sensor_scale = s.adc_gain * s.responsivity_a_per_w / s.adc_full_scale_a
    thermal_output_c = sensor_scale * dmonitor_dt @ t.ring_map
    v = driver.voltage_v
    actuator_derivatives = _actuator_operating_derivatives(p, driver.codes, network.heater_temperatures_k)
    dpdu = actuator_derivatives["dp_du"]
    ss = network.state_space()
    continuous_b = ss["B"] @ np.diag(dpdu)
    # This settled-state Jacobian includes the compliance target's temperature
    # dependence.  With a dynamic voltage state, only fixed-voltage TCR belongs
    # in the thermal block; the target dependence belongs in the driver block.
    dpdt = actuator_derivatives["dp_dt"]
    continuous_a = ss["A"] + ss["B"] @ np.diag(dpdt) @ t.heater_temperature_map
    if r.optical_heating:
        continuous_a += ss["Q"] @ t.ring_map.T @ dabsorption_dt @ t.ring_map
    if np.max(np.linalg.eigvals(continuous_a).real) >= 0:
        raise ValueError(
            "Initial electrothermal operating point is locally unstable; provide another "
            "branch or reduce optical power. Fixed-point convergence alone is insufficient."
        )
    jacobian = -(adc_observable[:, None] * thermal_output_c) @ np.linalg.solve(continuous_a, continuous_b)
    # Exact local driver and analog-filter dynamics are part of the identified
    # nominal model, as they are in the physical plant.
    thermal_states = t.n_states
    driver_states = n if p.settling_time_s > 0 else 0
    states = thermal_states + driver_states + n
    physical_a = np.zeros((states, states))
    physical_b = np.zeros((states, n))
    physical_a[:thermal_states, :thermal_states] = continuous_a
    if driver_states:
        voltage_slice = slice(thermal_states, thermal_states + n)
        physical_a[:thermal_states, :thermal_states] += (
            ss["B"] @ np.diag(actuator_derivatives["dp_dt_fixed_voltage"] - dpdt) @ t.heater_temperature_map
        )
        physical_a[:thermal_states, voltage_slice] = ss["B"] @ np.diag(2 * v / driver.resistance_ohm)
        physical_a[voltage_slice, voltage_slice] = -np.eye(n) / p.settling_time_s
        physical_a[voltage_slice, :thermal_states] = (
            np.diag(actuator_derivatives["dv_dt"]) @ t.heater_temperature_map / p.settling_time_s
        )
        physical_b[voltage_slice] = np.diag(actuator_derivatives["dv_du"]) / p.settling_time_s
    else:
        physical_b[:thermal_states] = continuous_b
    filter_slice = slice(states - n, states)
    sensor_rate = 2 * np.pi * s.bandwidth_hz
    physical_a[filter_slice, :thermal_states] = sensor_rate * thermal_output_c
    physical_a[filter_slice, filter_slice] = -sensor_rate * np.eye(n)
    if np.max(np.linalg.eigvals(physical_a).real) >= 0:
        raise ValueError(
            "Initial electrothermal operating point with driver dynamics is locally unstable; "
            "provide another branch or reduce optical power."
        )
    output_c = np.zeros((n, states))
    # Keep the analog filter's internal dynamics; clipping removes local ADC
    # observability at its output, rather than altering physical photocurrent.
    output_c[:, filter_slice] = np.diag(adc_observable)
    augmented = np.zeros((states + n, states + n))
    augmented[:states, :states] = physical_a
    augmented[:states, states:] = physical_b
    discrete = expm(augmented * r.sample_period_s)
    a = discrete[:states, :states]
    b = discrete[:states, states:]
    # Transport delay uses an exact within-sample split and a DAC-command history.
    # ceil(sensor latency / sample period) matches which timestamped ADC sample is
    # delivered at each control call; it does not invent interpolated ADC values.
    whole = int(np.floor(p.latency_s / r.sample_period_s + 1e-10))
    remainder = max(0.0, p.latency_s - whole * r.sample_period_s)
    if remainder < 1e-12 * r.sample_period_s:
        remainder = 0.0
    command_history = whole + (1 if remainder else 0)
    sensor_history = int(np.ceil(s.latency_s / r.sample_period_s - 1e-10))
    # The plant itself accepts arbitrary delay; this cap is only the bundled dense MPC.
    if max(command_history, sensor_history) <= 32 and states + n * (command_history + sensor_history) <= 512:
        if command_history:
            early = expm(augmented * remainder)
            late = expm(augmented * (r.sample_period_s - remainder))
            old_b = late[:states, :states] @ early[:states, states:]
            new_b = late[:states, states:]
            expanded_a = np.zeros((states + command_history * n, states + command_history * n))
            expanded_b = np.zeros((states + command_history * n, n))
            expanded_a[:states, :states] = a
            if whole == 0:
                expanded_b[:states] = new_b
            else:
                column = states + (whole - 1) * n
                expanded_a[:states, column : column + n] += new_b
            if remainder:
                column = states + whole * n
                expanded_a[:states, column : column + n] += old_b
            expanded_b[states : states + n] = np.eye(n)
            for j in range(1, command_history):
                expanded_a[states + j * n : states + (j + 1) * n, states + (j - 1) * n : states + j * n] = (
                    np.eye(n)
                )
            output_c = np.pad(output_c, ((0, 0), (0, command_history * n)))
            a, b = expanded_a, expanded_b
            states = len(a)
        if sensor_history:
            expanded_a = np.zeros((states + sensor_history * n, states + sensor_history * n))
            expanded_b = np.zeros((states + sensor_history * n, n))
            expanded_a[:states, :states] = a
            expanded_b[:states] = b
            expanded_a[states : states + n, :states] = output_c
            for j in range(1, sensor_history):
                expanded_a[states + j * n : states + (j + 1) * n, states + (j - 1) * n : states + j * n] = (
                    np.eye(n)
                )
            delayed_c = np.zeros((n, len(expanded_a)))
            delayed_c[:, states + (sensor_history - 1) * n : states + sensor_history * n] = np.eye(n)
            a, b, output_c = expanded_a, expanded_b, delayed_c
    return ControllerContext(
        r.sample_period_s,
        np.asarray(r.initial_codes, dtype=float),
        2**p.dac_bits - 1,
        target,
        jacobian,
        2**s.adc_bits - 1,
        a,
        b,
        output_c,
    )


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    traces: dict[str, np.ndarray]
    metrics: dict
    warnings: list[str]

    def save(self, folder):
        destination = Path(folder)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text(self.config.to_json() + "\n")
        np.savez_compressed(destination / "traces.npz", **self.traces)
        (destination / "metrics.json").write_text(
            json.dumps({"metrics": self.metrics, "warnings": self.warnings}, indent=2, allow_nan=False) + "\n"
        )
        columns = []
        arrays = []
        for name, value in self.traces.items():
            if value.ndim == 1:
                columns.append(name)
                arrays.append(value)
            elif value.ndim == 2:
                for i in range(value.shape[1]):
                    columns.append(f"{name}_{i}")
                    arrays.append(value[:, i])
        with (destination / "traces.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(zip(*arrays, strict=True))


def run_experiment(config: ExperimentConfig, controller=None, context=None, progress=None):
    start = time.monotonic()
    n = len(config.optical)
    r, t = config.run, config.thermal
    network, _ = _steady_state(config)
    # Independent deterministic streams: controller changes don't resample sensor noise.
    seeds = np.random.SeedSequence(r.seed).spawn(2)
    driver = HeaterDriver(config.actuator, n, np.random.default_rng(seeds[0]))
    driver.prime(r.initial_codes, network.heater_temperatures_k)
    sensor = MonitorADC(config.sensor, n, np.random.default_rng(seeds[1]))
    initial_bus = bus_response(config, network.ring_temperatures_k)
    sensor.filtered_current_a = (
        config.sensor.responsivity_a_per_w * initial_bus.monitor_power_w + config.sensor.dark_current_a
    )
    ctx = nominal_context(config) if context is None else context
    algorithm = make_controller(config.controller, ctx) if controller is None else controller
    if (
        controller is None
        and config.controller.kind == "mpc"
        and (
            max(config.sensor.latency_s, config.actuator.latency_s) / r.sample_period_s > 32
            or t.n_states
            + 2 * n
            + n
            * (
                int(np.ceil(config.sensor.latency_s / r.sample_period_s))
                + int(np.ceil(config.actuator.latency_s / r.sample_period_s))
            )
            > 512
        )
    ):
        raise ValueError(
            "Built-in dense MPC is limited to 32 samples of delay and 512 states; "
            "use a custom sparse/delay-aware controller for this case"
        )
    command = np.asarray(r.initial_codes, dtype=float)
    sample = sensor.sample()
    target_detuning = np.asarray(r.laser_frequency_hz) - np.array(
        [
            resonance_frequency(p, network.ring_temperatures_k[i], (r.junction_low_v + r.junction_high_v) / 2)
            for i, p in enumerate(config.optical)
        ]
    )
    if r.target_detuning_hz is not None:
        target_detuning = np.asarray(r.target_detuning_hz)
    steps = int(np.ceil(r.duration_s / r.sample_period_s))
    recorded_rows = (steps - 1) // r.record_every + 1 + int((steps - 1) % r.record_every != 0)
    scalar_keys = {"time_s", "ambient_temperature_k", "sample_time_s"}
    output = {
        k: np.empty(recorded_rows if k in scalar_keys else (recorded_rows, n), dtype=float)
        for k in (
            "time_s",
            "ambient_temperature_k",
            "ring_temperature_k",
            "heater_power_w",
            "command_codes",
            "adc_codes",
            "sample_time_s",
            "detuning_hz",
            "tracking_error_hz",
            "oma_w",
            "through_low_w",
            "through_high_w",
            "monitor_power_w",
            "alias_fraction",
        )
    }
    previous_locked = np.ones(n, dtype=bool)
    excursions = np.zeros(n, dtype=int)
    squared_errors = np.zeros(n)
    max_errors = np.zeros(n)
    locked_time = np.zeros(n)
    energy = np.zeros(n)
    saturation_time = np.zeros(n)
    adc_clipping_time = np.zeros(n)
    initial_oma = np.abs(initial_bus.output_high_w - initial_bus.output_low_w)
    minimum_oma = initial_oma.copy()
    maximum_alias = np.zeros(n)
    record_index = 0
    for k in range(steps):
        observation = Observation(
            network.time_s, sample.timestamp_s, sample.codes.copy(), sample.valid, sample.clipped.copy()
        )
        command = np.asarray(algorithm.step(observation), dtype=float)
        if command.shape != (n,) or not np.all(np.isfinite(command)):
            raise ValueError("Controller must return one finite DAC code per ring")
        dt = min(r.sample_period_s, r.duration_s - network.time_s)
        interval_energy = np.zeros(n)
        for _ in range(r.physics_substeps):
            small_dt = dt / r.physics_substeps
            midpoint = network.time_s + small_dt / 2
            ambient, heat, shift, power_factor = config.disturbance.at(midpoint)
            driver.step(small_dt, command, network.heater_temperatures_k)
            current_bus = bus_response(config, network.ring_temperatures_k, shift, power_factor)
            external = np.zeros(t.n_states) if heat is None else heat.copy()
            if r.optical_heating:
                external += t.ring_map.T @ current_bus.absorbed_power_w
            network.step(small_dt, driver.interval_average_power_w, ambient, external)
            endpoint_bus = bus_response(config, network.ring_temperatures_k, shift, power_factor)
            sensor.step(small_dt, 0.5 * (current_bus.monitor_power_w + endpoint_bus.monitor_power_w))
            interval_energy += driver.interval_average_power_w * small_dt
        sample = sensor.sample()
        ambient, _, shift, factor = config.disturbance.at(network.time_s)
        bus = bus_response(config, network.ring_temperatures_k, shift, factor)
        resonance = np.array(
            [
                resonance_frequency(
                    p, network.ring_temperatures_k[i], (r.junction_low_v + r.junction_high_v) / 2
                )
                for i, p in enumerate(config.optical)
            ]
        )
        detuning = np.asarray(r.laser_frequency_hz) + shift - resonance
        # Track the assigned unwrapped resonance, so a wrong-FSR lock isn't silently correct.
        error = detuning - target_detuning
        locked = abs(error) <= r.lock_tolerance_hz
        excursions += previous_locked & ~locked
        previous_locked = locked
        squared_errors += error**2 * dt
        max_errors = np.maximum(max_errors, abs(error))
        locked_time += locked * dt
        energy += interval_energy
        saturation_time += ((command <= 0) | (command >= ctx.max_code)) * dt
        adc_clipping_time += sample.clipped * dt
        oma = abs(bus.output_high_w - bus.output_low_w)
        minimum_oma = np.minimum(minimum_oma, oma)
        diagonal_drop = np.diag(bus.drop_by_ring_and_carrier_w)
        alias = np.divide(
            bus.monitor_power_w - diagonal_drop,
            bus.monitor_power_w,
            out=np.zeros(n),
            where=bus.monitor_power_w > 0,
        )
        maximum_alias = np.maximum(maximum_alias, alias)
        if k % r.record_every == 0 or k == steps - 1:
            row = {
                "time_s": network.time_s,
                "ambient_temperature_k": ambient,
                "ring_temperature_k": network.ring_temperatures_k.copy(),
                "heater_power_w": driver.power_w.copy(),
                "command_codes": command.copy(),
                "adc_codes": sample.codes.copy(),
                "sample_time_s": sample.timestamp_s if sample.valid else -1.0,
                "detuning_hz": detuning,
                "tracking_error_hz": error,
                "oma_w": oma,
                "through_low_w": bus.output_low_w,
                "through_high_w": bus.output_high_w,
                "monitor_power_w": bus.monitor_power_w,
                "alias_fraction": alias,
            }
            for key, value in row.items():
                output[key][record_index] = value
            record_index += 1
        if progress and (k % max(1, steps // 100) == 0 or k == steps - 1):
            progress((k + 1) / steps)
    traces = output
    warnings = [
        "Uncalibrated illustrative plant: physical consistency is not Lightmatter device validation.",
        "OMA is quasi-static, duty-averaged WDM; BER and high-speed ISI are not computed.",
        "Flank target and nominal calibration depend on optical input power; intensity drift is ambiguous.",
    ]
    if config.controller.kind == "dither" and config.controller.dither_objective != "flank":
        warnings.append(
            "Dither's selected objective is peak/slope monitor, distinct from the configured initial flank "
            "and maximum OMA. Tracking metrics use the same initial flank for all controllers."
        )
    if np.max(maximum_alias) > 0.01:
        warnings.append(
            "Nonassigned carriers contribute >1% of a monitor reading; WDM alias/crosstalk present."
        )
    if np.any(adc_clipping_time):
        warnings.append("ADC clipping occurred; measurement information is lost.")
    if np.any(saturation_time):
        warnings.append("DAC rails were commanded; heater has finite one-sided authority.")
    metrics = {
        "tracking_rms_hz_per_lane": np.sqrt(squared_errors / r.duration_s).tolist(),
        "tracking_max_abs_hz_per_lane": max_errors.tolist(),
        "lock_fraction_per_lane": (locked_time / r.duration_s).tolist(),
        "lock_excursions_per_lane": excursions.tolist(),
        "heater_energy_j_per_lane": energy.tolist(),
        "dac_rail_fraction_per_lane": (saturation_time / r.duration_s).tolist(),
        "adc_clipped_fraction_per_lane": (adc_clipping_time / r.duration_s).tolist(),
        "minimum_oma_w_per_lane": minimum_oma.tolist(),
        "initial_oma_w_per_lane": initial_oma.tolist(),
        "target_detuning_hz_per_lane": target_detuning.tolist(),
        "max_alias_fraction_per_lane": maximum_alias.tolist(),
        "controller_steps": steps,
        "recorded_rows": len(traces["time_s"]),
        "runtime_s": time.monotonic() - start,
        "calibrated": bool(config.provenance.get("calibrated", False)),
        "ber": None,
        "evaluation_sampling": "controller endpoints; this is not a continuous-time excursion bound",
    }
    return ExperimentResult(config, traces, metrics, warnings)


def compare_controllers(config, kinds=("open_loop", "pi", "dither", "mpc"), context=None):
    """Reinitialize state and noise streams identically for each controller."""
    context = nominal_context(config) if context is None else context
    return {
        kind: run_experiment(
            replace(config, controller=replace(config.controller, kind=kind)), context=context
        )
        for kind in kinds
    }
