"""Continuous electrothermal plant separated from its sampled control interface.

The adaptive integrator recomputes optical absorption at every internal stage.
It therefore follows self-heating and branch history rather than imposing an
instantaneous thermal equilibrium. No true state is passed to a controller.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import time

import numpy as np
from scipy.integrate import solve_ivp

from .config import Configuration, load
from .controllers import Interface, Observation, create, snapshot, summary
from .optics import C0, optical_response, resonance_frequency
from .provenance import loaded_source_hashes
from .thermal import default_ring_network

ELECTRON_CHARGE_C = 1.602176634e-19
BOLTZMANN_EV_PER_K = 8.617333262145e-5


@dataclass
class OpticalState:
    monitor_power_w: np.ndarray
    absorbed_power_w: np.ndarray
    through_power_w: np.ndarray
    oma_w: np.ndarray
    detuning_hz: np.ndarray
    nearest_line: np.ndarray
    input_power_w: np.ndarray


@dataclass
class SimulationResult:
    configuration: Configuration
    trace: dict[str, np.ndarray]
    metrics: dict
    diagnostics: dict
    final_state: np.ndarray
    controller: object
    thermal_model: object
    junction_model: object
    controller_spec: dict | None
    source_sha256: dict[str, str]


class Plant:
    def __init__(self, configuration, *, thermal_network=None, junction_model=None):
        self.config = load(configuration)
        c, n = self.config, self.config.array.count
        self.n = n
        # Scenario comparisons and the scheduler must use the same binary
        # representation of their integer-tick boundaries. For example,
        # literal 10e-6 and 10*1e-6 need not be identical floating-point values.
        self.scenario_start_tick = c.ticks(c.scenario.start_s, "scenario_start")
        self.scenario_start_s = self.scenario_start_tick * c.simulation.tick_s
        self.triangle_half_period_ticks = None
        if c.scenario.kind == "local_triangle":
            self.triangle_half_period_ticks = c.ticks(c.scenario.aggressor_period_s / 2,
                                                       "aggressor half period")
            if self.triangle_half_period_ticks == 0:
                raise ValueError("aggressor half period must be at least one simulation tick")
        self.dropout_start_tick = (c.ticks(c.scenario.dropout_start_s, "dropout_start")
                                   if c.scenario.dropout_start_s >= 0 else -1)
        self.dropout_end_tick = self.dropout_start_tick + c.ticks(c.scenario.dropout_duration_s,
                                                                  "dropout_duration")
        self.thermal = thermal_network or default_ring_network(n, c.thermal)
        self.junction = junction_model or c.junction
        self.ring_observers = np.array([self.thermal.observer_names.index(f"ring_{i}") for i in range(n)])
        self.heater_nodes = np.array([self.thermal.node_names.index(f"heater_{i}") for i in range(n)])
        self.package_node = self.thermal.node_names.index("package")
        self.heater_sources = np.array([self.thermal.source_names.index(f"heater_{i}") for i in range(n)])
        self.absorption_sources = np.array([self.thermal.source_names.index(f"absorption_{i}") for i in range(n)])
        self.aggressor_sources = np.array([self.thermal.source_names.index(f"aggressor_{i}") for i in range(n)])
        self.lines_hz = c.ring.reference_frequency_hz + (np.arange(n) - (n - 1) / 2) * c.array.spacing_hz
        errors = np.zeros(n) if not c.array.resonance_errors_hz else np.asarray(c.array.resonance_errors_hz)
        cold = self.lines_hz + c.array.cold_offset_hz + errors
        self.rings = [replace(c.ring, reference_wavelength_m=C0 / f) for f in cold]
        self.applied_code = np.full(n, c.simulation.initial_heater_code, np.int64)
        self.target_voltage_v = self.applied_code / (2**c.heater.bits - 1) * c.heater.voltage_max_v
        self.frequency_noise_hz = np.zeros(n)
        self.power_multiplier = np.ones(n)
        self.rng_sensor = np.random.default_rng(np.random.SeedSequence([c.simulation.seed, 1]))
        self.rng_laser = np.random.default_rng(np.random.SeedSequence([c.simulation.seed, 2]))
        self.noise_current_a = np.zeros(n)
        self.rhs_calls = 0
        self.junction_levels = self.junction.evaluate(np.array([c.data.bias_v-c.data.swing_v/2,
                                                                c.data.bias_v+c.data.swing_v/2]))
        self.optical_table = None
        if c.data.averaging == "stationary_nrz" and c.simulation.optical_evaluation == "table":
            from .optical_tables import build_nrz_average_table
            self.optical_table = build_nrz_average_table(
                c.ring, resonance_shift_hz=tuple(self.junction_levels.resonance_shift_hz),
                additional_loss_per_m=tuple(self.junction_levels.additional_loss_per_m),
                bit_rate_hz=c.data.bit_rate_hz, mark_probability=c.data.mark_probability,
                absolute_tolerance=c.simulation.optical_table_tolerance, max_nodes=32769)
        probability = c.data.mark_probability
        points, weights = np.polynomial.legendre.leggauss(16)
        voltages = c.data.bias_v + c.data.swing_v/2*points
        charge_change = c.data.swing_v/2 * np.dot(weights, self.junction.evaluate(voltages).capacitance_f)
        # For nonlinear differential C(V), charge per transition is its
        # integral. A full up/down cycle dissipates ΔV*ΔQ, not C(Vbias)*ΔV².
        switching = c.data.swing_v*charge_change*c.data.bit_rate_hz*probability*(1-probability)
        leakage = ((1-probability)*self.junction_levels.electrical_heat_w[0]
                   + probability*self.junction_levels.electrical_heat_w[1])
        if c.data.averaging == "cw":
            switching = 0.
            leakage = self.junction.evaluate(c.data.bias_v).electrical_heat_w
        self.junction_heat_w = c.data.junction_heat_fraction*(switching+leakage)
        # State blocks: absolute temperatures, analog heater voltage, filtered
        # current, integrated heater energy, integrated optical OMA. The final
        # integral is an evaluator statistic, not deposited thermal energy.
        self.state = np.r_[np.full(self.thermal.n_nodes, c.simulation.initial_temperature_k),
                           self.target_voltage_v, np.zeros(n), np.zeros(n), np.zeros(n)]
        thermal = self.state[:self.thermal.n_nodes]
        initial = self.optical(0., thermal)
        self.state[self.thermal.n_nodes + n:self.thermal.n_nodes + 2*n] = self.photocurrent(thermal, initial)
        self.max_observed_temperature_k = c.simulation.initial_temperature_k

    def environment(self, t):
        s = self.config.scenario
        elapsed = max(0., t - self.scenario_start_s)
        active = t >= self.scenario_start_s
        ambient, frequency_delta, multiplier = s.ambient_temperature_k, 0., 1.
        powers = np.zeros(self.thermal.n_sources)
        if s.kind == "ambient_step" and active:
            ambient += s.ambient_step_k
        elif s.kind == "ambient_ramp":
            ambient += s.ambient_ramp_k_per_s * elapsed
        elif s.kind == "local_triangle" and active:
            period_s = 2*self.triangle_half_period_ticks*self.config.simulation.tick_s
            fraction = (elapsed / period_s) % 1.0
            triangle = 1 - abs(2 * fraction - 1)
            powers[self.aggressor_sources[s.aggressor_ring]] = s.aggressor_power_w * triangle
        elif s.kind == "laser_drift":
            frequency_delta = s.laser_drift_hz_per_s * elapsed
        elif s.kind == "power_step" and active:
            multiplier += s.power_step_fraction
        if ambient <= 0:
            raise ValueError("scenario predicts a nonpositive absolute boundary temperature")
        return ambient, frequency_delta, multiplier, powers

    def optical(self, t, temperature_k):
        c, n = self.config, self.n
        ring_t = self.thermal.observe(temperature_k)[self.ring_observers]
        _, drift, power_factor, _ = self.environment(t)
        frequencies = self.lines_hz + drift + self.frequency_noise_hz
        incoming = c.laser.power_per_line_w * power_factor * self.power_multiplier
        current = incoming.copy()
        monitor, heat, own_oma, own_input = (np.zeros(n) for _ in range(4))
        averages = np.empty((n, n))
        center_resonance = np.zeros(n)
        jr = self.junction_levels
        for i, rp in enumerate(self.rings):
            shifts, losses = jr.resonance_shift_hz, jr.additional_loss_per_m
            own_input[i] = current[i]
            if c.data.averaging == "stationary_nrz":
                from .waveform import wdm_nrz_average_response
                # Dynamic isolated-mode NRZ moments near resonance; exact
                # per-pass quasistatic tails far away, with a continuous passive
                # blend. This is an explicit approximation, not an exact
                # broadband time-domain ring. See OPTICAL_MODEL.md.
                if self.optical_table is not None:
                    out = self.optical_table.evaluate(frequencies-resonance_frequency(rp, ring_t[i]), current)
                else:
                    out = wdm_nrz_average_response(rp, frequencies, temperature_k=ring_t[i],
                                              resonance_shift_hz=tuple(shifts), additional_loss_per_m=tuple(losses),
                                              input_power_w=current, bit_rate_hz=c.data.bit_rate_hz,
                                              mark_probability=c.data.mark_probability)
                transmitted = np.asarray(out.through_power_w)
                monitor[i] = np.sum(out.drop_power_w)
                heat[i] = np.sum(out.absorbed_power_w)
                own_oma[i] = np.asarray(out.oma_w)[i]
            else:
                if c.data.averaging == "cw":
                    j = self.junction.evaluate(c.data.bias_v)
                    shifts = np.array([j.resonance_shift_hz]*2)
                    losses = np.array([j.additional_loss_per_m]*2)
                a = optical_response(rp, frequencies, temperature_k=ring_t[i], resonance_shift_hz=shifts[0],
                                     additional_loss_per_m=losses[0], input_power_w=current)
                b = optical_response(rp, frequencies, temperature_k=ring_t[i], resonance_shift_hz=shifts[1],
                                     additional_loss_per_m=losses[1], input_power_w=current)
                p = c.data.mark_probability
                transmitted = (1-p)*a.through_power_w + p*b.through_power_w
                monitor[i] = np.sum((1-p)*a.drop_power_w + p*b.drop_power_w)
                heat[i] = np.sum((1-p)*a.absorbed_power_w + p*b.absorbed_power_w)
                own_oma[i] = abs(b.through_power_w[i]-a.through_power_w[i])
            averages[i] = np.divide(transmitted, current, out=np.ones(n), where=current > 0)
            current = transmitted * 10**(-c.array.bus_loss_db/10)
            center_resonance[i] = resonance_frequency(rp, ring_t[i], np.mean(shifts))
        # Downstream rings attenuate this lane's conditional data levels. The
        # independent-lane mean-field approximation neglects inter-channel beats.
        oma = own_oma.copy()
        for i in range(n):
            oma[i] *= np.prod(averages[i+1:, i]) * 10**(-c.array.bus_loss_db/10 * (n-i))
        detuning, nearest = np.zeros(n), np.zeros(n, int)
        for i, rp in enumerate(self.rings):
            offsets = (center_resonance[i] - frequencies + rp.fsr_hz/2) % rp.fsr_hz - rp.fsr_hz/2
            detuning[i] = offsets[i]
            nearest[i] = int(np.argmin(abs(offsets)))
        return OpticalState(monitor, heat, current, oma, detuning, nearest, own_input)

    def photocurrent(self, temperature, optical):
        s = self.config.sensor
        # Default detector shares package temperature; unknown placement remains
        # an explicit simplification, not ring temperature mislabelled as PD.
        pd_t = temperature[self.package_node]
        exponent = s.dark_activation_energy_ev / BOLTZMANN_EV_PER_K * (1/s.reference_temperature_k - 1/pd_t)
        dark = s.dark_current_a * np.exp(exponent)
        return s.responsivity_a_per_w * s.collection_fraction * optical.monitor_power_w + dark

    def heater_power(self, temperature, voltage):
        h = self.config.heater
        resistance = h.resistance_ohm * (1 + h.resistance_temperature_coefficient_per_k *
                                       (temperature[self.heater_nodes] - h.reference_temperature_k))
        if np.any(resistance <= 0):
            raise ValueError("heater resistance model leaves its positive-resistance domain")
        actual_v = np.maximum(0., np.minimum(voltage, np.minimum(h.voltage_max_v,
                    np.minimum(h.current_max_a * resistance, np.sqrt(h.power_max_w*resistance)))))
        return actual_v**2 / resistance

    def heater_terminal_voltage(self, temperature, voltage):
        h = self.config.heater
        resistance = h.resistance_ohm * (1+h.resistance_temperature_coefficient_per_k *
                                       (temperature[self.heater_nodes]-h.reference_temperature_k))
        return np.sqrt(self.heater_power(temperature, voltage)*resistance)

    def rhs(self, t, state):
        self.rhs_calls += 1
        c, n, nt = self.config, self.n, self.thermal.n_nodes
        temp, voltage, filtered = state[:nt], state[nt:nt+n], state[nt+n:nt+2*n]
        if np.any(temp <= 0):
            raise ValueError("integration left the positive Kelvin domain")
        opt = self.optical(t, temp)
        ambient, _, _, power = self.environment(t)
        heat = self.heater_power(temp, voltage)
        power[self.heater_sources] += heat
        power[self.absorption_sources] += opt.absorbed_power_w
        power[self.absorption_sources] += self.junction_heat_w
        thermal_derivative = self.thermal.derivative(temp, power, ambient)
        thermal_derivative[self.package_node] += n*c.data.driver_power_w/self.thermal.capacity_j_per_k[self.package_node]
        return np.r_[thermal_derivative, (self.target_voltage_v-voltage)/c.heater.settling_time_s,
                     (self.photocurrent(temp, opt)-filtered)/c.sensor.filter_time_s, heat, opt.oma_w]

    def advance(self, start_tick, end_tick):
        c, nt, n = self.config, self.thermal.n_nodes, self.n
        if end_tick < start_tick:
            raise ValueError("integration interval must not go backwards")
        if end_tick == start_tick:
            return
        atol = np.r_[np.full(nt, c.simulation.temperature_atol_k), np.full(n, c.simulation.voltage_atol_v),
                     np.full(n, c.simulation.current_atol_a), np.full(n, c.simulation.heater_energy_atol_j),
                     np.full(n, c.simulation.oma_integral_atol_j)]
        start_s, end_s = start_tick*c.simulation.tick_s, end_tick*c.simulation.tick_s
        end_left_s = np.nextafter(end_s, start_s)

        def interval_rhs(t, state):
            # Collocation methods evaluate at the terminal point. That point
            # belongs to the pre-event flow; using the new boundary/source
            # there otherwise injects a finite spurious pre-event contribution.
            # The next interval starts at the exact right-limit event time.
            return self.rhs(min(t, end_left_s), state)

        solution = solve_ivp(interval_rhs, (start_s, end_s), self.state,
                             method=c.simulation.method, rtol=c.simulation.rtol, atol=atol,
                             max_step=c.simulation.max_step_s)
        if not solution.success:
            raise RuntimeError(f"electrothermal integration failed: {solution.message}")
        self.state = solution.y[:, -1]
        self.max_observed_temperature_k = max(self.max_observed_temperature_k, float(np.max(solution.y[:nt])))

    def sample(self, tick):
        c, n, nt = self.config, self.n, self.thermal.n_nodes
        s = c.sensor
        signal = self.state[nt+n:nt+2*n]
        rho = np.exp(-s.sample_period_s/s.filter_time_s)
        # One-sided input PSD through a first-order AFE. Its stationary variance
        # is PSD/(4*tau), and exact AR(1) innovations preserve that variance.
        psd = 2*ELECTRON_CHARGE_C*np.maximum(signal, 0) + s.input_noise_a_per_sqrt_hz**2
        sigma = np.sqrt(psd/(4*s.filter_time_s) * (-np.expm1(-2*s.sample_period_s/s.filter_time_s)))
        if s.noise_enabled:
            self.noise_current_a = rho*self.noise_current_a + sigma*self.rng_sensor.standard_normal(n)
        else:
            self.noise_current_a.fill(0)
        observed = signal+self.noise_current_a
        clipped = (observed < 0) | (observed > s.full_scale_a)
        code = np.rint(np.clip(observed/s.full_scale_a, 0, 1)*(2**s.bits-1)).astype(np.int64)
        temperature_code = int(round(self.state[self.package_node]/s.temperature_resolution_k))
        return code, clipped, temperature_code

    def action(self, codes):
        values = np.asarray(codes)
        if values.shape != (self.n,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"controller action must be an integer vector of shape {(self.n,)}")
        if np.any(values < 0) or np.any(values > 2**self.config.heater.bits-1):
            raise ValueError("controller action outside the DAC code range")
        self.applied_code = values.astype(np.int64, copy=True)
        self.target_voltage_v = values/(2**self.config.heater.bits-1)*self.config.heater.voltage_max_v


def simulate(configuration=None, controller=None, *, thermal_network=None, junction_model=None,
             controller_configuration=None):
    """Advance exactly between I/O events. Ordering: latch, sample, release, control.

    An action generated at an edge becomes visible to physics only after its
    compute+actuation delay. A zero-delay action affects the following interval,
    never the sample that caused it. Records and metrics are evaluator-only.

    Builtin controller constructor settings are captured before reset. An
    injected custom object can only be saved with an explicit importable
    controller_configuration (name and parameters); the caller is responsible
    for declaring equivalent constructor settings and reset behavior.
    """
    started = time.perf_counter()
    configuration = load(configuration)
    record_count = int(np.ceil(configuration.simulation.duration_s/configuration.simulation.record_period_s))+1
    # Python list/object overhead during recording is substantially larger than
    # final NumPy buffers. Reserve a conservative factor, before any allocation.
    recording_bytes = record_count*(11*configuration.array.count+4)*8*5
    if recording_bytes > configuration.simulation.record_memory_limit_mb*1e6:
        raise ValueError("estimated trace memory exceeds record_memory_limit_mb; increase record_period_s "
                         "or deliberately raise the limit after checking available RAM")
    plant = Plant(configuration, thermal_network=thermal_network, junction_model=junction_model)
    c, n, dt = plant.config, plant.n, plant.config.simulation.tick_s
    if controller is None:
        if controller_configuration is not None:
            raise ValueError("controller_configuration is only for an injected controller object")
        controller = create(c.controller["name"], c.controller.get("parameters"))
        controller_spec = snapshot(controller, c.controller)
    else:
        if controller_configuration is not None and (
                not isinstance(controller_configuration, dict)
                or set(controller_configuration) - {"name", "parameters"}
                or not isinstance(controller_configuration.get("parameters", {}), dict)):
            raise ValueError("controller_configuration must contain name and parameters")
        controller_spec = snapshot(controller, controller_configuration)
    source_snapshot = loaded_source_hashes()
    controller.reset(Interface(n, 2**c.heater.bits-1, 2**c.sensor.bits-1, c.simulation.control_period_s))
    end = c.ticks(c.simulation.duration_s)
    sample_period, control_period, record_period = (c.ticks(v) for v in
        (c.sensor.sample_period_s, c.simulation.control_period_s, c.simulation.record_period_s))
    act_delay = c.ticks(c.heater.latency_s + c.simulation.compute_latency_s)
    read_delay = c.ticks(c.sensor.latency_s)
    pending_actions, pending_samples = [], []
    next_sample, next_control, next_record = sample_period, 0, 0
    seq, tick, last_sample_tick, consumed_tick = 0, 0, -1, -1
    released = np.zeros(n, np.int64)
    saturated = np.zeros(n, bool)
    temp_code = 0
    records = {key: [] for key in ("time_s", "ring_temperature_k", "heater_temperature_k",
                                  "package_temperature_k", "ambient_temperature_k", "heater_voltage_v", "heater_power_w",
                                  "heater_code", "monitor_code", "sample_time_s", "detuning_hz", "oma_w",
                                  "absorbed_power_w", "through_power_w", "nearest_line")}
    nt = plant.thermal.n_nodes
    event_count, command_count = 0, 0
    clipped_sample_count = np.zeros(n, np.int64)
    wrong_line_s = np.zeros(n)
    saturation_s = np.zeros(n)
    maximum_metric_interval_s = 0.0
    initial_optical = plant.optical(0., plant.state[:nt])
    max_abs_detuning = abs(initial_optical.detuning_hz)
    discontinuities = [plant.scenario_start_tick] if plant.scenario_start_tick > 0 else []
    if c.scenario.kind == "local_triangle":
        discontinuities = list(range(plant.scenario_start_tick, end+1, plant.triangle_half_period_ticks))
    discontinuities = [v for v in discontinuities if v <= end]
    while True:
        while pending_actions and pending_actions[0][0] <= tick:
            _, _, code = heapq.heappop(pending_actions)
            plant.action(code)
        if tick == next_sample:
            code, clip, temperature_code = plant.sample(tick)
            clipped_sample_count += clip
            in_dropout = plant.dropout_start_tick >= 0 and plant.dropout_start_tick <= tick < plant.dropout_end_tick
            if not in_dropout:
                seq += 1
                heapq.heappush(pending_samples, (tick+read_delay, seq, tick, code, clip, temperature_code))
            # Noise is sampled and held with this known bandwidth, not sold as
            # optical linewidth or an arbitrary dt-independent white source.
            plant.frequency_noise_hz = c.laser.frequency_noise_rms_hz*plant.rng_laser.standard_normal(n)
            plant.power_multiplier = np.maximum(0., 1+c.laser.power_fractional_noise_rms*plant.rng_laser.standard_normal(n))
            next_sample += sample_period
        while pending_samples and pending_samples[0][0] <= tick:
            _, _, last_sample_tick, released, saturated, temp_code = heapq.heappop(pending_samples)
        if tick == next_control and tick < end:
            obs = Observation(tick*dt, last_sample_tick*dt, released.copy(), saturated.copy(),
                              plant.applied_code.copy(), temp_code, last_sample_tick > consumed_tick)
            action = np.asarray(controller.step(obs))
            if action.shape != (n,) or not np.issubdtype(action.dtype, np.integer) or np.any(action < 0) or np.any(action > 2**c.heater.bits-1):
                raise ValueError("invalid controller action: expected one in-range integer DAC code per ring")
            consumed_tick = last_sample_tick
            seq += 1
            if act_delay == 0:
                plant.action(action)
            else:
                heapq.heappush(pending_actions, (tick+act_delay, seq, action.copy()))
            command_count += 1
            next_control += control_period
        optical = plant.optical(tick*dt, plant.state[:nt])
        # Include both sides of any held laser-noise or scenario jump. These
        # are observed event extrema, still not continuous-time bounds.
        max_abs_detuning = np.maximum(max_abs_detuning, abs(optical.detuning_hz))
        if tick == next_record or tick == end:
            values = {"time_s": tick*dt, "ring_temperature_k": plant.thermal.observe(plant.state[:nt])[plant.ring_observers],
                      "heater_temperature_k": plant.state[plant.heater_nodes].copy(),
                      "package_temperature_k": float(plant.state[plant.package_node]),
                      "ambient_temperature_k": float(plant.environment(tick*dt)[0]),
                      "heater_voltage_v": plant.heater_terminal_voltage(plant.state[:nt], plant.state[nt:nt+n]),
                      "heater_power_w": plant.heater_power(plant.state[:nt], plant.state[nt:nt+n]),
                      "heater_code": plant.applied_code.copy(), "monitor_code": released.copy(),
                      "sample_time_s": last_sample_tick*dt, "detuning_hz": optical.detuning_hz,
                      "oma_w": optical.oma_w, "absorbed_power_w": optical.absorbed_power_w,
                      "through_power_w": optical.through_power_w, "nearest_line": optical.nearest_line}
            for key in records:
                records[key].append(values[key])
            next_record += record_period
        if tick == end:
            break
        candidates = [end, next_sample, next_control, next_record]
        if pending_actions:
            candidates.append(pending_actions[0][0])
        if pending_samples:
            candidates.append(pending_samples[0][0])
        candidates.extend(v for v in discontinuities if v > tick)
        next_tick = min(v for v in candidates if v > tick)
        old_voltage = plant.applied_code.copy()
        plant.advance(tick, next_tick)
        after = plant.optical(np.nextafter(next_tick*dt, tick*dt), plant.state[:nt])
        duration = (next_tick-tick)*dt
        maximum_metric_interval_s = max(maximum_metric_interval_s, duration)
        wrong_line_s += (after.nearest_line != np.arange(n))*duration
        saturation_s += ((old_voltage == 0) | (old_voltage == 2**c.heater.bits-1))*duration
        max_abs_detuning = np.maximum(max_abs_detuning, abs(after.detuning_hz))
        tick = next_tick
        event_count += 1
    trace = {key: np.asarray(value) for key, value in records.items()}
    duration = c.simulation.duration_s
    metrics = {"mean_oma_w": plant.state[nt+3*n:nt+4*n].copy()/duration, "wrong_line_time_s": wrong_line_s,
               "heater_rail_time_s": saturation_s, "heater_energy_j": plant.state[nt+2*n:nt+3*n].copy(),
               "max_abs_detuning_hz": max_abs_detuning, "final_detuning_hz": optical.detuning_hz,
               "maximum_temperature_k": plant.max_observed_temperature_k,
               "claim": "conditional synthetic plant; no measured Lightmatter transient validation",
               "metric_resolution_s": maximum_metric_interval_s,
               "metric_note": "OMA and heater energy use adaptive continuous integrals. Detuning extrema use both event sides; wrong-line time uses interval-end identity and can miss intra-interval crossings. Temperature extrema use accepted solver states. Neither extrema nor occupancy are continuous safety bounds. Heater rail time means DAC code rails, not all analog current/power limits.",
               "ber": None, "ber_note": "No receiver/noise calibration: OMA is not a BER prediction."}
    diagnostics = {"wall_s": time.perf_counter()-started, "rhs_calls": plant.rhs_calls,
                   "sensor_clipped_samples": clipped_sample_count,
                   "events": event_count, "commands": command_count, "config_sha256": c.digest(),
                   "numerical_method": c.simulation.method, "rtol": c.simulation.rtol,
                   "oma_integral_atol_j": c.simulation.oma_integral_atol_j,
                   "heater_energy_atol_j": c.simulation.heater_energy_atol_j,
                   "event_boundary_convention": "integer-tick canonical right limits; integration closes on the pre-event left limit",
                   "record_count": len(trace["time_s"]), "trace_bytes": sum(v.nbytes for v in trace.values()),
                   "junction_heat_w_per_ring": float(plant.junction_heat_w),
                   "optical_model": c.data.averaging,
                   "spectral_table": plant.optical_table.diagnostics if plant.optical_table is not None else None,
                   "stationary_nrz_note": "Photon memory near resonance, exact quasistatic tails, continuous blend; ideal electrical edges and iid bits.",
                   "fsr_hz": [rp.fsr_hz for rp in plant.rings],
                   "nearest_line_note": "Nearest periodic resonance is an identity-risk diagnostic even when no lane is locked."}
    diagnostics["controller"] = {"name": controller_spec["name"] if controller_spec else None,
                                 **summary(controller)}
    return SimulationResult(c, trace, metrics, diagnostics, plant.state.copy(), controller,
                            plant.thermal, plant.junction, controller_spec, source_snapshot)
