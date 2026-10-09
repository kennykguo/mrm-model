"""Sampled heater/monitor interfaces with SI units and real timestamp queues."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

ELEMENTARY_CHARGE_C = 1.602176634e-19


@dataclass
class ActuatorParams:
    dac_bits: int = 12
    dac_reference_v: float = 3.3
    min_voltage_v: float = 0.0
    max_voltage_v: float = 3.3
    heater_resistance_ohm: float = 500.0
    resistance_tcr_per_k: float = 2e-4
    reference_temperature_k: float = 300.0
    settling_time_s: float = 1e-6
    latency_s: float = 0.0
    slew_rate_v_per_s: float = float("inf")
    max_current_a: float = 0.02
    max_power_w: float = 0.02
    dac_gain: float = 1.0
    dac_offset_v: float = 0.0
    voltage_noise_std_v: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.dac_bits, bool)
            or int(self.dac_bits) != self.dac_bits
            or not 1 <= self.dac_bits <= 32
        ):
            raise ValueError("dac_bits must be an integer in [1,32]")
        for name in (
            "dac_reference_v",
            "heater_resistance_ohm",
            "reference_temperature_k",
            "max_current_a",
            "max_power_w",
            "dac_gain",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("settling_time_s", "latency_s", "voltage_noise_std_v"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            not np.isfinite(self.min_voltage_v)
            or not np.isfinite(self.max_voltage_v)
            or not 0 <= self.min_voltage_v <= self.max_voltage_v
        ):
            raise ValueError("voltage limits must satisfy 0 <= minimum <= maximum")
        if np.isnan(self.slew_rate_v_per_s) or self.slew_rate_v_per_s <= 0:
            raise ValueError("slew rate must be positive")
        if not np.isfinite(self.resistance_tcr_per_k) or not np.isfinite(self.dac_offset_v):
            raise ValueError("TCR and DAC offset must be finite")


def _vector(value: Any, n_channels: int, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.full(n_channels, float(values))
    if values.shape != (n_channels,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite vector of length {n_channels}")
    return values


class HeaterDriver:
    """DAC/driver with timestamped transport latency and slew-limited settling.

    ``step`` returns instantaneous endpoint heater power.  The exact integral
    over the interval (temperature held constant) is additionally exposed as
    ``interval_average_power_w`` for a finite-step thermal integrator.  Actual
    resistance is evaluated at the supplied heater temperature, separately from
    the optical ring temperature.  Command codes may be repeated every physics
    step; only changed codes enqueue a new command.
    """

    def __init__(self, params: ActuatorParams, n_channels: int = 1, rng: np.random.Generator | None = None):
        if int(n_channels) != n_channels or n_channels < 1:
            raise ValueError("n_channels must be a positive integer")
        self.params = self.p = params
        self.n_channels = int(n_channels)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.time_s = 0.0
        self.codes = np.zeros(self.n_channels, dtype=np.int64)
        self.command_codes = self.codes.copy()
        self.voltage_v = np.zeros(self.n_channels)
        self.resistance_ohm = np.full(self.n_channels, params.heater_resistance_ohm)
        self.current_a = np.zeros(self.n_channels)
        self.power_w = np.zeros(self.n_channels)
        self.interval_average_power_w = np.zeros(self.n_channels)
        self._target_codes = self.codes.copy()
        self._last_submitted: np.ndarray | None = None
        self._queue: deque[tuple[float, np.ndarray]] = deque()

    def initialize(
        self, command_codes: Any, heater_temperature_k: Any, voltage_v: Any | None = None
    ) -> np.ndarray:
        """Set accepted pre-existing t=0 command/output without a startup delay.

        Used for specified initial conditions, before time advances. Subsequent
        command changes still traverse the normal transport queue. With no
        explicit initial voltage, the driver starts settled at its actual
        compliance-limited DAC voltage.
        """
        if self.time_s != 0:
            raise ValueError("initialize is only available before time advances")
        codes = _vector(command_codes, self.n_channels, "command_codes")
        codes = np.rint(np.clip(codes, 0, 2**self.p.dac_bits - 1)).astype(np.int64)
        self._target_codes = codes.copy()
        self.codes = codes.copy()
        self._last_submitted = codes.copy()
        self._queue.clear()
        self.step(0, codes, heater_temperature_k)
        if voltage_v is None:
            self.voltage_v = self._voltage_target()
        else:
            voltage = _vector(voltage_v, self.n_channels, "voltage_v")
            if np.any(voltage < 0):
                raise ValueError("initial voltage must be nonnegative")
            self.voltage_v = voltage.copy()
        return self.step(0, codes, heater_temperature_k)

    def prime(self, initial_codes: Any, heater_temperature_k: Any) -> np.ndarray:
        """Initialize a settled, accepted command at t=0 without transport delay."""
        return self.initialize(initial_codes, heater_temperature_k)

    def _voltage_target(self) -> np.ndarray:
        p = self.p
        raw = p.dac_gain * self._target_codes / (2**p.dac_bits - 1) * p.dac_reference_v + p.dac_offset_v
        upper = np.minimum(
            p.max_voltage_v,
            np.minimum(p.max_current_a * self.resistance_ohm, np.sqrt(p.max_power_w * self.resistance_ohm)),
        )
        # Compliance takes precedence if it falls below the configured minimum.
        return np.minimum(np.maximum(raw, p.min_voltage_v), upper)

    @staticmethod
    def _advance_one(
        voltage: float, target: float, duration: float, tau: float, slew: float
    ) -> tuple[float, float]:
        """Return endpoint V and exact integral of V**2 for clipped first order ODE."""
        if duration <= 0:
            return (target if tau == 0 and np.isinf(slew) else voltage), 0.0
        difference = target - voltage
        if difference == 0:
            return target, target * target * duration
        integral = 0.0
        if np.isfinite(slew):
            threshold = tau * slew
            linear_duration = max(0.0, (abs(difference) - threshold) / slew)
            linear_duration = min(duration, linear_duration)
            rate = np.copysign(slew, difference)
            integral += (
                voltage * voltage * linear_duration
                + voltage * rate * linear_duration**2
                + rate * rate * linear_duration**3 / 3
            )
            voltage += rate * linear_duration
            duration -= linear_duration
        if duration <= 0:
            return voltage, integral
        if tau == 0:
            return target, integral + target * target * duration
        delta = voltage - target
        one_minus_decay = -np.expm1(-duration / tau)
        one_minus_decay_squared = -np.expm1(-2 * duration / tau)
        integral += (
            target * target * duration
            + 2 * target * delta * tau * one_minus_decay
            + delta * delta * tau / 2 * one_minus_decay_squared
        )
        return target + delta * np.exp(-duration / tau), integral

    def _advance(self, duration: float) -> np.ndarray:
        target = self._voltage_target()
        integrals = np.zeros(self.n_channels)
        for channel in range(self.n_channels):
            self.voltage_v[channel], integrals[channel] = self._advance_one(
                float(self.voltage_v[channel]),
                float(target[channel]),
                duration,
                self.p.settling_time_s,
                self.p.slew_rate_v_per_s,
            )
        return integrals / self.resistance_ohm

    def step(self, dt_s: float, command_codes: Any, heater_temperature_k: Any) -> np.ndarray:
        if not np.isfinite(dt_s) or dt_s < 0:
            raise ValueError("dt_s must be finite and nonnegative")
        codes = _vector(command_codes, self.n_channels, "command_codes")
        codes = np.rint(np.clip(codes, 0, 2**self.p.dac_bits - 1)).astype(np.int64)
        temperature = _vector(heater_temperature_k, self.n_channels, "heater_temperature_k")
        if np.any(temperature <= 0):
            raise ValueError("absolute heater temperatures must be positive")
        self.resistance_ohm = self.p.heater_resistance_ohm * (
            1 + self.p.resistance_tcr_per_k * (temperature - self.p.reference_temperature_k)
        )
        if np.any(self.resistance_ohm <= 0):
            raise ValueError("TCR model extrapolated to nonpositive heater resistance")
        self.command_codes = codes.copy()
        if self._last_submitted is None or not np.array_equal(codes, self._last_submitted):
            self._queue.append((self.time_s + self.p.latency_s, codes.copy()))
            self._last_submitted = codes.copy()
        upper = np.minimum(
            self.p.max_voltage_v,
            np.minimum(
                self.p.max_current_a * self.resistance_ohm, np.sqrt(self.p.max_power_w * self.resistance_ohm)
            ),
        )
        self.voltage_v = np.clip(self.voltage_v, 0, upper)
        end = self.time_s + dt_s
        energy = np.zeros(self.n_channels)
        while self._queue and self._queue[0][0] <= end:
            event_time, event_codes = self._queue.popleft()
            event_time = max(event_time, self.time_s)
            energy += self._advance(event_time - self.time_s)
            self.time_s = event_time
            self._target_codes = event_codes
            self.codes = event_codes.copy()
        energy += self._advance(end - self.time_s)
        self.time_s = end
        if self.p.voltage_noise_std_v:
            # This is explicitly per-output-update voltage noise, not a PSD.
            self.voltage_v = np.clip(
                self.voltage_v + self.rng.normal(0, self.p.voltage_noise_std_v, self.n_channels), 0, upper
            )
        self.current_a = self.voltage_v / self.resistance_ohm
        self.power_w = self.voltage_v * self.current_a
        self.interval_average_power_w = energy / dt_s if dt_s else self.power_w.copy()
        return self.power_w.copy()


@dataclass
class SensorParams:
    responsivity_a_per_w: float = 0.8
    dark_current_a: float = 1e-9
    input_noise_density_a_per_sqrt_hz: float = 2e-12
    bandwidth_hz: float = 2e5
    adc_bits: int = 14
    adc_full_scale_a: float = 1e-3
    adc_gain: float = 1.0
    adc_offset_a: float = 0.0
    latency_s: float = 0.0
    shot_noise: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.adc_bits, bool)
            or int(self.adc_bits) != self.adc_bits
            or not 1 <= self.adc_bits <= 32
        ):
            raise ValueError("adc_bits must be an integer in [1,32]")
        for name in ("responsivity_a_per_w", "bandwidth_hz", "adc_full_scale_a", "adc_gain"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("dark_current_a", "input_noise_density_a_per_sqrt_hz", "latency_s"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.adc_offset_a):
            raise ValueError("adc_offset_a must be finite")


@dataclass(frozen=True)
class SensorObservation:
    codes: np.ndarray
    current_a: np.ndarray
    timestamp_s: float
    delivered_time_s: float
    valid: bool
    clipped: np.ndarray


class MonitorADC:
    """Photodiode, exact one-pole noise/filter update, ADC and transport delay.

    Noise densities are one-sided input-current PSDs.  A first-order low-pass
    with 3 dB bandwidth B has equivalent noise bandwidth pi*B/2; the Gaussian
    filtered shot-noise approximation is suitable at substantial photon rates.
    ``step`` updates analog physics. ``sample`` acquires at current time and
    returns the latest *delivered* sample; until delivery it returns valid=False.
    Repeated delayed observations retain their original acquisition timestamp.
    """

    def __init__(self, params: SensorParams, n_channels: int = 1, rng: np.random.Generator | None = None):
        if int(n_channels) != n_channels or n_channels < 1:
            raise ValueError("n_channels must be a positive integer")
        self.params = self.p = params
        self.n_channels = int(n_channels)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.time_s = 0.0
        self.filtered_current_a = np.full(self.n_channels, params.dark_current_a)
        self._queue: deque[SensorObservation] = deque()
        self._latest = SensorObservation(
            np.zeros(self.n_channels, dtype=np.int64),
            np.zeros(self.n_channels),
            float("nan"),
            float("nan"),
            False,
            np.zeros(self.n_channels, dtype=bool),
        )

    def _deliver_ready(self) -> None:
        while self._queue and self._queue[0].delivered_time_s <= self.time_s + 1e-15:
            self._latest = self._queue.popleft()

    def step(self, dt_s: float, optical_power_w: Any) -> None:
        if not np.isfinite(dt_s) or dt_s < 0:
            raise ValueError("dt_s must be finite and nonnegative")
        optical_power = _vector(optical_power_w, self.n_channels, "optical_power_w")
        if np.any(optical_power < 0):
            raise ValueError("optical power must be nonnegative")
        current = self.p.responsivity_a_per_w * optical_power + self.p.dark_current_a
        decay = np.exp(-2 * np.pi * self.p.bandwidth_hz * dt_s)
        self.filtered_current_a = current + (self.filtered_current_a - current) * decay
        density_squared = np.full(self.n_channels, self.p.input_noise_density_a_per_sqrt_hz**2)
        if self.p.shot_noise:
            density_squared += 2 * ELEMENTARY_CHARGE_C * current
        variance = (
            density_squared
            * (np.pi * self.p.bandwidth_hz / 2)
            * (-np.expm1(-4 * np.pi * self.p.bandwidth_hz * dt_s))
        )
        if np.any(variance):
            self.filtered_current_a += self.rng.normal(size=self.n_channels) * np.sqrt(variance)
        self.time_s += dt_s
        self._deliver_ready()

    def sample(self) -> SensorObservation:
        current = self.p.adc_gain * self.filtered_current_a + self.p.adc_offset_a
        clipped = (current < 0) | (current > self.p.adc_full_scale_a)
        max_code = 2**self.p.adc_bits - 1
        codes = np.rint(np.clip(current / self.p.adc_full_scale_a, 0, 1) * max_code).astype(np.int64)
        reconstructed = (codes / max_code * self.p.adc_full_scale_a - self.p.adc_offset_a) / self.p.adc_gain
        observation = SensorObservation(
            codes, reconstructed, self.time_s, self.time_s + self.p.latency_s, True, clipped
        )
        self._queue.append(observation)
        self._deliver_ready()
        return self.latest_observation

    @property
    def latest_observation(self) -> SensorObservation:
        sample = self._latest
        return SensorObservation(
            sample.codes.copy(),
            sample.current_a.copy(),
            sample.timestamp_s,
            sample.delivered_time_s,
            sample.valid,
            sample.clipped.copy(),
        )
