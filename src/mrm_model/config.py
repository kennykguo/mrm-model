"""One strict, serializable configuration for scripts, the CLI and the lab UI.

All physical quantities are SI. Provenance is data, not an implied accuracy claim.
"""

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np

from .controllers import ControllerParams
from .hardware import ActuatorParams, SensorParams
from .optics import OpticalParams
from .thermal import ThermalParams, default_thermal_params


def strict_dataclass(cls, values):
    if not isinstance(values, dict):
        raise TypeError(f"{cls.__name__} must be an object")
    allowed = {f.name for f in fields(cls) if f.init}
    unexpected = set(values) - allowed
    if unexpected:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unexpected)}")
    return cls(**values)


@dataclass(frozen=True)
class RunParams:
    duration_s: float = 0.01
    sample_period_s: float = 10e-6
    physics_substeps: int = 4
    record_every: int = 1
    seed: int = 7
    input_power_w: float = 100e-6
    laser_frequency_hz: tuple[float, ...] = (299792458 / 1310e-9,)
    initial_codes: tuple[float, ...] = (2048,)
    initial_temperature_k: tuple[float, ...] | None = None
    initial_state_rise_k: tuple[float, ...] | None = None
    # NRZ average monitor is weighted over two static voltage states. Waveform mode
    # provides the distinct photon/RC calculation, not an invented fast BER trace.
    junction_low_v: float = 0.0
    junction_high_v: float = -1.0
    data_one_fraction: float = 0.5
    optical_heating: bool = True
    bus_loss_db_between_rings: float = 0.0
    lock_tolerance_hz: float = 5e9
    target_detuning_hz: tuple[float, ...] | None = None
    max_steps: int = 1_000_000
    max_recorded_rows: int = 200_000
    max_trace_bytes: int = 256_000_000

    def __post_init__(self):
        if not np.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration_s must be finite and positive")
        if not np.isfinite(self.sample_period_s) or self.sample_period_s <= 0:
            raise ValueError("sample_period_s must be finite and positive")
        if not 1 <= self.physics_substeps <= 100 or self.record_every < 1:
            raise ValueError("physics_substeps must be in [1,100]; record_every must be positive")
        if self.max_steps < 1 or self.max_recorded_rows < 1 or self.max_trace_bytes < 1:
            raise ValueError("resource limits must be positive")
        steps = int(np.ceil(self.duration_s / self.sample_period_s))
        if steps > self.max_steps or steps // self.record_every + 2 > self.max_recorded_rows:
            raise ValueError("run exceeds configured work or recording limits; reduce duration or decimate")
        if not 0 <= self.data_one_fraction <= 1:
            raise ValueError("data_one_fraction must lie in [0,1]")
        if not np.isfinite(self.input_power_w) or self.input_power_w < 0:
            raise ValueError("input_power_w must be finite and nonnegative")
        if self.bus_loss_db_between_rings < 0 or self.lock_tolerance_hz <= 0:
            raise ValueError("bus loss must be nonnegative and lock tolerance positive")
        for name in ("junction_low_v", "junction_high_v", "bus_loss_db_between_rings", "lock_tolerance_hz"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("laser_frequency_hz", "initial_codes"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a finite vector")
        if np.any(np.asarray(self.laser_frequency_hz) <= 0):
            raise ValueError("laser frequencies must be positive")
        if self.initial_temperature_k is not None and self.initial_state_rise_k is not None:
            raise ValueError("provide either nodal temperature or generalized rise state, not both")
        for name in ("initial_temperature_k", "initial_state_rise_k"):
            value = getattr(self, name)
            if value is not None and (np.asarray(value).ndim != 1 or not np.all(np.isfinite(value))):
                raise ValueError(f"{name} must be a finite vector")


@dataclass(frozen=True)
class DisturbanceParams:
    kind: str = "ambient_ramp"
    start_s: float = 0.002
    ambient_temperature_k: float = 298.15
    ambient_amplitude_k: float = 2.0
    ramp_rate_k_per_s: float = 808.0
    period_s: float = 0.1
    laser_drift_hz_per_s: float = 0.0
    laser_power_fraction_amplitude: float = 0.0
    # Nodal watts or signed projected heat loads for a generalized thermal ROM.
    external_heat_peak_w: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.kind not in {"constant", "ambient_ramp", "ambient_step", "ambient_triangle", "local_heat"}:
            raise ValueError(f"Unknown disturbance {self.kind}")
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise ValueError(f"{f.name} must be finite")
        if self.ambient_temperature_k <= 0 or self.start_s < 0 or self.period_s <= 0:
            raise ValueError("invalid disturbance temperature or timing")
        if self.ambient_amplitude_k < 0 or self.ramp_rate_k_per_s < 0:
            raise ValueError("thermal amplitudes/rates must be nonnegative")
        if not 0 <= self.laser_power_fraction_amplitude <= 1:
            raise ValueError("laser power fraction amplitude must lie in [0,1]")
        if self.external_heat_peak_w is not None:
            heat = np.asarray(self.external_heat_peak_w, dtype=float)
            if heat.ndim != 1 or not np.all(np.isfinite(heat)):
                raise ValueError("external heat must be a finite vector")

    def at(self, time_s):
        elapsed = max(0.0, time_s - self.start_s)
        active = time_s >= self.start_s
        triangle = 1 - abs(2 * ((elapsed / self.period_s) % 1) - 1) if active else 0.0
        ambient = self.ambient_temperature_k
        if self.kind == "ambient_ramp":
            ambient += min(self.ambient_amplitude_k, self.ramp_rate_k_per_s * elapsed)
        elif self.kind == "ambient_step" and active:
            ambient += self.ambient_amplitude_k
        elif self.kind == "ambient_triangle":
            ambient += self.ambient_amplitude_k * triangle
        heat = None
        if self.kind == "local_heat" and self.external_heat_peak_w is not None:
            heat = np.asarray(self.external_heat_peak_w) * triangle
        frequency_shift = self.laser_drift_hz_per_s * elapsed
        power_fraction = 1 + self.laser_power_fraction_amplitude * np.sin(2 * np.pi * elapsed / self.period_s)
        return ambient, heat, frequency_shift, power_fraction


@dataclass(frozen=True)
class ExperimentConfig:
    optical: tuple[OpticalParams, ...]
    thermal: ThermalParams
    actuator: ActuatorParams
    sensor: SensorParams
    run: RunParams
    disturbance: DisturbanceParams
    controller: ControllerParams
    provenance: dict
    schema_version: int = 1

    def __post_init__(self):
        n = len(self.optical)
        if self.schema_version != 1:
            raise ValueError("Unsupported schema_version")
        if not 1 <= n <= 32:
            raise ValueError("1 to 32 rings are supported")
        steps = int(np.ceil(self.run.duration_s / self.run.sample_period_s))
        rows = (steps - 1) // self.run.record_every + 1 + int((steps - 1) % self.run.record_every != 0)
        if rows * (3 + 11 * n) * 8 > self.run.max_trace_bytes:
            raise ValueError("trace arrays exceed max_trace_bytes; increase record_every to bound RAM")
        if len(self.run.laser_frequency_hz) != n or len(self.run.initial_codes) != n:
            raise ValueError("one assigned laser frequency and initial DAC code are required per ring")
        if (
            np.asarray(self.thermal.heater_map).shape[1] != n
            or np.asarray(self.thermal.ring_map).shape[0] != n
        ):
            raise ValueError("thermal heater/ring counts must match optical count")
        if np.any(np.asarray(self.run.initial_codes) < 0) or np.any(
            np.asarray(self.run.initial_codes) > 2**self.actuator.dac_bits - 1
        ):
            raise ValueError("initial DAC codes outside actuator range")
        nodes = len(self.thermal.capacitance_j_per_k)
        if nodes > 256:
            raise ValueError("compact thermal models are limited to 256 states to bound dense-matrix RAM")
        if self.run.initial_temperature_k is not None and len(self.run.initial_temperature_k) != nodes:
            raise ValueError("initial_temperature_k needs one value per thermal state")
        if self.run.initial_temperature_k is not None and not self.thermal.physical_network:
            raise ValueError("modal ROM initial conditions use initial_state_rise_k, not nodal temperatures")
        if self.run.initial_state_rise_k is not None and len(self.run.initial_state_rise_k) != nodes:
            raise ValueError("initial_state_rise_k needs one generalized coordinate per thermal state")
        if self.run.target_detuning_hz is not None and (
            len(self.run.target_detuning_hz) != n or not np.all(np.isfinite(self.run.target_detuning_hz))
        ):
            raise ValueError("target_detuning_hz must be one finite frequency offset per assigned ring")
        if (
            self.disturbance.external_heat_peak_w is not None
            and len(self.disturbance.external_heat_peak_w) != nodes
        ):
            raise ValueError("external_heat_peak_w needs one value per thermal state")
        if (
            self.thermal.physical_network
            and self.disturbance.external_heat_peak_w is not None
            and np.any(np.asarray(self.disturbance.external_heat_peak_w) < 0)
        ):
            raise ValueError("physical nodal external heat must be nonnegative; modal loads may be signed")

    def to_dict(self):
        def clean(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (tuple, list)):
                return [clean(x) for x in value]
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, float) and np.isinf(value):
                # JSON has no infinity. null denotes no slew limit in actuator config.
                return None
            return value

        return clean(asdict(self))

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    @classmethod
    def from_dict(cls, values):
        allowed = {f.name for f in fields(cls)}
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError(
                f"Unknown configuration fields: {set(values) - allowed if isinstance(values, dict) else values}"
            )
        values = values.copy()
        values["optical"] = tuple(strict_dataclass(OpticalParams, p) for p in values["optical"])
        for key, typ in (
            ("thermal", ThermalParams),
            ("actuator", ActuatorParams),
            ("sensor", SensorParams),
            ("run", RunParams),
            ("disturbance", DisturbanceParams),
            ("controller", ControllerParams),
        ):
            item = values[key].copy()
            if key == "actuator" and item.get("slew_rate_v_per_s", 1) is None:
                item["slew_rate_v_per_s"] = float("inf")
            values[key] = strict_dataclass(typ, item)
        return cls(**values)

    @classmethod
    def from_json(cls, value):
        return cls.from_dict(json.loads(value))

    @classmethod
    def load(cls, path):
        return cls.from_json(Path(path).read_text())


def default_config(n_rings=1):
    """Synthetic device corners at a known flank, never fitted Lightmatter hardware."""
    if not 1 <= n_rings <= 32:
        raise ValueError("n_rings must be between 1 and 32")
    thermal = default_thermal_params(n_rings)
    actuator = ActuatorParams(
        dac_bits=16,
        dac_reference_v=1.6,
        max_voltage_v=1.6,
        heater_resistance_ohm=56,
        resistance_tcr_per_k=0.001,
        reference_temperature_k=298.15,
        max_current_a=0.03,
        max_power_w=0.04,
        settling_time_s=2e-6,
    )
    sensor = SensorParams(
        adc_bits=16, adc_full_scale_a=100e-6, bandwidth_hz=200e3, responsivity_a_per_w=0.8, latency_s=0
    )
    laser = 299792458 / 1310e-9 + (np.arange(n_rings) - (n_rings - 1) / 2) * 400e9
    codes = np.full(n_rings, 20000.0)
    temperature = np.full(len(thermal.capacitance_j_per_k), 298.15)
    # Self-consistent illustrative electrical/thermal equilibrium for initial bias.
    for _ in range(100):
        heater_t = np.asarray(thermal.heater_temperature_map) @ temperature
        resistance = actuator.heater_resistance_ohm * (
            1 + actuator.resistance_tcr_per_k * (heater_t - actuator.reference_temperature_k)
        )
        power = (actuator.dac_reference_v * codes / (2**actuator.dac_bits - 1)) ** 2 / resistance
        rise = np.linalg.solve(
            thermal.conductance_w_per_k,
            thermal.heater_map @ power
            + thermal.boundary_map_w_per_k
            @ np.full(thermal.boundary_map_w_per_k.shape[1], 298.15 - thermal.reference_temperature_k),
        )
        new_temperature = thermal.reference_temperature_k + rise
        if np.max(abs(new_temperature - temperature)) < 1e-8:
            temperature = new_temperature
            break
        temperature = 0.5 * (temperature + new_temperature)
    ring_t = np.asarray(thermal.ring_map) @ temperature
    optical = []
    for i in range(n_rings):
        p = OpticalParams()
        f0 = laser[i] - 25e9 - p.df_dtemperature_hz_per_k * (ring_t[i] - p.reference_temperature_k)
        optical.append(replace(p, reference_frequency_hz=float(f0)))
    return ExperimentConfig(
        tuple(optical),
        thermal,
        actuator,
        sensor,
        RunParams(laser_frequency_hz=tuple(laser), initial_codes=tuple(codes)),
        DisturbanceParams(),
        ControllerParams(),
        {
            "profile": "synthetic G26-inspired controller laboratory; uncalibrated",
            "public_constraints": {
                "center_wavelength_m": 1310e-9,
                "per_direction_spacing_hz": 400e9,
                "df_dtemperature_magnitude_hz_per_k": 15e9,
                "nrz_bit_rate_bps": 50e9,
            },
            "source": "Anand et al., OFC 2026 M4B.3, supplied PDF",
            "assumptions": "Every other numeric device/interface value is illustrative. The heater 56 ohm "
            "hypothesis resembles G25 textbook read-off, not a G26 specification. "
            "Reference resonance is synthetically aligned at a chosen initial heater bias.",
            "calibrated": False,
        },
    )
