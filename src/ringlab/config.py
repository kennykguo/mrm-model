"""Strict experiment configuration. All physical quantities use explicit SI units.

Resolved configurations contain every default, so results never depend on a
hidden value. Evidence labels describe data provenance, not numerical accuracy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import copy
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import yaml


@dataclass(frozen=True)
class ArraySettings:
    count: int = 1
    spacing_hz: float = 400e9
    cold_offset_hz: float = 300e9
    bus_loss_db: float = 0.0
    resonance_errors_hz: tuple = ()


@dataclass(frozen=True)
class HeaterSettings:
    resistance_ohm: float = 500.0
    resistance_temperature_coefficient_per_k: float = .001
    reference_temperature_k: float = 300.0
    voltage_max_v: float = .6
    current_max_a: float = .02
    power_max_w: float = .002
    bits: int = 12
    settling_time_s: float = 1e-6
    latency_s: float = 2e-6


@dataclass(frozen=True)
class SensorSettings:
    responsivity_a_per_w: float = .8
    collection_fraction: float = 1.0
    dark_current_a: float = 5e-9
    dark_activation_energy_ev: float = .35
    reference_temperature_k: float = 300.0
    filter_time_s: float = 2e-6
    input_noise_a_per_sqrt_hz: float = 2e-12
    full_scale_a: float = 200e-6
    bits: int = 12
    sample_period_s: float = 5e-6
    latency_s: float = 2e-6
    temperature_resolution_k: float = .05
    noise_enabled: bool = True


@dataclass(frozen=True)
class DataSettings:
    bias_v: float = 1.5
    swing_v: float = 1.0
    bit_rate_hz: float = 50e9
    mark_probability: float = .5
    averaging: str = "stationary_nrz"
    junction_heat_fraction: float = 1.0
    driver_power_w: float = 0.0


@dataclass(frozen=True)
class LaserSettings:
    power_per_line_w: float = .5e-3
    power_fractional_noise_rms: float = 0.0
    frequency_noise_rms_hz: float = 0.0


@dataclass(frozen=True)
class SimulationSettings:
    duration_s: float = .004
    tick_s: float = 1e-6
    control_period_s: float = 20e-6
    compute_latency_s: float = 1e-6
    record_period_s: float = 20e-6
    max_step_s: float = 5e-6
    rtol: float = 1e-6
    temperature_atol_k: float = 1e-7
    voltage_atol_v: float = 1e-8
    current_atol_a: float = 1e-12
    heater_energy_atol_j: float = 1e-15
    oma_integral_atol_j: float = 1e-18
    record_memory_limit_mb: float = 256.0
    optical_evaluation: str = "table"
    optical_table_tolerance: float = 1e-7
    method: str = "Radau"
    seed: int = 1
    initial_temperature_k: float = 300.0
    initial_heater_code: int = 0


@dataclass(frozen=True)
class ScenarioSettings:
    kind: str = "constant"
    ambient_temperature_k: float = 300.0
    start_s: float = .002
    ambient_step_k: float = 0.0
    ambient_ramp_k_per_s: float = 0.0
    aggressor_power_w: float = 0.0
    aggressor_period_s: float = .2
    aggressor_ring: int = 0
    laser_drift_hz_per_s: float = 0.0
    power_step_fraction: float = 0.0
    dropout_start_s: float = -1.0
    dropout_duration_s: float = 0.0


SECTIONS = {"array": ArraySettings, "heater": HeaterSettings, "sensor": SensorSettings,
            "data": DataSettings, "laser": LaserSettings, "simulation": SimulationSettings,
            "scenario": ScenarioSettings}


def _construct(cls, values, section):
    if not isinstance(values, dict):
        raise ValueError(f"{section} must be a mapping")
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {section} keys: {sorted(unknown)}")
    # PyYAML's YAML-1.1 parser treats 50e9 as text; SI scientific notation is
    # accepted deliberately for numeric fields, but arbitrary strings are not.
    normalized = dict(values)
    for f in fields(cls):
        if f.name not in normalized:
            continue
        value = normalized[f.name]
        kind = f.type if isinstance(f.type, str) else f.type.__name__
        label = f"{section}.{f.name}"
        if kind == "float":
            if isinstance(value, str):
                if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
                    raise ValueError(f"{label} must be numeric")
                value = float(value)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{label} must be a finite real number")
            normalized[f.name] = float(value)
        elif kind == "int":
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{label} must be an integer")
            normalized[f.name] = int(value)
        elif kind == "bool":
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{label} must be boolean")
            normalized[f.name] = bool(value)
        elif kind == "str":
            if not isinstance(value, str):
                raise ValueError(f"{label} must be text")
        elif kind == "tuple":
            if not isinstance(value, (tuple, list)) or any(
                    isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not np.isfinite(v)
                    for v in value):
                raise ValueError(f"{label} must be a sequence of finite real numbers")
            normalized[f.name] = tuple(float(v) for v in value)
    return cls(**normalized)


def _set(tree, path, value):
    parts = path.split(".")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"{path}: parent is not a mapping")
    node[parts[-1]] = value


def _flatten(tree, prefix=""):
    for k, v in tree.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _flatten(v, path)
        else:
            yield path, v


class Configuration:
    def __init__(self, raw):
        from .optics import RingParameters
        from .junction import DepletionJunction
        from .thermal import AssumedThermalParameters
        allowed = set(SECTIONS) | {"version", "name", "ring", "junction", "thermal", "controller", "evidence"}
        if set(raw) - allowed:
            raise ValueError(f"unknown sections: {sorted(set(raw) - allowed)}")
        if isinstance(raw.get("version", 1), bool) or raw.get("version", 1) != 1:
            raise ValueError("configuration version must be 1")
        self.name = raw.get("name", "synthetic_reference")
        if not isinstance(self.name, str):
            raise ValueError("name must be text")
        for name, cls in SECTIONS.items():
            setattr(self, name, _construct(cls, raw.get(name, {}), name))
        self.ring = _construct(RingParameters, raw.get("ring", {}), "ring")
        self.junction = _construct(DepletionJunction, raw.get("junction", {}), "junction")
        self.thermal = _construct(AssumedThermalParameters, raw.get("thermal", {}), "thermal")
        self.controller = copy.deepcopy(raw.get("controller", {"name": "scan_pi", "parameters": {}}))
        if not isinstance(self.controller, dict):
            raise ValueError("controller must be a mapping")
        if set(self.controller) - {"name", "parameters"}:
            raise ValueError("controller only supports name and parameters")
        if not isinstance(self.controller.get("name"), str) or not self.controller["name"].strip():
            raise ValueError("controller.name must be nonempty text")
        if not isinstance(self.controller.get("parameters", {}), dict):
            raise ValueError("controller.parameters must be a mapping")
        self.evidence = copy.deepcopy(raw.get("evidence", {}))
        if not isinstance(self.evidence, dict) or any(
                not isinstance(path, str) or not isinstance(item, dict)
                or not isinstance(item.get("status"), str) or not isinstance(item.get("source"), str)
                for path, item in self.evidence.items()):
            raise ValueError("evidence must map parameter paths to status/source text mappings")
        self.validate()

    def validate(self):
        if isinstance(self.array.count, bool) or int(self.array.count) != self.array.count or not 1 <= self.array.count <= 32:
            raise ValueError("array.count must be an integer in [1,32]")
        if self.array.spacing_hz <= 0 or self.array.bus_loss_db < 0:
            raise ValueError("array spacing must be positive and bus loss nonnegative")
        for section, cls in SECTIONS.items():
            for field in fields(cls):
                value = getattr(getattr(self, section), field.name)
                if isinstance(value, (int, float)) and not np.isfinite(value):
                    raise ValueError(f"{section}.{field.name} must be finite")
        positive = {"heater": ("resistance_ohm", "reference_temperature_k", "voltage_max_v", "current_max_a",
                               "power_max_w", "settling_time_s"),
                    "sensor": ("responsivity_a_per_w", "filter_time_s", "full_scale_a", "sample_period_s",
                               "temperature_resolution_k", "reference_temperature_k"),
                    "simulation": ("duration_s", "tick_s", "control_period_s", "record_period_s", "max_step_s",
                                   "rtol", "temperature_atol_k", "voltage_atol_v", "current_atol_a",
                                   "heater_energy_atol_j", "oma_integral_atol_j",
                                   "initial_temperature_k", "record_memory_limit_mb", "optical_table_tolerance"), "data": ("bit_rate_hz",)}
        for section, keys in positive.items():
            for key in keys:
                if getattr(getattr(self, section), key) <= 0:
                    raise ValueError(f"{section}.{key} must be positive")
        for key in ("heater", "sensor"):
            bits = getattr(self, key).bits
            if isinstance(bits, bool) or int(bits) != bits or not 2 <= bits <= 24:
                raise ValueError(f"{key}.bits must be integer in [2,24]")
        if not 0 <= self.data.mark_probability <= 1 or self.data.swing_v < 0:
            raise ValueError("invalid NRZ probability or swing")
        if not 0 <= self.data.junction_heat_fraction <= 1 or self.data.driver_power_w < 0:
            raise ValueError("invalid junction heat fraction or driver power")
        junction_levels = self.junction.evaluate(np.array([self.data.bias_v - self.data.swing_v / 2,
                                                         self.data.bias_v + self.data.swing_v / 2]))
        if np.any(self.ring.intrinsic_loss_per_m + junction_levels.additional_loss_per_m < 0):
            raise ValueError("junction data levels predict negative total optical loss")
        if self.data.averaging not in ("stationary_nrz", "quasistatic", "cw"):
            raise ValueError("data.averaging must be stationary_nrz, quasistatic or cw")
        if self.scenario.kind not in ("constant", "ambient_step", "ambient_ramp", "local_triangle", "laser_drift", "power_step"):
            raise ValueError("unknown scenario kind")
        if self.scenario.aggressor_period_s <= 0 or self.scenario.aggressor_power_w < 0:
            raise ValueError("aggressor period must be positive and power nonnegative")
        if not 0 <= self.scenario.aggressor_ring < self.array.count:
            raise ValueError("aggressor_ring out of range")
        if not 0 <= self.sensor.collection_fraction <= 1 or self.sensor.dark_current_a < 0 or self.sensor.input_noise_a_per_sqrt_hz < 0:
            raise ValueError("invalid detector collection or noise")
        if self.laser.power_per_line_w < 0 or self.scenario.power_step_fraction < -1:
            raise ValueError("optical powers cannot be negative")
        if self.laser.power_fractional_noise_rms < 0 or self.laser.frequency_noise_rms_hz < 0:
            raise ValueError("noise RMS values cannot be negative")
        if self.scenario.ambient_temperature_k <= 0 or self.scenario.dropout_duration_s < 0:
            raise ValueError("invalid scenario boundary temperature or dropout duration")
        if isinstance(self.simulation.seed, bool) or int(self.simulation.seed) != self.simulation.seed or self.simulation.seed < 0:
            raise ValueError("simulation.seed must be a nonnegative integer")
        if self.simulation.method not in ("Radau", "BDF", "DOP853"):
            raise ValueError("method must be Radau, BDF or DOP853")
        if self.simulation.optical_evaluation not in ("table", "direct"):
            raise ValueError("optical_evaluation must be table or direct")
        if self.data.averaging == "stationary_nrz" and self.ring.beta2_s2_per_m != 0:
            raise ValueError("stationary_nrz wideband optics does not implement nonzero beta2; use quasistatic or cw")
        errors = self.array.resonance_errors_hz
        if len(errors) not in (0, self.array.count) or not np.all(np.isfinite(errors)):
            raise ValueError("resonance_errors_hz must have one value per ring")
        for label, seconds in self.times().items():
            self.ticks(seconds, label)
        if self.simulation.control_period_s < self.sensor.sample_period_s:
            raise ValueError("control_period_s must be at least sample_period_s")
        if not 0 <= self.simulation.initial_heater_code <= 2**self.heater.bits - 1:
            raise ValueError("initial_heater_code outside DAC range")

    def times(self):
        values = {"duration": self.simulation.duration_s, "control_period": self.simulation.control_period_s,
                "record_period": self.simulation.record_period_s, "sample_period": self.sensor.sample_period_s,
                "heater_latency": self.heater.latency_s, "sensor_latency": self.sensor.latency_s,
                "compute_latency": self.simulation.compute_latency_s, "scenario_start": self.scenario.start_s}
        if self.scenario.dropout_start_s >= 0:
            values["dropout_start"] = self.scenario.dropout_start_s
        values["dropout_duration"] = self.scenario.dropout_duration_s
        if self.scenario.kind == "local_triangle":
            values["aggressor_half_period"] = self.scenario.aggressor_period_s / 2
        return values

    def ticks(self, seconds, label="time"):
        value = float(seconds) / self.simulation.tick_s
        tolerance = max(1e-7, 4*abs(np.spacing(value))) if np.isfinite(value) else 0.
        if value < 0 or not np.isfinite(value) or (value > 0 and round(value) == 0) or abs(value - round(value)) > tolerance:
            raise ValueError(f"{label} must be a nonnegative integer multiple of simulation.tick_s")
        return int(round(value))

    def resolved(self):
        data = {name: asdict(getattr(self, name)) for name in (*SECTIONS, "ring", "junction", "thermal")}
        data.update(version=1, name=self.name, controller=self.controller)
        evidence = {path: {"status": "assumed", "source": "synthetic reference; requires calibration"}
                    for path, _ in _flatten(data) if path not in ("name", "version")}
        evidence.update(self.evidence)
        data["evidence"] = evidence
        return data

    def digest(self):
        return hashlib.sha256(json.dumps(self.resolved(), sort_keys=True).encode()).hexdigest()


def load(path_or_mapping=None, overrides=None):
    if isinstance(path_or_mapping, Configuration):
        raw = path_or_mapping.resolved()
    elif path_or_mapping is None:
        raw = {}
    elif isinstance(path_or_mapping, Mapping):
        raw = copy.deepcopy(dict(path_or_mapping))
    else:
        raw = yaml.safe_load(Path(path_or_mapping).read_text())
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    if isinstance(overrides, Mapping):
        entries = overrides.items()
    else:
        entries = []
        for item in overrides or []:
            path, sep, value = item.partition("=")
            if not sep:
                raise ValueError("override must be section.parameter=value")
            entries.append((path, yaml.safe_load(value)))
    for path, value in entries:
        _set(raw, path, value)
        raw.setdefault("evidence", {})[path] = {"status": "override", "source": "experiment input"}
    return Configuration(raw)
