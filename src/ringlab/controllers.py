"""Controllers receive delayed quantized measurements, never simulator state."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import importlib
import inspect
from numbers import Integral, Real
from pathlib import Path
from typing import Protocol

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(value, name, minimum=0.):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def _code(value, name):
    values = np.asarray(value)
    if values.ndim > 1 or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
        raise ValueError(f"{name} must contain nonnegative integer DAC codes")
    return int(values) if values.ndim == 0 else values.copy()


def _after_application(controller, observation, requested, applied):
    """Conservatively establish application from the public I/O observation.

    Applied codes describe now, not the delayed ADC sample. The first matching
    observation therefore establishes a lower bound for a useful sample time.
    This does not assert analog or thermal settling; dwell remains configurable.
    """
    if not np.array_equal(requested, applied):
        controller._first_applied_s = None
        controller.frames = 0
        return False
    if controller._first_applied_s is None:
        controller._first_applied_s = observation.time_s
    return observation.fresh and observation.sample_time_s > controller._first_applied_s


@dataclass(frozen=True)
class Interface:
    rings: int
    heater_max_code: int
    adc_max_code: int
    control_period_s: float

    def __post_init__(self):
        _integer(self.rings, "rings", 1)
        _integer(self.heater_max_code, "heater_max_code", 1)
        _integer(self.adc_max_code, "adc_max_code", 1)
        if _real(self.control_period_s, "control_period_s") == 0:
            raise ValueError("control_period_s must be positive")


@dataclass(frozen=True)
class Observation:
    time_s: float
    sample_time_s: float
    monitor_code: np.ndarray
    saturated: np.ndarray
    applied_heater_code: np.ndarray
    temperature_code: int
    fresh: bool


class Controller(Protocol):
    def reset(self, interface: Interface) -> None: ...
    def step(self, observation: Observation) -> np.ndarray: ...


class Hold:
    """Constant code baseline; useful for identifying the uncontrolled plant."""
    def __init__(self, code=0):
        self.code = _code(code, "code")

    def reset(self, interface):
        self.action = np.broadcast_to(np.asarray(self.code), (interface.rings,)).copy()
        if np.any(self.action > interface.heater_max_code):
            raise ValueError("code exceeds this heater DAC range")
        self.action = self.action.astype(np.int64)

    def step(self, observation):
        return self.action.copy()


class PID:
    """Explicit setpoint PI/PID on a specified fringe; integral anti-windup.

    sign=+1 assumes monitor current falls with increasing heater code at the
    chosen operating point. Acquisition is deliberately separate.
    """
    def __init__(self, target_code=1000, initial_code=1500, kp=0.3, ki_per_s=2000., kd_s=0., sign=1.):
        self.target = _real(target_code, "target_code")
        self.initial = _integer(initial_code, "initial_code")
        self.kp, self.ki, self.kd = (_real(value, name) for value, name in
                                  ((kp, "kp"), (ki_per_s, "ki_per_s"), (kd_s, "kd_s")))
        self.sign = _real(sign, "sign", -1.)
        if self.sign not in (-1., 1.):
            raise ValueError("sign must be -1 or +1")

    def reset(self, interface):
        if self.target > interface.adc_max_code or self.initial > interface.heater_max_code:
            raise ValueError("target_code or initial_code exceeds this ADC/DAC range")
        self.interface = interface
        self.integral = np.full(interface.rings, float(self.initial))
        self.previous_error = np.zeros(interface.rings)
        self.action = np.round(self.integral).astype(np.int64)

    def step(self, observation):
        if not observation.fresh:
            return self.action.copy()
        error = self.sign * (observation.monitor_code - self.target)
        dt = self.interface.control_period_s
        candidate = self.integral + self.ki * dt * error
        derivative = self.kd / dt * (error - self.previous_error)
        raw = candidate + self.kp * error + derivative
        hi = self.interface.heater_max_code
        allow = ((raw >= 0) & (raw <= hi)) | ((raw < 0) & (error > 0)) | ((raw > hi) & (error < 0))
        self.integral = np.where(allow, candidate, self.integral)
        self.action = np.rint(np.clip(self.integral + self.kp * error + derivative, 0, hi)).astype(np.int64)
        self.previous_error = error.copy()
        return self.action.copy()


class StepLock:
    """Fixed-size feedback steps on an externally chosen red-side fringe.

    Increasing the heater must decrease the measured monitor signal near this
    operating point. The controller neither identifies that fringe nor acquires
    it: initial_code and the explicit prebias warmup are experiment inputs.
    Warmup starts when the initial requested vector is first observed applied.

    Applied codes describe now, not the delayed ADC sample. After each change,
    wait for the requested vector to appear in applied_heater_code, then use
    only a strictly later sample. This prevents repeated steps on pre-action
    measurements; it does not assert analog or thermal settling after a step.
    """
    def __init__(self, target_code=260, initial_code=2660, step_codes=4,
                 deadband_codes=2, warmup_s=0.0005):
        self.target = _real(target_code, "target_code")
        self.initial = _integer(initial_code, "initial_code")
        self.step_codes = _integer(step_codes, "step_codes", 1)
        self.deadband = _integer(deadband_codes, "deadband_codes")
        self.warmup_s = _real(warmup_s, "warmup_s")

    def reset(self, interface):
        if self.target > interface.adc_max_code or self.initial > interface.heater_max_code:
            raise ValueError("target_code or initial_code exceeds this ADC/DAC range")
        self.interface = interface
        self.action = np.full(interface.rings, self.initial, dtype=np.int64)
        self._first_applied_s = None
        self._warmup_end_s = None

    def step(self, observation):
        if not np.array_equal(self.action, observation.applied_heater_code):
            self._first_applied_s = None
            return self.action.copy()
        if self._first_applied_s is None:
            self._first_applied_s = observation.time_s
        if self._warmup_end_s is None:
            self._warmup_end_s = self._first_applied_s + self.warmup_s
        if (not observation.fresh or observation.sample_time_s <= self._first_applied_s
                or observation.sample_time_s < self._warmup_end_s):
            return self.action.copy()
        error = observation.monitor_code - self.target
        direction = np.where(~observation.saturated & (np.abs(error) > self.deadband),
                             np.sign(error), 0.)
        # A step wider than the converter range simply reaches the same rail.
        increment = min(self.step_codes, self.interface.heater_max_code) * direction
        action = np.clip(self.action + increment, 0, self.interface.heater_max_code).astype(np.int64)
        if not np.array_equal(action, self.action):
            self.action = action
            self._first_applied_s = None
        return self.action.copy()


class ScanPI:
    """Sequential full-range acquisition, then a red-side drop-port PI lock.

    The measured scan identifies each peak and a fractional-height operating
    point. It sees all WDM light on the detector and cannot prove wavelength
    identity; the evaluator must separately check wrong-line capture.
    """
    def __init__(self, scan_steps=48, settle_frames=2, target_fraction=.65,
                 kp=0.3, ki_per_s=2000., minimum_peak_code=8):
        self.steps = _integer(scan_steps, "scan_steps", 8)
        self.settle = _integer(settle_frames, "settle_frames", 1)
        self.fraction = _real(target_fraction, "target_fraction")
        if not 0 < self.fraction < 1:
            raise ValueError("target_fraction must be in (0, 1)")
        self.kp, self.ki, self.minimum = (_real(value, name) for value, name in
                    ((kp, "kp"), (ki_per_s, "ki_per_s"), (minimum_peak_code, "minimum_peak_code")))

    def reset(self, interface):
        if self.steps > interface.heater_max_code + 1:
            raise ValueError("scan_steps exceeds the number of distinct DAC codes")
        if self.minimum > interface.adc_max_code:
            raise ValueError("minimum_peak_code exceeds this ADC range")
        self.interface = interface
        self.grid = np.rint(np.linspace(0, interface.heater_max_code, self.steps)).astype(np.int64)
        self.action = np.zeros(interface.rings, np.int64)
        self.target = np.zeros(interface.rings)
        self.integral = np.zeros(interface.rings)
        self.acquired = np.zeros(interface.rings, bool)
        self.failed = np.zeros(interface.rings, bool)
        self.ring, self.index, self.frames = 0, 0, 0
        self._first_applied_s = None
        self.readings = np.zeros(self.steps)
        self.scan_saturated = np.zeros(self.steps, dtype=bool)

    def step(self, observation):
        eligible = self.ring < self.interface.rings and _after_application(
            self, observation, self.action[self.ring], observation.applied_heater_code[self.ring])
        if not observation.fresh:
            return self.action.copy()
        i = self.interface
        error = observation.monitor_code - self.target
        candidate = self.integral + self.ki * i.control_period_s * error
        raw = candidate + self.kp * error
        allow = ((raw >= 0) & (raw <= i.heater_max_code)) | ((raw < 0) & (error > 0)) | ((raw > i.heater_max_code) & (error < 0))
        self.integral = np.where(self.acquired & allow, candidate, self.integral)
        self.action = np.where(self.acquired, np.rint(np.clip(self.integral + self.kp * error, 0, i.heater_max_code)), self.action).astype(np.int64)
        if self.ring >= i.rings or not eligible:
            return self.action.copy()
        n = self.ring
        self.frames += 1
        if self.frames < self.settle:
            return self.action.copy()
        self.frames = 0
        self.readings[self.index] = observation.monitor_code[n]
        self.scan_saturated[self.index] = observation.saturated[n]
        self.index += 1
        if self.index < self.steps:
            self.action[n] = self.grid[self.index]
            self._first_applied_s = None
            return self.action.copy()
        peak = int(np.argmax(self.readings))
        peak_value = self.readings[peak]
        self.target[n] = self.fraction * peak_value
        # A clipped scan does not identify the true peak height. In particular,
        # the last point being unsaturated cannot validate an earlier maximum.
        if peak_value < self.minimum or peak >= self.steps - 2 or np.any(self.scan_saturated):
            self.failed[n] = True
            self.action[n] = 0
        else:
            branch = self.readings[peak + 1:]
            pick = peak + 1 + int(np.argmin(np.abs(branch - self.target[n])))
            self.action[n] = self.grid[pick]
            self.integral[n] = self.action[n]
            self.acquired[n] = True
        self.ring += 1
        self.index, self.frames = 0, 0
        self._first_applied_s = None
        self.readings.fill(0)
        self.scan_saturated.fill(False)
        return self.action.copy()


class DitherLock:
    """Two-sided perturbation that maximizes the measured drop-port signal.

    A peak lock is a useful resonance baseline, but is not automatically the
    best data-modulation operating point. OMA is reported independently.
    """
    def __init__(self, initial_code=1500, amplitude_codes=16, gain=0.4, dwell_frames=4):
        self.initial = _integer(initial_code, "initial_code")
        self.amplitude = _integer(amplitude_codes, "amplitude_codes", 1)
        self.gain = _real(gain, "gain")
        self.dwell = _integer(dwell_frames, "dwell_frames", 1)

    def reset(self, interface):
        if self.initial > interface.heater_max_code or 2*self.amplitude > interface.heater_max_code:
            raise ValueError("initial_code or amplitude_codes exceeds this heater DAC range")
        self.interface = interface
        self.center = np.full(interface.rings, float(self.initial))
        self.sign, self.frames = 1, 0
        self._first_applied_s = None
        self.high = np.zeros(interface.rings)
        self.action = np.rint(np.clip(self.center + self.amplitude, 0, interface.heater_max_code)).astype(np.int64)

    def step(self, observation):
        if not _after_application(self, observation, self.action, observation.applied_heater_code):
            return self.action.copy()
        self.frames += 1
        if self.frames < self.dwell:
            return self.action.copy()
        self.frames = 0
        if self.sign == 1:
            self.high = observation.monitor_code.astype(float)
        else:
            self.center += self.gain * (self.high - observation.monitor_code)
            self.center = np.clip(self.center, self.amplitude, self.interface.heater_max_code - self.amplitude)
        self.sign *= -1
        self.action = np.rint(np.clip(self.center + self.sign * self.amplitude, 0, self.interface.heater_max_code)).astype(np.int64)
        self._first_applied_s = None
        return self.action.copy()


REGISTRY = {"hold": Hold, "pid": PID, "step": StepLock, "scan_pi": ScanPI, "dither": DitherLock}

# These map public constructor arguments to configuration attributes, never
# runtime state such as a PID integral or the scan's acquired target vector.
_PARAMETERS = {
    Hold: {"code": "code"},
    PID: {"target_code": "target", "initial_code": "initial", "kp": "kp", "ki_per_s": "ki", "kd_s": "kd", "sign": "sign"},
    StepLock: {"target_code": "target", "initial_code": "initial", "step_codes": "step_codes", "deadband_codes": "deadband", "warmup_s": "warmup_s"},
    ScanPI: {"scan_steps": "steps", "settle_frames": "settle", "target_fraction": "fraction", "kp": "kp", "ki_per_s": "ki", "minimum_peak_code": "minimum"},
    DitherLock: {"initial_code": "initial", "amplitude_codes": "amplitude", "gain": "gain", "dwell_frames": "dwell"},
}
_PLUGIN_OBSERVED_SOURCE = {}


def _module_source_hash(module):
    source = inspect.getsourcefile(module)
    if source is None or not Path(source).is_file():
        raise ValueError(f"controller plugin {module.__name__!r} needs a readable Python source file for saved replay")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def _factory(name):
    if name in REGISTRY:
        return REGISTRY[name]
    if isinstance(name, str) and name.count(":") == 1 and all(name.split(":")):
        module, attr = name.split(":")
        factory = getattr(importlib.import_module(module), attr)
        for module_name in {module, factory.__module__}:
            digest = _module_source_hash(importlib.import_module(module_name))
            previous = _PLUGIN_OBSERVED_SOURCE.setdefault(module_name, digest)
            if previous != digest:
                raise ValueError("controller plugin source changed after loading; start a fresh Python process before replay")
        return factory
    raise ValueError(f"unknown controller {name!r}; available: {sorted(REGISTRY)} or module:Class")


def plugin_source_hashes(name):
    """Fingerprint the entry module and factory module, not external resources.

    A plugin may load data, import additional packages or contact services; this
    is deliberately not advertised as a complete capture of those dependencies.
    """
    factory = _factory(name)
    entry_module = name.split(":", 1)[0]
    hashes = {}
    for module_name in sorted({entry_module, factory.__module__}):
        module = importlib.import_module(module_name)
        hashes[module_name] = _module_source_hash(module)
    return hashes


def snapshot(controller, configured=None):
    """Capture constructor inputs before reset mutates a controller.

    An arbitrary injected object has no general constructor/state serializer.
    Such objects remain usable in simulations, but saving needs an explicitly
    supplied importable specification, or construction through configuration.
    """
    cls = type(controller)
    if cls in _PARAMETERS:
        name = next(key for key, value in REGISTRY.items() if value is cls)
        parameters = {key: copy.deepcopy(getattr(controller, attr)) for key, attr in _PARAMETERS[cls].items()}
        return {"name": name, "parameters": parameters, "kind": "builtin"}
    if configured is None:
        return None
    name = configured.get("name")
    if not isinstance(name, str) or ":" not in name:
        raise ValueError("custom controller replay requires an explicit module:Class configuration")
    factory = _factory(name)
    if not isinstance(factory, type) or cls is not factory:
        raise ValueError("custom controller replay configuration must name the actual controller class")
    return {"name": name, "parameters": copy.deepcopy(configured.get("parameters", {})),
            "kind": "importable", "source_sha256": plugin_source_hashes(name),
            "capture_note": "Entry/factory Python source and explicit constructor parameters only; imported dependencies, external resources and pre-existing object state are not captured."}


def summary(controller):
    if type(controller) is ScanPI:
        return {"acquired": controller.acquired.copy(), "failed": controller.failed.copy(),
                "target_code": controller.target.copy(), "scan_ring": controller.ring,
                "scan_index": controller.index}
    return {}


def create(name, parameters=None):
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError("controller parameters must be a mapping")
    return _factory(name)(**copy.deepcopy(parameters or {}))
