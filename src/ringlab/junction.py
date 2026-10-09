"""Replaceable reverse-biased junction models; no private geometry is inferred."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class JunctionResponse:
    resonance_shift_hz: NDArray
    additional_loss_per_m: NDArray
    capacitance_f: NDArray
    resistance_ohm: NDArray
    leakage_current_a: NDArray
    electrical_heat_w: NDArray


class JunctionModel(Protocol):
    def evaluate(self, reverse_voltage_v: ArrayLike) -> JunctionResponse: ...


@dataclass(frozen=True)
class DepletionJunction:
    """Illustrative abrupt-junction square-root baseline, not semiconductor TCAD.

    Voltage means positive reverse-bias magnitude. Cj=Cj0/sqrt(1+V/Vbi).
    Frequency and loss follow an independently fitted depletion-width shape.
    Their coefficients contain unknown geometry, modal overlap and carrier
    response, and must be calibrated. Default negative frequency and loss
    slopes represent depletion increasing n_eff and reducing free-carrier loss.
    """

    built_in_voltage_v: float = 0.8
    zero_bias_capacitance_f: float = 30e-15
    parasitic_capacitance_f: float = 5e-15
    series_resistance_ohm: float = 70.0
    frequency_shift_scale_hz: float = -12e9
    loss_change_scale_per_m: float = -8.0
    leakage_current_a: float = 1e-9
    minimum_voltage_v: float = 0.0
    maximum_voltage_v: float = 5.0
    reference_voltage_v: float = 0.0

    def __post_init__(self) -> None:
        if not all(np.isfinite(v) for v in vars(self).values()):
            raise ValueError("junction parameters must be finite")
        if self.built_in_voltage_v <= 0 or self.zero_bias_capacitance_f <= 0 or self.series_resistance_ohm <= 0:
            raise ValueError("built-in potential, junction capacitance and resistance must be positive")
        if self.parasitic_capacitance_f < 0 or self.leakage_current_a < 0:
            raise ValueError("parasitic capacitance and leakage must be nonnegative")
        if not 0 <= self.minimum_voltage_v < self.maximum_voltage_v:
            raise ValueError("reverse-bias bounds must satisfy 0 <= minimum < maximum")
        if not self.minimum_voltage_v <= self.reference_voltage_v <= self.maximum_voltage_v:
            raise ValueError("reference_voltage_v must lie inside the calibrated voltage domain")

    def evaluate(self, reverse_voltage_v: ArrayLike) -> JunctionResponse:
        voltage = np.asarray(reverse_voltage_v, dtype=float)
        if np.any(~np.isfinite(voltage)) or np.any(voltage < self.minimum_voltage_v) or np.any(voltage > self.maximum_voltage_v):
            raise ValueError("reverse voltage outside junction model domain")
        depletion = np.sqrt(1 + voltage / self.built_in_voltage_v)
        change = depletion - np.sqrt(1 + self.reference_voltage_v / self.built_in_voltage_v)
        capacitance = self.zero_bias_capacitance_f / depletion + self.parasitic_capacitance_f
        resistance = np.full(voltage.shape, self.series_resistance_ohm)
        leakage = np.full(voltage.shape, self.leakage_current_a)
        return JunctionResponse(
            np.asarray(self.frequency_shift_scale_hz * change),
            np.asarray(self.loss_change_scale_per_m * change),
            np.asarray(capacitance), resistance, leakage, voltage * leakage,
        )


@dataclass(frozen=True)
class MeasuredJunctionTable:
    """Piecewise-linear measured/extracted data with a hard no-extrapolation rule.

    ``additional_loss_per_m`` is relative to the ring's intrinsic baseline and
    can be negative. The optical evaluator checks that their sum is passive.
    ``provenance`` should identify a measurement or solver, its conditions,
    wavelength, temperature, reference voltage and uncertainty. A 1-D table
    cannot establish temperature dependence or high-frequency parasitics.
    """

    voltage_v: ArrayLike
    resonance_shift_hz: ArrayLike
    additional_loss_per_m: ArrayLike
    capacitance_f: ArrayLike
    resistance_ohm: ArrayLike
    leakage_current_a: ArrayLike
    provenance: str

    def __post_init__(self) -> None:
        names = ("voltage_v", "resonance_shift_hz", "additional_loss_per_m", "capacitance_f", "resistance_ohm", "leakage_current_a")
        for name in names:
            array = np.array(getattr(self, name), dtype=float, copy=True)
            if array.ndim != 1 or array.size < 2 or np.any(~np.isfinite(array)):
                raise ValueError(f"{name} must be a finite 1D array with at least two entries")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if any(getattr(self, name).shape != self.voltage_v.shape for name in names):
            raise ValueError("all junction columns must have the same length")
        if self.voltage_v[0] < 0 or np.any(np.diff(self.voltage_v) <= 0):
            raise ValueError("voltage_v must be nonnegative and strictly increasing")
        if np.any(self.capacitance_f <= 0) or np.any(self.resistance_ohm <= 0) or np.any(self.leakage_current_a < 0):
            raise ValueError("capacitance/resistance must be positive and leakage nonnegative")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("measurement/extraction provenance is required")

    def evaluate(self, reverse_voltage_v: ArrayLike) -> JunctionResponse:
        voltage = np.asarray(reverse_voltage_v, dtype=float)
        if np.any(~np.isfinite(voltage)) or np.any(voltage < self.voltage_v[0]) or np.any(voltage > self.voltage_v[-1]):
            raise ValueError("reverse voltage outside measured table; extrapolation prohibited")
        columns = [np.asarray(np.interp(voltage, self.voltage_v, getattr(self, name))) for name in (
            "resonance_shift_hz", "additional_loss_per_m", "capacitance_f", "resistance_ohm", "leakage_current_a"
        )]
        return JunctionResponse(*columns, voltage * columns[-1])
