"""Bounded spectrum fitting with heldout evaluation and explicit ambiguity.

Power spectra do not, by themselves, uniquely identify every passive ring
parameter. This module exposes fixed assumptions and an exact alternate branch
instead of presenting the optimizer's chosen branch as a unique measurement.
"""

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .optics import C0, RingParameters, optical_response

_FIT_NAMES = ("resonance_shift_hz", "intrinsic_loss_per_m", "input_coupling_power", "drop_coupling_power")


@dataclass(frozen=True)
class SpectrumData:
    frequency_hz: ArrayLike
    provenance: str
    temperature_k: float = 300.0
    through_fraction: ArrayLike | None = None
    drop_fraction: ArrayLike | None = None
    through_sigma: ArrayLike | None = None
    drop_sigma: ArrayLike | None = None

    def __post_init__(self):
        frequency = np.array(self.frequency_hz, dtype=float, copy=True)
        if frequency.ndim != 1 or frequency.size < 8 or np.any(~np.isfinite(frequency)) or np.any(frequency <= 0):
            raise ValueError("frequency_hz must contain at least eight finite positive frequencies")
        frequency.setflags(write=False)
        object.__setattr__(self, "frequency_hz", frequency)
        if self.through_fraction is None and self.drop_fraction is None:
            raise ValueError("at least one normalized optical power port is required")
        for name in ("through_fraction", "drop_fraction", "through_sigma", "drop_sigma"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.array(np.broadcast_to(value, frequency.shape), dtype=float, copy=True)
            if np.any(~np.isfinite(array)) or (name.endswith("sigma") and np.any(array <= 0)):
                raise ValueError(f"{name} must be finite; supplied standard deviations must be positive")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if not np.isfinite(self.temperature_k) or self.temperature_k <= 0:
            raise ValueError("temperature_k must be positive and finite")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("measurement provenance and normalization description are required")


@dataclass(frozen=True)
class SpectrumFit:
    parameters: RingParameters
    fitted_values: dict[str, float]
    fixed_parameter_names: tuple[str, ...]
    training_rmse_by_port: dict[str, float]
    heldout_rmse_by_port: dict[str, float]
    weighted_training_cost: float
    jacobian_rank: int
    jacobian_singular_values: NDArray
    optimizer_success: bool
    optimizer_message: str
    alternate_power_equivalent_parameters: RingParameters | None
    alternate_satisfies_fit_constraints: bool
    numerically_equivalent_solutions: tuple[dict[str, float], ...]
    identifiability_notes: tuple[str, ...]
    provenance: str


def load_spectrum_csv(path: str | Path, *, provenance: str, temperature_k: float = 300.0) -> SpectrumData:
    """Read explicit SI/normalized columns; no inferred gain, units or baseline.

    Required header: frequency_hz. Supply through_fraction and/or drop_fraction.
    Optional through_sigma/drop_sigma are standard deviations of those same
    normalized fractions. Dark offsets and path insertion losses must already
    be characterized; an arbitrary normalization can bias extracted losses.
    No data file is written or changed.
    """
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        if "frequency_hz" not in names:
            raise ValueError("CSV requires an explicit frequency_hz column")
        selected = [name for name in ("frequency_hz", "through_fraction", "drop_fraction", "through_sigma", "drop_sigma") if name in names]
        columns = {name: [] for name in selected}
        for row in reader:
            for name in selected:
                columns[name].append(float(row[name]))
    return SpectrumData(**columns, provenance=provenance, temperature_k=temperature_k)


def equivalent_power_parameters(params: RingParameters) -> RingParameters | None:
    """Return the exact swapped branch of through AND drop *power* spectra.

    Put r=sqrt(1-K1), b=a*sqrt(1-K2), and d=K1*K2*a. Through power depends on
    r,b symmetrically, and drop power on d. Swapping r,b and solving
    a' - r^2/a' = d/(1-b^2) preserves both spectra. The optical complex phases
    and distributed stored energy need not agree. External component knowledge
    or complex-field measurements can distinguish these branches.
    """
    attenuation = np.exp(-params.intrinsic_loss_per_m * params.roundtrip_length_m / 2)
    r = np.sqrt(1 - params.input_coupling_power)
    b = attenuation * np.sqrt(1 - params.drop_coupling_power)
    new_input = 1 - b**2
    if not 0 < new_input < 1:
        return None
    d = params.input_coupling_power * params.drop_coupling_power * attenuation
    u = d / new_input
    new_attenuation = (u + np.sqrt(u**2 + 4 * r**2)) / 2
    new_drop = 1 - (r / new_attenuation)**2
    if new_attenuation > 1 + 1e-12 or new_attenuation <= 0 or new_drop < -1e-12 or new_drop >= 1:
        return None
    alternate = replace(
        params, intrinsic_loss_per_m=max(0.0, -2 * np.log(min(new_attenuation, 1.0)) / params.roundtrip_length_m),
        input_coupling_power=float(new_input), drop_coupling_power=float(max(new_drop, 0.0)),
    )
    if np.allclose([alternate.intrinsic_loss_per_m, alternate.input_coupling_power, alternate.drop_coupling_power],
                   [params.intrinsic_loss_per_m, params.input_coupling_power, params.drop_coupling_power], rtol=1e-9, atol=1e-12):
        return None
    return alternate


def fit_spectrum(
    data: SpectrumData,
    initial: RingParameters,
    *,
    vary: tuple[str, ...],
    bounds: Mapping[str, tuple[float, float]],
    training_mask: ArrayLike,
    starts: int = 8,
    seed: int = 983,
) -> SpectrumFit:
    """Fit explicitly selected parameters and score data excluded from fitting.

    ``resonance_shift_hz`` is relative to initial.reference_frequency_hz.
    Unlisted RingParameters stay fixed and are named in the result. Every free
    parameter requires an explicit bounded range; there are no hidden priors or
    fitted scale/offset terms. All starts use only training points. The heldout
    split is supplied by the caller; withholding points in one scan checks
    interpolation, not extrapolation to new temperature/power/device conditions.

    A full-rank local Jacobian does NOT prove global identifiability. Exact
    discrete power-spectrum ambiguity is reported separately. Unspecified noise
    scales imply ordinary least squares and do not justify confidence intervals.
    """
    if not vary or len(set(vary)) != len(vary) or any(name not in _FIT_NAMES for name in vary):
        raise ValueError(f"vary must select unique names from {_FIT_NAMES}")
    if set(bounds) != set(vary):
        raise ValueError("explicit bounds must be provided for exactly the varied parameters")
    if not isinstance(starts, int) or starts < 1:
        raise ValueError("starts must be a positive integer")
    train = np.asarray(training_mask)
    if train.dtype != bool or train.shape != data.frequency_hz.shape:
        raise ValueError("training_mask must be a boolean vector matching the measurements")
    if np.count_nonzero(train) < max(5, 2 * len(vary)) or np.count_nonzero(~train) < 3:
        raise ValueError("need enough training points and at least three explicitly heldout points")
    intervals = np.asarray([bounds[name] for name in vary], dtype=float)
    if intervals.shape != (len(vary), 2) or np.any(~np.isfinite(intervals)) or np.any(intervals[:, 1] <= intervals[:, 0]):
        raise ValueError("all bounds must be finite increasing pairs")
    for name, (lower, upper) in zip(vary, intervals):
        if name == "intrinsic_loss_per_m" and lower < 0:
            raise ValueError("loss bounds must be nonnegative")
        if name == "input_coupling_power" and not 0 < lower < upper < 1:
            raise ValueError("input-coupling bounds must lie strictly inside (0,1)")
        if name == "drop_coupling_power" and not 0 <= lower < upper < 1:
            raise ValueError("drop-coupling bounds must lie in [0,1)")
        if name == "resonance_shift_hz" and initial.reference_frequency_hz + lower <= 0:
            raise ValueError("frequency-shift lower bound predicts nonpositive resonance frequency")
    lower, span = intervals[:, 0], np.diff(intervals, axis=1)[:, 0]

    def values_and_parameters(normalized):
        fitted = dict(zip(vary, lower + normalized * span))
        physical = {name: float(value) for name, value in fitted.items() if name != "resonance_shift_hz"}
        frequency = initial.reference_frequency_hz + fitted.get("resonance_shift_hz", 0.0)
        return fitted, replace(initial, reference_wavelength_m=C0 / frequency, **physical)

    ports = [name for name in ("through", "drop") if getattr(data, name + "_fraction") is not None]

    def residual(normalized):
        _, parameters = values_and_parameters(normalized)
        response = optical_response(parameters, data.frequency_hz[train], temperature_k=data.temperature_k)
        pieces = []
        for port in ports:
            observed = getattr(data, port + "_fraction")[train]
            sigma = getattr(data, port + "_sigma")
            pieces.append((getattr(response, port + "_fraction") - observed) / (1.0 if sigma is None else sigma[train]))
        return np.concatenate(pieces)

    starting_values = np.asarray([0.0 if name == "resonance_shift_hz" else getattr(initial, name) for name in vary])
    normalized_start = np.clip((starting_values - lower) / span, 1e-8, 1 - 1e-8)
    random = np.random.default_rng(seed)
    seeds = [normalized_start] + [random.uniform(0.05, 0.95, len(vary)) for _ in range(starts - 1)]
    solutions = [least_squares(residual, point, bounds=(np.zeros(len(vary)), np.ones(len(vary))),
                               xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=800) for point in seeds]
    best = min(solutions, key=lambda solution: solution.cost)
    fitted, parameters = values_and_parameters(best.x)
    response = optical_response(parameters, data.frequency_hz, temperature_k=data.temperature_k)
    training_rmse, heldout_rmse = {}, {}
    for port in ports:
        error = getattr(response, port + "_fraction") - getattr(data, port + "_fraction")
        training_rmse[port] = float(np.sqrt(np.mean(error[train]**2)))
        heldout_rmse[port] = float(np.sqrt(np.mean(error[~train]**2)))
    singular = np.linalg.svd(best.jac, compute_uv=False)
    rank = int(np.count_nonzero(singular > (singular[0] * 1e-7 if singular.size else 0)))
    alternate = equivalent_power_parameters(parameters)
    alternate_valid = alternate is not None
    if alternate is not None:
        for name in ("intrinsic_loss_per_m", "input_coupling_power", "drop_coupling_power"):
            if name not in vary:
                alternate_valid &= bool(np.isclose(getattr(alternate, name), getattr(initial, name), rtol=1e-8, atol=1e-12))
            else:
                alternate_valid &= bounds[name][0] <= getattr(alternate, name) <= bounds[name][1]
    equivalent = []
    for solution in solutions:
        if solution.cost <= best.cost + max(1e-10, best.cost * 1e-5):
            values, _ = values_and_parameters(solution.x)
            if not any(np.linalg.norm((np.asarray([values[n] for n in vary]) - np.asarray([old[n] for n in vary])) / span) < 1e-4 for old in equivalent):
                equivalent.append({name: float(value) for name, value in values.items()})
    notes = [
        "All non-varied ring parameters are fixed assumptions; fitted gain/offset/background terms are absent.",
        "Heldout points in one scan do not validate unseen temperatures, optical powers, voltages or fabricated devices.",
        "Local Jacobian rank does not remove discrete branch ambiguity or uncertainty in fixed parameters.",
    ]
    if ports == ["through"]:
        notes.append("Through-only power cannot label input versus internal-return attenuation; unknown intrinsic loss and drop coupling are additionally confounded.")
    if alternate is not None:
        notes.append("An exact alternate passive branch reproduces both through/drop powers. Complex phase or independent coupling/loss knowledge is needed to label it.")
    if any(getattr(data, port + "_sigma") is None for port in ports):
        notes.append("One or more noise standard deviations were unspecified; these residuals do not establish confidence intervals.")
    if any(name == "resonance_shift_hz" and bounds[name][1] - bounds[name][0] >= initial.fsr_hz for name in vary):
        notes.append("Shift bounds span an FSR; resonance-order aliases may also be present.")
    fixed = tuple(name for name in vars(initial) if name not in vary and not (name == "reference_wavelength_m" and "resonance_shift_hz" in vary))
    return SpectrumFit(
        parameters, {name: float(value) for name, value in fitted.items()}, fixed,
        training_rmse, heldout_rmse, float(best.cost), rank, singular,
        bool(best.success), best.message, alternate, bool(alternate_valid),
        tuple(equivalent), tuple(notes), data.provenance,
    )
