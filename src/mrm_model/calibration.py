"""Bounded compact-model fits with explicit held-out evaluation.

Synthetic recovery verifies algorithms; only independent measured device data
can calibrate or validate a Lightmatter model. Local least-squares covariance
does not account for model error, correlated residuals or undocumented setup
conditions. Sample holdouts from one scan/trace are not independent experiments.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logit

from .optics import OpticalParams, optical_response


def _finite_vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return array


def _covariance(result, physical_derivatives, n_residuals):
    """Local covariance in physical parameter units, with rank diagnostics."""
    J = result.jac
    _, singular, _ = np.linalg.svd(J, full_matrices=False)
    if len(singular) == 0 or singular[-1] == 0:
        condition = math.inf
    else:
        condition = float(singular[0] / singular[-1])
    variance = 2 * result.cost / max(1, n_residuals - J.shape[1])
    covariance = np.linalg.pinv(J.T @ J, rcond=1e-12) * variance
    derivative = np.asarray(physical_derivatives)
    covariance = covariance * derivative[:, None] * derivative[None, :]
    warnings = [
        "Local covariance assumes the chosen model and independent, equal-variance residuals; it is not a device-accuracy bound."
    ]
    if not math.isfinite(condition) or condition > 1e6:
        warnings.append(
            "Fit is poorly identifiable: parameter correlations or insufficient excitation make covariance unreliable."
        )
        # A pseudoinverse can report tiny errors along an unconstrained direction.
        # Decline that misleading precision instead of returning a fake bound.
        covariance = None
    if np.any(result.active_mask):
        warnings.append(
            "One or more fitted parameters reached a bound; local Gaussian uncertainty is unreliable."
        )
    return covariance, condition, warnings


@dataclass(frozen=True)
class OpticalFit:
    params: OpticalParams
    success: bool
    message: str
    fitted_fields: tuple[str, ...]
    training_rmse_w: float
    validation_rmse_w: float
    covariance: np.ndarray | None
    standard_errors: dict[str, float | None]
    jacobian_condition: float
    identifiability_warnings: tuple[str, ...]
    prediction_through_w: np.ndarray
    prediction_drop_w: np.ndarray
    training_indices: np.ndarray
    validation_indices: np.ndarray


def fit_optical_spectrum(
    params: OpticalParams,
    frequency_hz,
    through_power_w,
    *,
    drop_power_w=None,
    input_power_w=1.0,
    temperature_k=None,
    junction_voltage_v=0.0,
    fit_fields=("reference_frequency_hz", "loss_power_np_per_m", "coupling_bus_power", "coupling_drop_power"),
    holdout_fraction=0.2,
    training_mask=None,
    relative_noise_floor=1e-6,
    max_nfev=4000,
):
    """Fit periodic spectra with positive losses and bounded power couplings.

    Geometry/group index remain fixed; one resonance cannot identify their
    product independently. Absolute input powers and port reference planes are
    required. Normalizing every scan to its own minimum destroys this evidence.
    Through-only loss/coupling swaps are ambiguous even when covariance is small;
    include drop measurements or fix an independently known parameter.
    Default sample holdout spans the scan. Supply independent scans afterwards.
    """
    f = _finite_vector(frequency_hz, "frequency_hz")
    measured_through = _finite_vector(through_power_w, "through_power_w")
    measured_drop = None if drop_power_w is None else _finite_vector(drop_power_w, "drop_power_w")
    if len(f) < max(12, 3 * len(fit_fields)) or measured_through.shape != f.shape:
        raise ValueError("frequency and measured through spectrum must match and contain enough samples")
    if measured_drop is not None and measured_drop.shape != f.shape:
        raise ValueError("drop spectrum must match frequency array")
    if (
        np.any(f <= 0)
        or np.any(measured_through < 0)
        or (measured_drop is not None and np.any(measured_drop < 0))
    ):
        raise ValueError("frequency must be positive and measured powers nonnegative")
    if not math.isfinite(relative_noise_floor) or relative_noise_floor <= 0:
        raise ValueError("relative_noise_floor must be positive")
    fields = tuple(fit_fields)
    permitted = {"reference_frequency_hz", "loss_power_np_per_m", "coupling_bus_power", "coupling_drop_power"}
    if not fields or len(set(fields)) != len(fields) or not set(fields) <= permitted:
        raise ValueError(f"fit_fields must be distinct members of {sorted(permitted)}")
    if training_mask is None:
        if not 0 < holdout_fraction < 0.5:
            raise ValueError("holdout_fraction must lie in (0, .5)")
        # Evenly spaced deterministic held-out frequency samples.
        validation = np.unique(
            np.linspace(0, len(f) - 1, max(2, round(len(f) * holdout_fraction))).round().astype(int)
        )
        train = np.ones(len(f), dtype=bool)
        train[validation] = False
    else:
        train = np.asarray(training_mask, dtype=bool)
        if train.shape != f.shape:
            raise ValueError("training_mask must match frequency array")
        validation = np.flatnonzero(~train)
    if train.sum() <= len(fields) or len(validation) < 2:
        raise ValueError("need enough training points and at least two held-out points")
    power = np.broadcast_to(np.asarray(input_power_w, dtype=float), f.shape)
    if not np.all(np.isfinite(power)) or np.any(power <= 0):
        raise ValueError("input power must be finite and positive")
    scale_power = max(float(np.max(power)), float(np.max(measured_through)), relative_noise_floor)
    frequency_scale = max(params.linewidth_hz, 1e9)
    x0, lo, hi = [], [], []
    for name in fields:
        value = getattr(params, name)
        if name == "reference_frequency_hz":
            x0.append(0.0)
            lo.append(-params.fsr_hz / 2 / frequency_scale)
            hi.append(params.fsr_hz / 2 / frequency_scale)
        elif name == "loss_power_np_per_m":
            x0.append(math.log(max(value, 1e-6)))
            lo.append(math.log(1e-12))
            hi.append(math.log(1e8))
        else:
            x0.append(float(logit(np.clip(value, 1e-7, 1 - 1e-7))))
            lo.append(float(logit(1e-9)))
            hi.append(float(logit(1 - 1e-9)))

    def decode(x):
        updates = {}
        for name, value in zip(fields, x, strict=True):
            if name == "reference_frequency_hz":
                updates[name] = params.reference_frequency_hz + float(value) * frequency_scale
            elif name == "loss_power_np_per_m":
                updates[name] = float(np.exp(value))
            else:
                updates[name] = float(expit(value))
        return replace(params, **updates)

    def evaluate(p):
        return optical_response(
            p, f, temperature_k=temperature_k, junction_voltage_v=junction_voltage_v, input_power_w=power
        )

    def residual(x):
        prediction = evaluate(decode(x))
        error = [(prediction.through_power_w[train] - measured_through[train]) / scale_power]
        if measured_drop is not None:
            error.append((prediction.drop_power_w[train] - measured_drop[train]) / scale_power)
        return np.concatenate(error)

    fitted = least_squares(
        residual, x0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=max_nfev
    )
    p = decode(fitted.x)
    prediction = evaluate(p)
    derivatives = [
        frequency_scale
        if name == "reference_frequency_hz"
        else getattr(p, name)
        if name == "loss_power_np_per_m"
        else getattr(p, name) * (1 - getattr(p, name))
        for name in fields
    ]
    covariance, condition, warnings = _covariance(fitted, derivatives, len(fitted.fun))
    warnings.append(
        "Held-out points from this spectrum are not an independent temperature, bias, power or device validation."
    )
    warnings.append(
        "Intensity-only spectra can retain discrete under/overcoupling alternatives even with both ports; phase or independently measured loss/coupling resolves them. This fit is local to its initial parameter branch."
    )
    if measured_drop is None and any(
        name in fields for name in ("loss_power_np_per_m", "coupling_bus_power", "coupling_drop_power")
    ):
        warnings.append(
            "Through-only spectra have loss/coupling topology ambiguities; measured drop spectra or independent parameter constraints are needed."
        )
    if params.voltage_points_v:
        warnings.append(
            "Fixed voltage tables override some fitted linear coefficients; verify this measurement excites fitted parameters."
        )
    error_arrays = [prediction.through_power_w - measured_through]
    if measured_drop is not None:
        error_arrays.append(prediction.drop_power_w - measured_drop)
    error = np.stack(error_arrays)
    return OpticalFit(
        p,
        bool(fitted.success),
        fitted.message,
        fields,
        float(np.sqrt(np.mean(error[:, train] ** 2))),
        float(np.sqrt(np.mean(error[:, ~train] ** 2))),
        covariance,
        {
            name: None if covariance is None else float(np.sqrt(max(0, covariance[i, i])))
            for i, name in enumerate(fields)
        },
        condition,
        tuple(warnings),
        prediction.through_power_w,
        prediction.drop_power_w,
        np.flatnonzero(train),
        validation,
    )


@dataclass(frozen=True)
class ThermalStepFit:
    tau_s: np.ndarray
    amplitudes_hz: np.ndarray
    gains_hz_per_w: np.ndarray
    offset_hz: float
    success: bool
    message: str
    training_rmse_hz: float
    validation_rmse_hz: float
    covariance: np.ndarray | None
    jacobian_condition: float
    identifiability_warnings: tuple[str, ...]
    training_indices: np.ndarray
    validation_indices: np.ndarray
    step_time_s: float

    def predict(self, time_s):
        t = np.maximum(np.asarray(time_s, dtype=float) - self.step_time_s, 0.0)
        return self.offset_hz + np.sum(self.amplitudes_hz * -np.expm1(-t[..., None] / self.tau_s), axis=-1)


def fit_thermal_step(
    time_s,
    frequency_shift_hz,
    *,
    n_modes=2,
    step_power_w=1.0,
    step_time_s=None,
    holdout_fraction=0.2,
    training_mask=None,
    amplitude_sign=None,
    max_nfev=8000,
):
    """Fit same-sign exponential step modes; evaluate held-out late response.

    Assumes a known instantaneous step in ACTUAL heater power from an initial
    thermal equilibrium, with unchanged ambient, laser and closed-loop control.
    Fit the open-loop resonance frequency, not a nonlinear drop-port ADC trace.
    Same-sign modes are appropriate for a monotone reduced response; a spatial
    thermal network may require a more general, separately identified model.
    Time constants and weights are fitted, not claimed Lightmatter values.
    """
    t = _finite_vector(time_s, "time_s")
    y = _finite_vector(frequency_shift_hz, "frequency_shift_hz")
    if t.shape != y.shape or np.any(np.diff(t) <= 0):
        raise ValueError("time and frequency must match, with strictly increasing time")
    if not isinstance(n_modes, int) or not 1 <= n_modes <= 8 or len(t) < 5 * n_modes + 5:
        raise ValueError("n_modes must be 1..8 with sufficient response samples")
    if not math.isfinite(step_power_w) or step_power_w <= 0:
        raise ValueError("step_power_w must be finite and positive")
    t0 = float(t[0]) if step_time_s is None else float(step_time_s)
    elapsed = np.maximum(t - t0, 0.0)
    duration = float(elapsed[-1])
    if duration <= 0 or not math.isfinite(t0):
        raise ValueError("step must precede end of measured response")
    if training_mask is None:
        if not 0 < holdout_fraction < 0.5:
            raise ValueError("holdout_fraction must lie in (0, .5)")
        train = np.arange(len(t)) < max(1, round(len(t) * (1 - holdout_fraction)))
    else:
        train = np.asarray(training_mask, dtype=bool)
        if train.shape != t.shape:
            raise ValueError("training_mask must match time array")
    if train.sum() <= 2 * n_modes + 1 or (~train).sum() < 2:
        raise ValueError("need sufficient training points and at least two held-out samples")
    sign = (1 if y[-1] > y[0] else -1) if amplitude_sign is None else amplitude_sign
    if sign not in (-1, 1):
        raise ValueError("amplitude_sign must be +1 or -1")
    scale = max(float(np.ptp(y)), 1.0)
    dt = float(np.min(np.diff(t)))
    taus = np.geomspace(max(dt * 3, duration / 100), duration / 3, n_modes)
    x0 = np.r_[np.full(n_modes, -math.log(n_modes)), np.log(taus / duration), y[0] / scale]
    lo = np.r_[np.full(n_modes, math.log(1e-8)), np.full(n_modes, math.log(dt / duration / 100)), -np.inf]
    hi = np.r_[np.full(n_modes, math.log(100)), np.full(n_modes, math.log(100)), np.inf]

    def unpack(x):
        return sign * scale * np.exp(x[:n_modes]), duration * np.exp(x[n_modes : 2 * n_modes]), x[-1] * scale

    def predict(x):
        amplitudes, tau, offset = unpack(x)
        return offset + np.sum(amplitudes * -np.expm1(-elapsed[:, None] / tau), axis=1)

    result = least_squares(
        lambda x: (predict(x)[train] - y[train]) / scale,
        x0,
        bounds=(lo, hi),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=max_nfev,
    )
    amplitude, tau, offset = unpack(result.x)
    covariance, condition, warnings = _covariance(result, np.r_[amplitude, tau, scale], len(result.fun))
    order = np.argsort(tau)
    permutation = np.r_[order, order + n_modes, 2 * n_modes]
    if covariance is not None:
        covariance = covariance[np.ix_(permutation, permutation)]
    tau, amplitude = tau[order], amplitude[order]
    warnings.append(
        "Late samples of this same trace are a holdout, not independent validation across heater amplitudes or ambient temperatures."
    )
    if duration < 3 * np.max(tau):
        warnings.append(
            "Record is shorter than three slow time constants; steady gain and the slow mode may be weakly identifiable."
        )
    if np.any(np.diff(tau) / tau[:-1] < 0.25):
        warnings.append("Nearby time constants cannot usually be resolved separately; consider fewer modes.")
    if np.min(tau) < 2 * dt:
        warnings.append("Fastest mode is not resolved by the measurement sampling interval.")
    error = predict(result.x) - y
    return ThermalStepFit(
        tau,
        amplitude,
        amplitude / step_power_w,
        float(offset),
        bool(result.success),
        result.message,
        float(np.sqrt(np.mean(error[train] ** 2))),
        float(np.sqrt(np.mean(error[~train] ** 2))),
        covariance,
        condition,
        tuple(warnings),
        np.flatnonzero(train),
        np.flatnonzero(~train),
        t0,
    )
