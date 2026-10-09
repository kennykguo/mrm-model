"""Fit a stated thermal topology to supplied measurements and validate separately.

No bundled data are represented as measurements.  Synthetic tests verify the
estimation machinery only.  Fits preserve passivity by optimizing positive
capacity and conductance scale factors.  Missing spatial geometry cannot be
identified from one scalar step trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares

from ringlab.thermal import ThermalNetwork


@dataclass(frozen=True)
class ThermalStepTrace:
    """Measured rise after a known zero-to-constant-power step at t=0.

    Initial conditions must be thermal equilibrium at a constant, uniform
    boundary temperature. temperature_rise_k is baseline-subtracted and has
    shape (time, observer), or (time,) for one observer. power_step_w is a
    vector over the topology's heat-source ports. dataset_id identifies the
    independent acquisition, not an arbitrary train/test row split.
    """

    time_s: np.ndarray
    temperature_rise_k: np.ndarray
    power_step_w: np.ndarray
    dataset_id: str
    provenance: str

    def __post_init__(self):
        t = np.array(self.time_s, dtype=float, copy=True)
        y = np.array(self.temperature_rise_k, dtype=float, copy=True)
        p = np.atleast_1d(np.array(self.power_step_w, dtype=float, copy=True))
        if y.ndim == 1:
            y = y[:, None]
        if (t.ndim != 1 or t.size < 5 or not np.all(np.isfinite(t))
                or t[0] < 0 or np.any(np.diff(t) <= 0)):
            raise ValueError("at least five finite increasing nonnegative step times are required")
        if y.ndim != 2 or y.shape[0] != t.size or not np.all(np.isfinite(y)):
            raise ValueError("finite temperature-rise rows must match the time samples")
        if p.ndim != 1 or np.any(p < 0) or not np.all(np.isfinite(p)) or not np.any(p > 0):
            raise ValueError("power_step_w must have positive total heating and no negative entries")
        if not self.dataset_id.strip() or not self.provenance.strip():
            raise ValueError("dataset identity and measurement/synthetic provenance are required")
        for name, value in (("time_s", t), ("temperature_rise_k", y), ("power_step_w", p)):
            value.flags.writeable = False
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class CalibrationResult:
    network: ThermalNetwork
    capacity_scale: float
    conductance_scale: float
    training_rmse_k: float
    validation_rmse_k: float
    validation_max_absolute_error_k: float
    jacobian_condition_number: float
    training_dataset_id: str
    validation_dataset_id: str
    solver_success: bool
    solver_message: str

    def metrics(self):
        """Serializable evidence; small residuals do not establish external validity."""
        return {name: getattr(self, name) for name in (
            "capacity_scale", "conductance_scale", "training_rmse_k", "validation_rmse_k",
            "validation_max_absolute_error_k", "jacobian_condition_number",
            "training_dataset_id", "validation_dataset_id", "solver_success", "solver_message")}


def load_step_csv(path: str | Path, *, power_step_w, baseline_temperature_k: float,
                   dataset_id: str, provenance: str,
                   time_column: str = "time_s", temperature_column: str = "temperature_k"):
    """Read one observed temperature column and an explicit independently known power.

    CSV columns default to time_s,temperature_k. Times are measured relative to
    the heat step; the baseline is specified explicitly rather than estimated
    from the first post-step sample. Units and step amplitude are not inferred.
    """
    if not np.isfinite(baseline_temperature_k) or baseline_temperature_k <= 0:
        raise ValueError("baseline_temperature_k must be positive Kelvin")
    rows = np.genfromtxt(Path(path), delimiter=",", names=True, dtype=float, encoding="utf-8")
    if rows.dtype.names is None or time_column not in rows.dtype.names or temperature_column not in rows.dtype.names:
        raise ValueError(f"CSV must contain {time_column!r} and {temperature_column!r}")
    return ThermalStepTrace(np.atleast_1d(rows[time_column]),
                            np.atleast_1d(rows[temperature_column]) - baseline_temperature_k,
                            np.asarray(power_step_w), dataset_id, provenance)


def _modal_step_predictor(network: ThermalNetwork, trace: ThermalStepTrace):
    if not network.is_asymptotically_stable:
        raise ValueError("calibration requires an anchored thermal topology")
    if (trace.power_step_w.shape != (network.n_sources,)
            or trace.temperature_rise_k.shape[1] != network.n_observers):
        raise ValueError("trace source and observer counts must match the thermal topology")
    root_inverse_c = 1.0 / np.sqrt(network.capacity_j_per_k)
    symmetric_g = root_inverse_c[:, None] * network.conductance_w_per_k * root_inverse_c[None, :]
    poles, vectors = eigh(symmetric_g)
    if np.any(poles <= 0):
        raise ValueError("thermal topology must have strictly positive decay rates")
    transformed_source = vectors.T @ (root_inverse_c * (network.source_map @ trace.power_step_w))
    transformed_observer = (network.observer_matrix * root_inverse_c[None, :]) @ vectors
    amplitudes = transformed_observer * (transformed_source / poles)[None, :]

    def predict(capacity_scale=1.0, conductance_scale=1.0):
        # expm1 retains accuracy before the first thermal time constant.
        responses = -np.expm1(-trace.time_s[:, None] * poles[None, :]
                              * conductance_scale / capacity_scale) / conductance_scale
        return responses @ amplitudes.T

    return predict


def predict_step(network: ThermalNetwork, trace: ThermalStepTrace):
    """Evaluate a network on a trace's sampling times and known power inputs."""
    return _modal_step_predictor(network, trace)()


def fit_network_scales(template: ThermalNetwork, training: ThermalStepTrace,
                        validation: ThermalStepTrace, *, sigma_k: float = 1.0,
                        scale_bounds=(1e-4, 1e4)) -> CalibrationResult:
    """Fit two positive global C/G scales using training only; score a heldout run.

    This preserves all topology/source/observer assumptions. It cannot estimate
    all nodes from a single step. Residual structure or poor heldout results
    call for richer measurements/topology; they are not automatically corrected
    by adding poles. Dataset IDs must identify independent acquisitions.
    """
    if training.dataset_id == validation.dataset_id:
        raise ValueError("heldout validation must use a separate dataset_id/acquisition")
    if not np.isfinite(sigma_k) or sigma_k <= 0:
        raise ValueError("sigma_k must be a positive measurement-error scale")
    bounds = np.asarray(scale_bounds, dtype=float)
    if bounds.shape != (2,) or not 0 < bounds[0] < 1 < bounds[1] or not np.all(np.isfinite(bounds)):
        raise ValueError("scale_bounds must be finite positive bounds surrounding one")
    predict_train = _modal_step_predictor(template, training)
    predict_validation = _modal_step_predictor(template, validation)

    def residual(log_scales):
        c, g = np.exp(log_scales)
        return ((predict_train(c, g) - training.temperature_rise_k) / sigma_k).ravel()

    solved = least_squares(residual, np.zeros(2), bounds=(np.log(bounds[0]), np.log(bounds[1])),
                            xtol=1e-12, ftol=1e-12, gtol=1e-12)
    c, g = np.exp(solved.x)
    train_error = predict_train(c, g) - training.temperature_rise_k
    validation_error = predict_validation(c, g) - validation.temperature_rise_k
    singular_values = np.linalg.svd(solved.jac, compute_uv=False)
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float("inf")
    network = ThermalNetwork(template.capacity_j_per_k * c, template.conductance_w_per_k * g,
                              template.source_map, template.boundary_conductance_w_per_k * g,
                              template.observer_matrix, template.node_names,
                              template.source_names, template.observer_names)
    return CalibrationResult(network, float(c), float(g), float(np.sqrt(np.mean(train_error ** 2))),
                              float(np.sqrt(np.mean(validation_error ** 2))),
                              float(np.max(np.abs(validation_error))), condition,
                              training.dataset_id, validation.dataset_id,
                              bool(solved.success), str(solved.message))


def fit_single_pole(training: ThermalStepTrace, validation: ThermalStepTrace, *,
                     initial_gain_k_per_w=1000.0, initial_time_constant_s=1e-5,
                     sigma_k=1.0) -> CalibrationResult:
    """Fit R_th and tau from supplied data; return C=tau/R and G=1/R.

    Use only when heldout residuals support a single thermal pole. This model
    has one heat source and one temperature observation and is not a spatial
    reconstruction of an MRM.
    """
    if (not np.isfinite(initial_gain_k_per_w) or not np.isfinite(initial_time_constant_s)
            or initial_gain_k_per_w <= 0 or initial_time_constant_s <= 0):
        raise ValueError("initial gain and time constant must be positive")
    g = 1 / initial_gain_k_per_w
    template = ThermalNetwork([initial_time_constant_s * g], [[g]], [[1.0]], [[g]], [[1.0]])
    return fit_network_scales(template, training, validation, sigma_k=sigma_k)
