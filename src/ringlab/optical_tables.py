"""Validated optional acceleration of the stated wideband NRZ compact model.

Adaptive periodic linear interpolation preserves average power balance and
passivity. Validation checks numerical interpolation only, not the physical
approximations in waveform.wdm_nrz_average_response or device calibration.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .optics import RingParameters, _finite, decay_rates
from .waveform import WDMNRZAverageResponse, wdm_nrz_average_response


_DIAGNOSTIC_COLUMNS = (
    "dynamic_weight", "absolute_detuning_over_fsr",
    "tcmt_static_absorption_discrepancy_fraction",
    "tcmt_dynamic_absorption_change_fraction",
    "tcmt_static_through_discrepancy_fraction",
)


def _pack(response: WDMNRZAverageResponse, energy_rate: float) -> NDArray:
    """Every first-nine column is power-normalized, including energy*k_loaded."""
    columns = [response.through_power_w, response.drop_power_w,
               response.intrinsic_loss_power_w, response.absorbed_power_w,
               response.stored_energy_j * energy_rate,
               response.conditional_through_power_w[..., 0], response.conditional_through_power_w[..., 1],
               response.conditional_drop_power_w[..., 0], response.conditional_drop_power_w[..., 1]]
    columns.extend(response.diagnostics[name] for name in _DIAGNOSTIC_COLUMNS)
    return np.stack(columns, axis=-1)


@dataclass(frozen=True)
class NRZAveragingTable:
    params: RingParameters
    resonance_shift_hz: tuple[float, float]
    additional_loss_per_m: tuple[float, float]
    bit_rate_hz: float
    mark_probability: float
    transition_inner_fsr: float
    transition_outer_fsr: float
    detuning_grid_hz: NDArray
    values: NDArray
    energy_rate_per_s: float
    diagnostics: dict[str, Any]

    def direct(self, detuning_hz: ArrayLike, input_power_w: ArrayLike = 1.0) -> WDMNRZAverageResponse:
        """Evaluate the reference equations, using a single periodic interval."""
        detuning = (_finite(detuning_hz, "detuning_hz") + self.params.fsr_hz / 2) % self.params.fsr_hz - self.params.fsr_hz / 2
        return wdm_nrz_average_response(
            self.params, self.params.reference_frequency_hz + detuning,
            temperature_k=self.params.reference_temperature_k,
            resonance_shift_hz=self.resonance_shift_hz,
            additional_loss_per_m=self.additional_loss_per_m,
            input_power_w=input_power_w, bit_rate_hz=self.bit_rate_hz,
            mark_probability=self.mark_probability,
            transition_inner_fsr=self.transition_inner_fsr,
            transition_outer_fsr=self.transition_outer_fsr,
        )

    def evaluate(self, detuning_hz: ArrayLike, input_power_w: ArrayLike = 1.0) -> WDMNRZAverageResponse:
        """Detuning is laser minus thermal resonance, *excluding* junction shift.

        Arrays broadcast. A table can be shared between rings differing only in
        their reference wavelength: convert each laser to that ring's thermal
        detuning first. Coupling/loss/group delay/junction/data parameters must
        otherwise be identical. Input powers may vary because this underlying
        model is linear in incident optical power.

        If adaptive construction or independent validation failed, this method
        falls back to direct equations and reports that in diagnostics.
        """
        detuning, pin = np.broadcast_arrays(_finite(detuning_hz, "detuning_hz"), _finite(input_power_w, "input_power_w"))
        if np.any(pin < 0):
            raise ValueError("input_power_w must be nonnegative")
        if not self.diagnostics["validated"]:
            response = self.direct(detuning, pin)
            response.diagnostics.update({"used_spectral_table": False, "table_fallback_reason": self.diagnostics["reason"]})
            return response
        wrapped = (detuning + self.params.fsr_hz / 2) % self.params.fsr_hz - self.params.fsr_hz / 2
        # Search once for all physical columns. Column-by-column np.interp would
        # repeatedly copy strided columns and erase much of the speed gain.
        left = np.clip(np.searchsorted(self.detuning_grid_hz, wrapped, side="right") - 1, 0, self.detuning_grid_hz.size - 2)
        weight = (wrapped - self.detuning_grid_hz[left]) / (self.detuning_grid_hz[left + 1] - self.detuning_grid_hz[left])
        interpolated = (1 - weight[..., None]) * self.values[left] + weight[..., None] * self.values[left + 1]
        powers = interpolated[..., :9] * pin[..., None]
        conditional_through = powers[..., 5:7]
        center_shift = np.mean(self.resonance_shift_hz)
        diag = {name: interpolated[..., 9 + i] for i, name in enumerate(_DIAGNOSTIC_COLUMNS)}
        diag.update({
            "nearest_resonance_order": np.rint((detuning - center_shift) / self.params.fsr_hz).astype(np.int64),
            "used_spectral_table": True,
            "interpretation": "Validated numerical interpolation of an approximate hybrid; model discrepancy diagnostics are not error bounds.",
        })
        return WDMNRZAverageResponse(
            powers[..., 0], powers[..., 1], powers[..., 2], powers[..., 3],
            powers[..., 4] / self.energy_rate_per_s, conditional_through,
            powers[..., 7:9], np.abs(conditional_through[..., 1] - conditional_through[..., 0]), diag,
        )


def build_nrz_average_table(
    params: RingParameters,
    *,
    resonance_shift_hz: tuple[float, float] = (0.0, 0.0),
    additional_loss_per_m: tuple[float, float] = (0.0, 0.0),
    bit_rate_hz: float = 50e9,
    mark_probability: float = 0.5,
    transition_inner_fsr: float = 0.08,
    transition_outer_fsr: float = 0.12,
    absolute_tolerance: float = 1e-6,
    initial_nodes: int = 129,
    max_nodes: int = 16385,
    heldout_points: int = 2048,
    seed: int = 7319,
) -> NRZAveragingTable:
    """Build a bounded-memory periodic table, with direct evaluation fallback.

    Each interval is checked at its quarter, half and three-quarter points.
    Failing intervals receive all three points. Once these checks pass, an
    independent random validation set is evaluated. If that set fails, failed
    points become refinement data and a NEW independent set is used on the
    next validation round (at most three rounds). Node-budget exhaustion or
    validation failure disables interpolation without changing model equations.

    Tolerance applies to power/input-power fractions and energy*k_loaded/input
    power. OMA, being a difference of two interpolated powers, can accumulate
    twice their pointwise errors. Sampled checks are not a rigorous uniform
    mathematical error bound; convergence should be repeated at tighter
    tolerance for conclusions sensitive to small optical changes.
    """
    if not np.isfinite(absolute_tolerance) or absolute_tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive and finite")
    if not isinstance(initial_nodes, int) or not isinstance(max_nodes, int) or not 5 <= initial_nodes <= max_nodes:
        raise ValueError("integer node counts must satisfy 5 <= initial_nodes <= max_nodes")
    if not isinstance(heldout_points, int) or heldout_points < 32:
        raise ValueError("at least 32 integer heldout_points required")
    energy_rate = float(sum(decay_rates(params, float(np.mean(additional_loss_per_m)))))
    kwargs = dict(
        temperature_k=params.reference_temperature_k, resonance_shift_hz=resonance_shift_hz,
        additional_loss_per_m=additional_loss_per_m, bit_rate_hz=bit_rate_hz,
        mark_probability=mark_probability, transition_inner_fsr=transition_inner_fsr,
        transition_outer_fsr=transition_outer_fsr,
    )
    evaluations = 0

    def reference(detuning):
        nonlocal evaluations
        evaluations += np.size(detuning)
        return _pack(wdm_nrz_average_response(params, params.reference_frequency_hz + detuning, **kwargs), energy_rate)

    grid = np.linspace(-params.fsr_hz / 2, params.fsr_hz / 2, initial_nodes)
    values = reference(grid)
    validation_round, refinements = 0, 0
    validated = False
    reason = "not evaluated"
    quarter_error, random_error = np.inf, np.inf
    while True:
        fractions = np.array([0.25, 0.5, 0.75])
        probes = grid[:-1, None] + np.diff(grid)[:, None] * fractions
        actual = reference(probes)
        estimate = values[:-1, None, :] * (1 - fractions[None, :, None]) + values[1:, None, :] * fractions[None, :, None]
        errors = np.max(np.abs(actual[..., :9] - estimate[..., :9]), axis=-1)
        quarter_error = float(np.max(errors))
        failed_intervals = np.any(errors > absolute_tolerance, axis=-1)
        if np.any(failed_intervals):
            new_points = probes[failed_intervals].ravel()
            if grid.size + new_points.size > max_nodes:
                reason = "node budget exhausted during quarter/midpoint refinement; direct evaluation enabled"
                break
            grid = np.sort(np.concatenate([grid, new_points]))
            values = reference(grid)
            refinements += 1
            continue
        random = np.random.default_rng(np.random.SeedSequence([seed, validation_round]))
        test_points = random.uniform(-params.fsr_hz / 2, params.fsr_hz / 2, heldout_points)
        truth = reference(test_points)
        predicted = np.stack([np.interp(test_points, grid, column) for column in values.T], axis=-1)
        errors = np.max(np.abs(truth[..., :9] - predicted[..., :9]), axis=-1)
        random_error = float(np.max(errors))
        if random_error <= absolute_tolerance:
            validated = True
            reason = "quarter/midpoint and independent heldout checks passed"
            break
        validation_round += 1
        additions = test_points[errors > absolute_tolerance]
        if validation_round >= 3 or grid.size + additions.size > max_nodes:
            reason = "independent validation failed within allowed budget; direct evaluation enabled"
            break
        grid = np.sort(np.concatenate([grid, additions]))
        values = reference(grid)
        refinements += 1
    # The periodic endpoint is mathematically the same point. Tiny trig roundoff
    # differences are replaced by its first representation after validation.
    endpoint_error = float(np.max(np.abs(values[0, :9] - values[-1, :9])))
    if endpoint_error > absolute_tolerance:
        validated = False
        reason = "periodic endpoint consistency failed; direct evaluation enabled"
    values[-1] = values[0]
    grid.setflags(write=False)
    values.setflags(write=False)
    report = {
        "validated": validated, "reason": reason,
        "absolute_tolerance": absolute_tolerance,
        "quarter_midpoint_max_error": quarter_error,
        "independent_heldout_max_error": random_error,
        "periodic_endpoint_error": endpoint_error,
        "heldout_points": heldout_points, "seed": seed,
        "independent_validation_round": validation_round,
        "nodes": grid.size, "max_nodes": max_nodes,
        "refinements": refinements, "reference_point_evaluations": evaluations,
        "stored_energy_normalization_rate_per_s": energy_rate,
        "memory_bytes": grid.nbytes + values.nbytes,
        "interpretation": "Sampled interpolation validation of the stated compact model, not a device-accuracy claim or rigorous uniform bound.",
    }
    return NRZAveragingTable(
        params, tuple(resonance_shift_hz), tuple(additional_loss_per_m), bit_rate_hz,
        mark_probability, transition_inner_fsr, transition_outer_fsr,
        grid, values, energy_rate, report,
    )
