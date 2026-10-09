"""Explicit parameter corners, fixed nominal controller knowledge, bounded memory.

No independent uniform ranges or invented manufacturing distributions are sampled.
A corner is an entire jointly chosen parameter perturbation. Correlations belong
in the measured/solver-derived corner specification supplied by the researcher.
"""

from dataclasses import replace

import numpy as np

from .config import ExperimentConfig
from .optics import resonance_frequency
from .simulation import _steady_state, compare_controllers, nominal_context


def apply_overrides(config, overrides):
    """Dot paths use JSON field names and integer array indices; all changes validated."""
    data = config.to_dict()
    for path, value in overrides.items():
        pieces = path.split(".")
        target = data
        try:
            for part in pieces[:-1]:
                target = target[int(part)] if isinstance(target, list) else target[part]
            final = int(pieces[-1]) if isinstance(target, list) else pieces[-1]
            # New fields are not allowed; catches typographical errors before replacement.
            if isinstance(target, dict) and final not in target:
                raise KeyError(final)
            target[final] = value
        except (KeyError, IndexError, ValueError, TypeError) as error:
            raise ValueError(f"Invalid parameter path {path}") from error
    data["provenance"] = dict(
        data["provenance"],
        calibrated=False,
        corner_note="Perturbed nominal device; original calibration does not prove this corner",
    )
    return ExperimentConfig.from_dict(data)


def corner_study(config, corners, kinds=("open_loop", "pi", "dither", "mpc")):
    """Yield compact summaries, discarding each corner's traces before the next.

    Controller calibration and assigned target remain fixed at the nominal values.
    This prevents each changed device from receiving knowledge of its own unknown
    parameters or having its initial error silently defined to zero.
    """
    context = nominal_context(config)
    nominal, _ = _steady_state(config)
    target = np.asarray(config.run.laser_frequency_hz) - np.array(
        [
            resonance_frequency(
                p,
                nominal.ring_temperatures_k[i],
                (config.run.junction_low_v + config.run.junction_high_v) / 2,
            )
            for i, p in enumerate(config.optical)
        ]
    )
    if config.run.target_detuning_hz is not None:
        target = np.asarray(config.run.target_detuning_hz)
    for index, corner in enumerate(corners):
        if not isinstance(corner, dict) or set(corner) - {"name", "overrides"}:
            raise ValueError("corners need name and overrides objects")
        plant = apply_overrides(config, corner.get("overrides", {}))
        plant = replace(plant, run=replace(plant.run, target_detuning_hz=tuple(target)))
        results = compare_controllers(plant, kinds, context=context)
        yield {
            "name": corner.get("name", f"corner-{index}"),
            "overrides": corner.get("overrides", {}),
            "metrics": {kind: result.metrics for kind, result in results.items()},
            "warnings": {kind: result.warnings for kind, result in results.items()},
            "calibration_policy": "fixed nominal context and target; no per-corner recalibration",
        }
