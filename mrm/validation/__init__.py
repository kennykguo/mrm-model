# compare published measurements with predictions from matching model protocols.
# start with targets/README.md for target roles and fields.
# targets_io.py loads and checks them.
# replicas.py reproduces each measurement protocol on the plant.
# `evaluate` compares its prediction with the target's declared tolerance.
# calibration targets may be fitted.
# held-out validation and bounds stay separate.
# excluded values retain reasons for being neither fitted nor scored.
# oracle.py provides a true-state control benchmark.
# sensitivity.py measures uncertain-input effects and local parameter identifiability.
# report.py builds the validation report through `mrm report`.
# latest_predictions.json stores replica results.
# target APIs load immediately.
# replica exports load on first access, so loading target definitions does not also load the plant or create an import cycle.
from __future__ import annotations

from importlib import import_module
from typing import Any

from mrm.validation.targets_io import (
    REPLICA_NAMES,
    Evaluation,
    ExcludedItem,
    Target,
    TargetSchemaError,
    evaluate,
    load_all,
    load_calibration_targets,
    load_excluded,
    load_targets,
)

# names that are re-exported lazily from mrm.validation.replicas (every replica function included)
_REPLICA_EXPORTS = (
    "ReplicaResult", "ReplicaBlocked", "StaticRing", "LockPoint", "LockRunRequest", "REPLICAS",
    "LM_DLAMBDA_DT_M_PER_K", "PREDICTIONS_PATH", "default_run_fn", "resolve_plant", "load_config", "preset_for",
    "bank_gains", "bus_line_powers", "run_target", "predict_all", "write_predictions", *REPLICA_NAMES,
)

__all__ = ["REPLICA_NAMES", "Evaluation", "ExcludedItem", "Target", "TargetSchemaError", "evaluate", "load_all",
           "load_calibration_targets", "load_excluded", "load_targets", "replicas", "targets_io",
           *_REPLICA_EXPORTS]


# lazy re-export of mrm.validation.replicas (and of the submodules themselves).
def __getattr__(name: str) -> Any:
    if name in ("replicas", "targets_io"):
        return import_module(f"mrm.validation.{name}")
    if name in _REPLICA_EXPORTS:
        return getattr(import_module("mrm.validation.replicas"), name)
    raise AttributeError(f"module 'mrm.validation' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
