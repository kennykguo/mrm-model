"""Validation: the frozen target split (``targets_io``) and the replicas of the published measurement protocols
(``replicas``), 07_design_critique.md P0-5 and section 4.

The target-file API is imported eagerly (it is light: numpy and yaml). The replica API is re-exported lazily
(PEP 562), so ``import mrm.validation.targets_io`` (for example from ``mrm/calibration``, which may read only
``calibration.yaml``) does not import the plant modules, and ``mrm.validation.replicas`` can import this package
without a cycle::

    from mrm.validation import load_all, evaluate, run_target, REPLICAS
    res = run_target(load_all()["validation"][0])        # ReplicaResult; res.value is shaped for evaluate()
"""
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

_REPLICA_EXPORTS = (
    "ReplicaResult", "ReplicaBlocked", "StaticRing", "LockPoint", "LockRunRequest", "REPLICAS",
    "LM_DLAMBDA_DT_M_PER_K", "PREDICTIONS_PATH", "default_run_fn", "resolve_plant", "load_config", "preset_for",
    "bank_gains", "bus_line_powers", "run_target", "predict_all", "write_predictions", *REPLICA_NAMES,
)

__all__ = ["REPLICA_NAMES", "Evaluation", "ExcludedItem", "Target", "TargetSchemaError", "evaluate", "load_all",
           "load_calibration_targets", "load_excluded", "load_targets", "replicas", "targets_io",
           *_REPLICA_EXPORTS]


def __getattr__(name: str) -> Any:
    """Lazy re-export of mrm.validation.replicas (and the submodules themselves)."""
    if name in ("replicas", "targets_io"):
        return import_module(f"mrm.validation.{name}")
    if name in _REPLICA_EXPORTS:
        return getattr(import_module("mrm.validation.replicas"), name)
    raise AttributeError(f"module 'mrm.validation' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
