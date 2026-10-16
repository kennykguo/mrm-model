"""Calibration of the model to the calibration targets (07 P0-5, section 4.3).

Reads only ``mrm/validation/targets/calibration.yaml`` (through ``targets_io.load_calibration_targets``) and
predicts only through the replicas of :mod:`mrm.validation.replicas`. Modules: :mod:`.core` (least squares, profile
likelihoods, the half-prior rule, hypothesis likelihood ratios), :mod:`.groups` (which targets identify which
leaves), :mod:`.pipeline` (the run and the leaf decisions), :mod:`.preset` (``g26_calibrated.yaml``), :mod:`.report`
(before/after residuals), :mod:`.cli` (``mrm calibrate``). Findings: docs/handoff/calibration.md.
"""
from mrm.calibration.core import (FitResult, Param, Problem, Profile, calibration_targets, compare_hypotheses, fit,
                                  half_prior, profile, restrict, target_residuals)

__all__ = ["Param", "Problem", "FitResult", "Profile", "calibration_targets", "target_residuals", "fit", "profile",
           "half_prior", "compare_hypotheses", "restrict"]
