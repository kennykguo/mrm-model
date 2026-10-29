# fit declared calibration targets and write presets with evidence for each update.
# read pipeline.py for group order and decision rules, groups.py for the physical quantities being fitted, and core.py for least squares and profile likelihoods.
# preset.py writes bank and bench children.
# report.py compares their predictions with the original presets.
# latest_fit.json stores the last fit record.
# only targets/calibration.yaml may be fitted.
# predictions use validation replicas.
# a calibrated label requires a narrow, closed, connected profile interval from a declared prior and a fit that meets target tolerances.
# other fitted values retain their weaker provenance.
# run with `mrm calibrate`.
# see docs/handoff/calibration.md for methods and findings.
from mrm.calibration.core import (FitResult, Param, Problem, Profile, calibration_targets, compare_hypotheses, fit,
                                  half_prior, profile, restrict, target_residuals)

__all__ = ["Param", "Problem", "FitResult", "Profile", "calibration_targets", "target_residuals", "fit", "profile",
           "half_prior", "compare_hypotheses", "restrict"]
