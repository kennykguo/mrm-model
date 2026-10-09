"""Disturbance scenarios: parametric generators evaluated inside numba, the named library, the registry and the
aggressor-power calibration (07 P0-9, P0-10, P0-11, P1-4, P1-12, D6).

Quick use::

    from mrm.scenarios import create, calibrate_scenario, foster_from_system
    sc = create("a1")                                   # slide-10 aggressor, resonance referencing (pending power)
    resp = foster_from_system(thermal_system, victim_ring=0, site="aggressor")
    sc, cal = calibrate_scenario(sc, {"aggressor": resp}, model={"s0_m_per_k": 70e-12, "beta_per_k": 1.4e-3,
                                                                  "t_ref_k": 298.15})
    arrays = sc.compile(dt_s=1e-6, n_rings=8, n_lines=8, batch=B, seed=0, aggressor_sites=["aggressor"])
    # kernel: scen_eval_b(tick, *arrays.kernel_args()...) per batch row (docs/handoff/scenarios.md)

The engine contract (signal table, array layout, time semantics) is in docs/handoff/scenarios.md.
"""
from mrm.scenarios.generators import (
    SIGNALS, SIGNAL_BY_NAME, AggressorSpec, CompressionError, Component, ConvergenceCertificate, ConvergenceReport,
    DisturbanceFrame, ResetSpec, Scenario, ScenarioArrays, ScenarioError, SignalSpec, assess_convergence,
    check_time_compression, convergence_plan, require_compression_validated, scen_eval, scen_eval_b, scen_init_b,
    scen_eval_signal_b, scen_next_break_b, scen_render, signal_id, stack,
)
from mrm.scenarios.calibrate import (
    AggressorTarget, CalibratedWaveform, FosterResponse, calibrate_aggressor_power, calibrate_scenario,
    foster_from_system, resonance_to_rise_k, rise_to_resonance_m,
)
from mrm.scenarios.registry import available, create, describe, from_config, register, resolve
from mrm.scenarios import library

__all__ = [
    "SIGNALS", "SIGNAL_BY_NAME", "SignalSpec", "signal_id", "Component", "AggressorSpec", "ResetSpec", "Scenario",
    "ScenarioArrays", "DisturbanceFrame", "ScenarioError", "CompressionError", "stack",
    "scen_eval", "scen_init_b", "scen_eval_b", "scen_eval_signal_b", "scen_next_break_b", "scen_render",
    "check_time_compression", "convergence_plan", "assess_convergence", "require_compression_validated",
    "ConvergenceCertificate", "ConvergenceReport",
    "AggressorTarget", "CalibratedWaveform", "FosterResponse", "calibrate_aggressor_power", "calibrate_scenario",
    "foster_from_system", "resonance_to_rise_k", "rise_to_resonance_m",
    "available", "create", "describe", "from_config", "register", "resolve", "library",
]
