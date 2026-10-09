from dataclasses import replace

import numpy as np
import pytest

from mrm_model.config import ExperimentConfig, default_config
from mrm_model.controllers import Observation
from mrm_model.simulation import bus_response, compare_controllers, nominal_context, run_experiment


def brief_config(rings=1):
    config = default_config(rings)
    return replace(
        config,
        run=replace(config.run, duration_s=0.001, sample_period_s=10e-6),
        disturbance=replace(config.disturbance, start_s=0.0002, ambient_amplitude_k=0.2),
    )


def test_config_roundtrip_and_unknown_fields():
    config = default_config(4)
    recovered = ExperimentConfig.from_json(config.to_json())
    assert recovered.to_dict() == config.to_dict()
    data = config.to_dict()
    data["optical"][0]["loaded_q"] = 3500
    with pytest.raises(ValueError, match="Unknown OpticalParams"):
        ExperimentConfig.from_dict(data)


def test_resource_limit_rejects_long_bit_by_bit_thermal_runs():
    config = default_config()
    with pytest.raises(ValueError, match="limits"):
        replace(config.run, duration_s=31 * 3600, sample_period_s=1 / 50e9)


def test_seed_reproducibility_and_controller_only_measured_interface():
    config = brief_config()

    class RecordingController:
        def __init__(self):
            self.samples = []

        def step(self, observation):
            assert isinstance(observation, Observation)
            assert not hasattr(observation, "temperature_k")
            assert not hasattr(observation, "detuning_hz")
            self.samples.append(observation)
            return np.asarray(config.run.initial_codes)

    controller = RecordingController()
    first = run_experiment(config, controller)
    second = run_experiment(config, RecordingController())
    for name in first.traces:
        np.testing.assert_array_equal(first.traces[name], second.traces[name])
    assert len(controller.samples) == 100
    assert first.metrics["ber"] is None


def test_same_flank_controller_comparison_improves_ramp_tracking():
    config = brief_config()
    results = compare_controllers(config, ("open_loop", "pi", "mpc"))
    baseline = results["open_loop"].metrics["tracking_rms_hz_per_lane"][0]
    for kind in ("pi", "mpc"):
        assert results[kind].metrics["tracking_rms_hz_per_lane"][0] < baseline * 0.65
        np.testing.assert_array_equal(
            results[kind].metrics["target_detuning_hz_per_lane"],
            results["open_loop"].metrics["target_detuning_hz_per_lane"],
        )
    assert config.controller.dither_objective == "flank"


def test_boundary_step_has_state_memory_and_substep_convergence():
    config = brief_config()
    config = replace(
        config,
        controller=replace(config.controller, kind="open_loop"),
        disturbance=replace(config.disturbance, kind="ambient_step", ambient_amplitude_k=1, start_s=0),
    )
    # Explicit initial state makes this an actual boundary step rather than initial equilibrium.
    from mrm_model.simulation import _steady_state

    constant = replace(config, disturbance=replace(config.disturbance, kind="constant"))
    initial, _ = _steady_state(constant)
    config = replace(
        config, run=replace(config.run, initial_temperature_k=tuple(initial.state_temperature_k))
    )
    coarse = run_experiment(config)
    fine = run_experiment(replace(config, run=replace(config.run, physics_substeps=8)))
    rise = coarse.traces["ring_temperature_k"][0, 0] - initial.ring_temperatures_k[0]
    assert 0 < rise < 1
    np.testing.assert_allclose(
        coarse.traces["ring_temperature_k"], fine.traces["ring_temperature_k"], atol=0.01
    )


def test_bus_retains_wdm_aliases_and_all_carrier_energy_budget():
    config = default_config(4)
    from mrm_model.simulation import _steady_state

    thermal, _ = _steady_state(config)
    response = bus_response(config, thermal.ring_temperatures_k)
    assert response.drop_by_ring_and_carrier_w.shape == (4, 4)
    assert np.any(
        response.drop_by_ring_and_carrier_w - np.diag(np.diag(response.drop_by_ring_and_carrier_w)) > 0
    )
    incident = config.run.input_power_w * 4
    through = (
        (1 - config.run.data_one_fraction) * response.output_low_w
        + config.run.data_one_fraction * response.output_high_w
    ).sum()
    # Assigned rail averages equal each ring's duty mean; all ports conserve power.
    np.testing.assert_allclose(
        through + response.monitor_power_w.sum() + response.absorbed_power_w.sum(), incident, rtol=1e-11
    )


def test_delayed_samples_preserve_acquisition_timestamp():
    config = brief_config()
    config = replace(
        config,
        sensor=replace(config.sensor, latency_s=25e-6),
        controller=replace(config.controller, kind="open_loop"),
    )
    result = run_experiment(config)
    trace = result.traces
    valid = trace["sample_time_s"] >= 0
    assert np.all(trace["time_s"][valid] - trace["sample_time_s"][valid] >= 25e-6 - 1e-14)
    assert np.all(trace["sample_time_s"][:2] == -1)


def test_nominal_context_can_be_held_fixed_under_device_changes():
    config = brief_config()
    nominal = nominal_context(config)
    altered = replace(config, optical=(replace(config.optical[0], df_dtemperature_hz_per_k=-12e9),))
    result = run_experiment(altered, context=nominal)
    assert result.metrics["controller_steps"] == 100
    assert np.all(np.isfinite(result.traces["tracking_error_hz"]))


@pytest.mark.parametrize("initialization", ("nodal", "state_rise", "modal_state_rise"))
def test_explicit_initial_states_keep_heater_calibration_and_pi_response(initialization):
    from pathlib import Path

    from mrm_model.simulation import _steady_state

    config = (
        ExperimentConfig.load(Path(__file__).parents[1] / "examples/fem_ring.json")
        if initialization == "modal_state_rise"
        else brief_config()
    )
    config = replace(
        config,
        run=replace(config.run, duration_s=0.0003),
        disturbance=replace(config.disturbance, kind="constant"),
        sensor=replace(config.sensor, shot_noise=False, input_noise_density_a_per_sqrt_hz=0),
    )
    network, driver = _steady_state(config)
    reference = nominal_context(config)
    states = (
        {"initial_temperature_k": tuple(network.state_temperature_k)}
        if initialization == "nodal"
        else {"initial_state_rise_k": tuple(network.state_rise_k)}
    )
    explicit = replace(config, run=replace(config.run, **states))
    _, explicit_driver = _steady_state(explicit)
    np.testing.assert_allclose(explicit_driver.power_w, driver.power_w, rtol=1e-10)
    context = nominal_context(explicit)
    assert np.linalg.norm(context.jacobian_per_code) > 0
    np.testing.assert_allclose(context.jacobian_per_code, reference.jacobian_per_code, rtol=1e-10)
    implicit_result = run_experiment(config)
    explicit_result = run_experiment(explicit)
    for name in implicit_result.traces:
        # Round-tripping absolute nodal K may change a modal rise by roundoff;
        # subtracting ~230 THz optical frequencies has 0.03125 Hz float64 ULPs.
        atol = 0.1 if name in {"detuning_hz", "tracking_error_hz"} else 1e-12
        np.testing.assert_allclose(explicit_result.traces[name], implicit_result.traces[name], atol=atol)


def test_invalid_controller_output_rejected():
    class Invalid:
        def step(self, observation):
            return [np.nan]

    with pytest.raises(ValueError, match="finite DAC"):
        run_experiment(brief_config(), Invalid())


def test_positive_local_heat_projects_to_signed_modal_coordinates():
    from pathlib import Path

    config = ExperimentConfig.load(Path(__file__).parents[1] / "examples/fem_ring.json")
    projected_heat = config.thermal.heater_map[:, 0] * 0.001
    assert np.any(projected_heat < 0)
    disturbance = replace(
        config.disturbance,
        kind="local_heat",
        start_s=0,
        period_s=0.0004,
        external_heat_peak_w=tuple(projected_heat),
    )
    altered = replace(
        config,
        disturbance=disturbance,
        run=replace(config.run, duration_s=0.0004),
        controller=replace(config.controller, kind="open_loop"),
    )
    recovered = ExperimentConfig.from_json(altered.to_json())
    result = run_experiment(recovered)
    assert np.max(result.traces["ring_temperature_k"]) > result.traces["ring_temperature_k"][0, 0]
    nodal = brief_config()
    negative_nodal_heat = replace(nodal.disturbance, external_heat_peak_w=(-0.001,) * nodal.thermal.n_states)
    with pytest.raises(ValueError, match="physical nodal external heat"):
        replace(nodal, disturbance=negative_nodal_heat)


def test_local_nominal_model_matches_actual_fractional_delays():
    config = brief_config()
    config = replace(
        config,
        actuator=replace(config.actuator, latency_s=15e-6),
        sensor=replace(config.sensor, latency_s=25e-6, shot_noise=False, input_noise_density_a_per_sqrt_hz=0),
        disturbance=replace(config.disturbance, kind="constant"),
    )
    context = nominal_context(config)

    class OneCodeStep:
        def step(self, observation):
            return np.asarray(config.run.initial_codes) + 1

    result = run_experiment(config, controller=OneCodeStep(), context=context)
    state = np.zeros(len(context.a))
    predicted = []
    for _ in result.traces["time_s"]:
        state = context.a @ state + context.b[:, 0]
        predicted.append((context.c @ state)[0] + context.target_fraction[0])
    valid = result.traces["sample_time_s"] >= 0
    observed = result.traces["adc_codes"][:, 0] / context.adc_max_code
    np.testing.assert_allclose(
        observed[valid], np.asarray(predicted)[valid], atol=3 / context.adc_max_code, rtol=0
    )


def test_trace_decimation_preserves_metrics_and_final_sample():
    config = brief_config()
    full = run_experiment(config)
    decimated = run_experiment(replace(config, run=replace(config.run, record_every=17)))
    for name in ("tracking_rms_hz_per_lane", "tracking_max_abs_hz_per_lane", "max_alias_fraction_per_lane"):
        assert full.metrics[name] == decimated.metrics[name]
    assert len(decimated.traces["time_s"]) == 7
    assert decimated.traces["time_s"][-1] == full.traces["time_s"][-1]
    with pytest.raises(ValueError, match="max_trace_bytes"):
        replace(config, run=replace(config.run, max_trace_bytes=1))


def test_transient_alias_warning_survives_trace_decimation(monkeypatch):
    import mrm_model.simulation as simulation

    config = brief_config(2)
    config = replace(
        config,
        run=replace(config.run, duration_s=3 * config.run.sample_period_s, record_every=2),
        controller=replace(config.controller, kind="open_loop"),
    )
    context = nominal_context(config)
    initial = simulation._steady_state(config)
    monkeypatch.setattr(simulation, "_steady_state", lambda _: initial)
    actual_bus = simulation.bus_response
    calls = 0

    def telemetry_fixture(*args, **kwargs):
        nonlocal calls
        calls += 1
        response = actual_bus(*args, **kwargs)
        dropped = np.diag(np.diag(response.drop_by_ring_and_carrier_w))
        # Inject an alias diagnostic only at the unrecorded middle endpoint:
        # initial call + (2*4 substep calls + endpoint) * 2 = call 19.
        if calls == 19:
            dropped[0, 1] = dropped[0, 0]
        return replace(response, drop_by_ring_and_carrier_w=dropped, monitor_power_w=dropped.sum(axis=1))

    monkeypatch.setattr(simulation, "bus_response", telemetry_fixture)
    result = run_experiment(config, context=context)
    assert calls == 28
    assert np.max(result.traces["alias_fraction"]) == 0
    assert result.metrics["max_alias_fraction_per_lane"][0] == 0.5
    assert any("Nonassigned carriers contribute" in warning for warning in result.warnings)
