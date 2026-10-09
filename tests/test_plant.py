import numpy as np

from ringlab.config import load
from ringlab.plant import Plant, simulate


def short(**overrides):
    base = {"simulation.duration_s": 40e-6, "simulation.record_period_s": 1e-6,
            "controller.name": "hold", "controller.parameters": {"code": 500},
            "sensor.noise_enabled": False, "data.averaging": "cw"}
    base.update(overrides)
    return load(overrides=base)


def test_action_pipeline_has_one_owner_for_each_delay():
    cfg = short()
    result = simulate(cfg)
    ticks = np.rint(result.trace["time_s"]/cfg.simulation.tick_s).astype(int)
    codes = result.trace["heater_code"][:, 0]
    assert np.all(codes[ticks < 3] == 0)  # 1us compute + 2us DAC delay
    assert np.all(codes[ticks >= 3] == 500)
    # First physical sample at 5us is released at 7us, never at t=0.
    sample_times = result.trace["sample_time_s"]
    assert np.all(sample_times[ticks < 7] < 0)
    assert round(sample_times[ticks == 7][0]/cfg.simulation.tick_s) == 5


def test_seed_replay_and_record_decimation_do_not_change_plant():
    a = simulate(short(**{"sensor.noise_enabled": True}))
    b = simulate(short(**{"sensor.noise_enabled": True}))
    for key in a.trace:
        np.testing.assert_array_equal(a.trace[key], b.trace[key])
    # Records are additional integration boundaries. Tolerance convergence,
    # rather than bit equality, is the scientifically relevant comparison.
    c = simulate(short(**{"simulation.record_period_s": 10e-6}))
    d = simulate(short())
    np.testing.assert_allclose(c.final_state, d.final_state, rtol=1e-5, atol=1e-8)


def test_self_heating_is_an_actual_feedback_path():
    overrides = {"controller.parameters": {"code": 0}, "array.cold_offset_hz": 10e9,
                 "laser.power_per_line_w": 1e-3}
    hot = simulate(short(**overrides))
    dark = simulate(short(**dict(overrides, **{"ring.absorption_fraction": 0.0})))
    assert hot.trace["ring_temperature_k"][-1, 0] > dark.trace["ring_temperature_k"][-1, 0]
    assert hot.trace["absorbed_power_w"][-1, 0] > 0
    assert dark.trace["absorbed_power_w"][-1, 0] == 0


def test_tighter_tolerance_predicts_same_nonlinear_trajectory():
    base = short(**{"data.averaging": "stationary_nrz", "array.cold_offset_hz": 10e9})
    a = simulate(base)
    b = simulate(load(base, {"simulation.rtol": 1e-8, "simulation.max_step_s": .25e-6,
                             "simulation.temperature_atol_k": 1e-9}))
    np.testing.assert_allclose(a.trace["ring_temperature_k"], b.trace["ring_temperature_k"], atol=1e-4, rtol=0)
    np.testing.assert_allclose(a.trace["detuning_hz"], b.trace["detuning_hz"], atol=2e6, rtol=0)


def test_spectral_acceleration_preserves_coupled_self_heating_trajectory():
    # Includes a transient from optical/data heating near the resonance, not
    # merely interpolation of a static curve.
    cfg = short(**{"data.averaging": "stationary_nrz", "array.cold_offset_hz": 10e9})
    fast = simulate(cfg)
    direct = simulate(load(cfg, {"simulation.optical_evaluation": "direct"}))
    assert fast.diagnostics["spectral_table"]["validated"]
    np.testing.assert_allclose(fast.trace["ring_temperature_k"], direct.trace["ring_temperature_k"], atol=2e-5, rtol=0)
    np.testing.assert_allclose(fast.trace["oma_w"], direct.trace["oma_w"], atol=2e-9, rtol=2e-4)
