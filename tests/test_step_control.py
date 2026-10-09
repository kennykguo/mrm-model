import json

import numpy as np
import pytest

from ringlab.config import load
from ringlab.controllers import Interface, Observation, StepLock, create, snapshot
from ringlab.experiments import rerun, save
from ringlab.plant import simulate


def observation(time, sample, applied, monitor=100, fresh=True, clipped=False):
    codes = np.atleast_1d(applied).astype(np.int64)
    measured = np.broadcast_to(np.asarray(monitor, np.int64), codes.shape).copy()
    saturated = np.broadcast_to(np.asarray(clipped, bool), codes.shape).copy()
    return Observation(time, sample, measured, saturated, codes, 6000, fresh)


def controller(rings=1, **parameters):
    settings = {"target_code": 100, "initial_code": 50, "step_codes": 4,
                "deadband_codes": 2, "warmup_s": 0}
    settings.update(parameters)
    ctrl = StepLock(**settings)
    ctrl.reset(Interface(rings, 255, 255, 1.))
    return ctrl


def test_red_side_sign_deadband_and_independent_clipping_across_rings():
    ctrl = controller(rings=6)
    ctrl.step(observation(0, -1, [50]*6, fresh=False))
    result = ctrl.step(observation(1, .5, [50]*6, [103, 97, 102, 98, 100, 200],
                                   clipped=[False, False, False, False, False, True]))
    np.testing.assert_array_equal(result, [54, 46, 50, 50, 50, 50])
    assert result.dtype == np.int64
    result[:] = 0  # Callers cannot mutate the controller's requested vector.
    np.testing.assert_array_equal(ctrl.action, [54, 46, 50, 50, 50, 50])


def test_fractional_target_is_preserved():
    ctrl = controller(rings=2, target_code=100.5, deadband_codes=0)
    ctrl.step(observation(0, -1, [50, 50], fresh=False))
    np.testing.assert_array_equal(ctrl.step(observation(1, .5, [50, 50], [100, 101])), [46, 54])


def test_explicit_warmup_uses_sample_time_after_initial_application():
    ctrl = controller(warmup_s=2.)
    # Time spent waiting for the prebias to reach the DAC is not warmup.
    assert ctrl.step(observation(10, 9, 0, monitor=200))[0] == 50
    assert ctrl.step(observation(11, 10, 50, monitor=200))[0] == 50
    # It is now past the warmup end, but this delayed sample precedes it.
    assert ctrl.step(observation(15, 12.999, 50, monitor=200))[0] == 50
    # Equality at the explicit warmup end is accepted.
    assert ctrl.step(observation(16, 13, 50, monitor=200))[0] == 54
    # Warmup is once per reset; it is not silently repeated after every step.
    assert ctrl.step(observation(17, 16, 54, monitor=200))[0] == 54
    assert ctrl.step(observation(18, 17.5, 54, monitor=200))[0] == 58


def test_stale_clipped_and_pre_action_samples_never_step():
    ctrl = controller()
    assert ctrl.step(observation(0, -1, 50, monitor=200, fresh=False))[0] == 50
    assert ctrl.step(observation(1, .5, 50, monitor=200, clipped=True))[0] == 50
    assert ctrl.step(observation(2, 1.5, 50, monitor=200, fresh=False))[0] == 50
    assert ctrl.step(observation(3, 2.5, 50, monitor=200))[0] == 54
    # Even new samples must not trigger more steps while the DAC still holds
    # the old request. Neither arrival time nor fresh alone proves application.
    for time in range(4, 10):
        assert ctrl.step(observation(time, time - .5, 50, monitor=200))[0] == 54
    assert ctrl.step(observation(10, 9.5, 54, monitor=200))[0] == 54
    assert ctrl.step(observation(11, 10, 54, monitor=200))[0] == 54
    assert ctrl.step(observation(12, 10.5, 54, monitor=200))[0] == 58


def test_all_requested_codes_must_be_applied_before_the_next_vector_update():
    ctrl = controller(rings=2)
    ctrl.step(observation(0, -1, [50, 50], fresh=False))
    np.testing.assert_array_equal(ctrl.step(observation(1, .5, [50, 50], [110, 90])), [54, 46])
    np.testing.assert_array_equal(ctrl.step(observation(2, 1.5, [54, 50], [110, 90])), [54, 46])
    np.testing.assert_array_equal(ctrl.step(observation(3, 2.5, [54, 46], [110, 90])), [54, 46])
    np.testing.assert_array_equal(ctrl.step(observation(4, 3.5, [54, 46], [110, 90])), [58, 42])


@pytest.mark.parametrize("initial,measured,expected", [(254, 200, 255), (1, 0, 0)])
def test_steps_clamp_at_both_rails(initial, measured, expected):
    ctrl = controller(initial_code=initial)
    ctrl.step(observation(0, -1, initial, fresh=False))
    assert ctrl.step(observation(1, .5, initial, measured))[0] == expected
    ctrl.step(observation(2, 1.5, expected, measured))
    assert ctrl.step(observation(3, 2.5, expected, measured))[0] == expected
    # No hidden integral winds up while an outward step is rail limited.
    reversed_measurement = 0 if expected else 200
    assert ctrl.step(observation(4, 3.5, expected, reversed_measurement))[0] == abs(expected - 4)


def test_step_wider_than_dac_range_clamps_without_integer_overflow():
    ctrl = controller(step_codes=10**100)
    ctrl.step(observation(0, -1, 50, fresh=False))
    assert ctrl.step(observation(1, .5, 50, 200))[0] == 255


def test_controller_uses_only_the_public_observation_contract():
    class PublicObservation:
        __slots__ = ("time_s", "sample_time_s", "applied_heater_code",
                     "monitor_code", "saturated", "fresh")

        def __init__(self, time, sample):
            self.time_s, self.sample_time_s = time, sample
            self.applied_heater_code = np.array([50])
            self.monitor_code = np.array([200])
            self.saturated = np.array([False])
            self.fresh = True

    ctrl = controller()
    assert ctrl.step(PublicObservation(0, -1))[0] == 50
    assert ctrl.step(PublicObservation(1, .5))[0] == 54


@pytest.mark.parametrize("parameter,value", [
    ("target_code", -1), ("target_code", float("nan")),
    ("target_code", float("inf")), ("target_code", True), ("target_code", [100]),
    ("initial_code", -1), ("initial_code", 1.5), ("initial_code", False), ("initial_code", [50]),
    ("step_codes", 0), ("step_codes", -1), ("step_codes", 1.5), ("step_codes", True),
    ("deadband_codes", -1), ("deadband_codes", .5), ("deadband_codes", True),
    ("warmup_s", -.1), ("warmup_s", float("nan")), ("warmup_s", float("inf")),
    ("warmup_s", True),
])
def test_constructor_rejects_invalid_values(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        StepLock(**{parameter: value})


@pytest.mark.parametrize("parameters", [{"target_code": 256}, {"initial_code": 256}])
def test_converter_bounds_are_checked_on_reset(parameters):
    with pytest.raises(ValueError, match="ADC/DAC range"):
        controller(**parameters)


def test_registry_snapshot_and_reset_preserve_constructor_inputs():
    ctrl = create("step")
    expected = {"name": "step", "kind": "builtin", "parameters": {
        "target_code": 260., "initial_code": 2660, "step_codes": 4,
        "deadband_codes": 2, "warmup_s": .0005}}
    assert isinstance(ctrl, StepLock)
    assert snapshot(ctrl) == expected
    ctrl.reset(Interface(2, 4095, 4095, 1e-5))
    ctrl.step(observation(0, -1, [2660, 2660], fresh=False))
    np.testing.assert_array_equal(ctrl.step(observation(.001, .0005, [2660, 2660], 300)), [2664, 2664])
    assert snapshot(ctrl) == expected
    ctrl.reset(Interface(2, 4095, 4095, 1e-5))
    np.testing.assert_array_equal(ctrl.step(observation(0, -1, [0, 0])), [2660, 2660])
    assert ctrl._first_applied_s is None and ctrl._warmup_end_s is None


def test_step_control_saved_replay_includes_explicit_parameters(tmp_path):
    cfg = load(overrides={"simulation.duration_s": 100e-6,
                          "simulation.record_period_s": 5e-6,
                          "simulation.control_period_s": 5e-6,
                          "simulation.compute_latency_s": 0.,
                          "heater.latency_s": 20e-6, "sensor.latency_s": 10e-6,
                          "data.averaging": "cw", "sensor.noise_enabled": False})
    ctrl = StepLock(target_code=0.5, initial_code=700, step_codes=3,
                    deadband_codes=0, warmup_s=15e-6)
    result = simulate(cfg, controller=ctrl)
    path = save(result, tmp_path, plot=False)
    assert json.loads((path / "controller.json").read_text()) == snapshot(ctrl)
    replay = rerun(path, tmp_path)
    with np.load(path / "trace.npz") as first, np.load(replay / "trace.npz") as second:
        assert set(first.files) == set(second.files)
        for name in first.files:
            np.testing.assert_array_equal(first[name], second[name])
    # Warmup cannot start before the first command reaches the DAC at 20 us;
    # only a measurement taken >=35 us may trigger the first increment.
    times = result.trace["time_s"]
    codes = result.trace["heater_code"][:, 0]
    assert np.all(codes[(times >= 20e-6) & (times < 65e-6)] == 700)
    assert np.max(codes) > 700
