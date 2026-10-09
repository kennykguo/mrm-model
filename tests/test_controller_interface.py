import numpy as np
import pytest

from ringlab.controllers import Interface, Observation, PID, DitherLock, create


def obs(code, fresh=True):
    return Observation(.001, .00099, np.array([code], dtype=np.int64), np.array([False]), np.array([0]), 6000, fresh)


def test_integrator_does_not_wind_up_at_rail():
    ctrl = PID(target_code=1000, initial_code=2048, kp=2, ki_per_s=1e5)
    ctrl.reset(Interface(1, 4095, 4095, 20e-6))
    for _ in range(100):
        assert ctrl.step(obs(4095))[0] == 4095
    assert ctrl.integral[0] == 2048
    assert ctrl.step(obs(900))[0] < 2048


def test_stale_measurement_does_not_reintegrate():
    ctrl = PID()
    ctrl.reset(Interface(1, 4095, 4095, 20e-6))
    action = ctrl.step(obs(1400))
    for _ in range(100):
        np.testing.assert_array_equal(ctrl.step(obs(0, fresh=False)), action)


def test_custom_controller_and_unknown_name():
    assert type(create("ringlab.controllers:Hold", {"code": 12})).__name__ == "Hold"
    with pytest.raises(ValueError, match="unknown controller"):
        create("missing")


def test_observation_has_no_physical_truth():
    observation = obs(12)
    assert not hasattr(observation, "temperature_k")
    assert not hasattr(observation, "detuning_hz")
    assert not hasattr(observation, "model")
