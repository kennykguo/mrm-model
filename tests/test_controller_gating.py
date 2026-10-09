import numpy as np
import pytest

from ringlab.config import load
from ringlab.controllers import DitherLock, Hold, Interface, Observation, PID, ScanPI
from ringlab.plant import simulate


def observation(time, sample, applied, monitor=100, fresh=True):
    codes = np.atleast_1d(applied).astype(np.int64)
    measured = np.broadcast_to(np.asarray(monitor, np.int64), codes.shape).copy()
    return Observation(time, sample, measured, np.zeros(codes.shape, bool), codes, 6000, fresh)


def test_scan_counts_only_samples_after_observed_command_application():
    ctrl = ScanPI(scan_steps=8, settle_frames=2)
    ctrl.reset(Interface(1, 4095, 4095, 1.))
    ctrl.step(observation(0, -1, 0, fresh=False))
    ctrl.step(observation(1, .5, 0, monitor=10))
    desired = ctrl.step(observation(2, 1.5, 0, monitor=20))[0]
    assert ctrl.index == 1 and ctrl.readings[0] == 20
    # Latency exceeds dwell: repeated fresh samples still describe the old code.
    for time in range(3, 9):
        assert ctrl.step(observation(time, time - .5, 0))[0] == desired
        assert ctrl.frames == 0 and ctrl.index == 1
    ctrl.step(observation(9, 8.5, desired))
    ctrl.step(observation(10, 9., desired))  # equality is not strictly after application
    assert ctrl.frames == 0
    ctrl.step(observation(11, 9.5, desired, monitor=30))
    assert ctrl.frames == 1
    ctrl.step(observation(12, 9.5, desired, monitor=30, fresh=False))
    assert ctrl.frames == 1
    ctrl.step(observation(13, 12.5, desired, monitor=40))
    assert ctrl.index == 2 and ctrl.readings[1] == 40


def test_scan_acquired_pi_updates_while_next_ring_waits_for_application():
    ctrl = ScanPI(scan_steps=8, settle_frames=2, kp=1, ki_per_s=1)
    ctrl.reset(Interface(2, 4095, 4095, .01))
    ctrl.acquired[0] = True
    ctrl.target[0] = 100
    ctrl.integral[0] = 500
    ctrl.action[:] = [500, 585]
    ctrl.ring = 1
    result = ctrl.step(observation(.01, .005, [500, 0], monitor=[120, 10]))
    assert result[0] > 500
    assert result[1] == 585 and ctrl.frames == 0


def test_scan_rejects_a_clipped_peak_even_when_last_point_is_unsaturated():
    ctrl = ScanPI(scan_steps=8, settle_frames=1)
    ctrl.reset(Interface(1, 4095, 4095, 1.))
    readings = [10, 100, 300, 4095, 3000, 2000, 1000, 500]
    for index, measured in enumerate(readings):
        requested = ctrl.action.copy()
        # First observe application; only a later physical sample is eligible.
        ctrl.step(observation(2*index, 2*index - .5, requested, monitor=measured))
        sample = observation(2*index + 1, 2*index + .5, requested, monitor=measured)
        sample.saturated[0] = index == 3
        ctrl.step(sample)
    assert ctrl.ring == 1
    # A clipped peak gives only a lower bound on its true height, so its
    # fractional-height target cannot support a successful acquisition claim.
    assert ctrl.failed[0]
    assert not ctrl.acquired[0]
    assert ctrl.action[0] == 0


def test_dither_waits_for_all_applied_codes_and_a_later_sample():
    ctrl = DitherLock(initial_code=1000, amplitude_codes=10, dwell_frames=2, gain=.5)
    ctrl.reset(Interface(2, 4095, 4095, 1.))
    for time in range(6):
        np.testing.assert_array_equal(ctrl.step(observation(time, time - .5, [1010, 0])), [1010, 1010])
        assert ctrl.frames == 0
    ctrl.step(observation(6, 5.5, [1010, 1010]))
    ctrl.step(observation(7, 6, [1010, 1010]))
    assert ctrl.frames == 0
    ctrl.step(observation(8, 7.5, [1010, 1010], monitor=[120, 80]))
    low = ctrl.step(observation(9, 8.5, [1010, 1010], monitor=[120, 80]))
    np.testing.assert_array_equal(low, [990, 990])
    for time in range(10, 16):
        np.testing.assert_array_equal(ctrl.step(observation(time, time - .5, [1010, 1010])), low)
        assert ctrl.sign == -1 and ctrl.frames == 0
    ctrl.step(observation(16, 15.5, low))
    ctrl.step(observation(17, 16.5, low, monitor=[100, 100]))
    high = ctrl.step(observation(18, 17.5, low, monitor=[100, 100]))
    np.testing.assert_array_equal(high, [1020, 1000])


@pytest.mark.parametrize("controller", [ScanPI(scan_steps=8, settle_frames=1),
                                       DitherLock(initial_code=1000, dwell_frames=1)])
def test_pipeline_longer_than_dwell_does_not_run_ahead_of_hardware(controller):
    cfg = load(overrides={"simulation.duration_s": 100e-6, "simulation.control_period_s": 5e-6,
                          "simulation.record_period_s": 1e-6, "simulation.compute_latency_s": 0.,
                          "heater.latency_s": 40e-6, "sensor.latency_s": 15e-6,
                          "data.averaging": "cw", "sensor.noise_enabled": False})
    result = simulate(cfg, controller=controller)
    if isinstance(controller, ScanPI):
        # The first nonzero point cannot arrive until the first zero reading
        # plus 40 us; no further point has been sampled by the final edge.
        assert controller.index <= 2
        assert len(np.unique(result.trace["heater_code"])) <= 2
    else:
        # High first reaches the DAC at 40 us, a strictly later sample releases
        # at 60 us, so the low command first reaches it at 100 us.
        times = result.trace["time_s"]
        codes = result.trace["heater_code"][:, 0]
        assert np.all(codes[(times >= 40e-6) & (times < 99e-6)] == 1016)


@pytest.mark.parametrize("factory,kwargs", [
    (Hold, {"code": 1.5}), (Hold, {"code": True}), (Hold, {"code": [1., 2.]}),
    (PID, {"initial_code": 1.5}), (PID, {"target_code": float("nan")}),
    (PID, {"sign": 0}), (PID, {"kp": -.1}), (PID, {"ki_per_s": float("inf")}),
    (ScanPI, {"scan_steps": 8.5}), (ScanPI, {"scan_steps": True}),
    (ScanPI, {"settle_frames": 1.5}), (ScanPI, {"target_fraction": float("nan")}),
    (ScanPI, {"minimum_peak_code": -1}), (ScanPI, {"ki_per_s": -1}),
    (DitherLock, {"dwell_frames": 1.5}), (DitherLock, {"amplitude_codes": 1.5}),
    (DitherLock, {"initial_code": False}), (DitherLock, {"gain": float("nan")}),
])
def test_invalid_controller_constructor_values_are_not_truncated(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


@pytest.mark.parametrize("controller", [Hold(4096), PID(initial_code=4096), PID(target_code=4096),
                                       ScanPI(scan_steps=4097), ScanPI(minimum_peak_code=4096),
                                       DitherLock(initial_code=4096), DitherLock(amplitude_codes=2048)])
def test_controller_parameters_respect_actual_converter_ranges(controller):
    with pytest.raises(ValueError):
        controller.reset(Interface(1, 4095, 4095, 1e-6))
