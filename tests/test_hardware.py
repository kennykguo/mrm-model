import numpy as np
import pytest
from scipy.integrate import quad

from mrm_model.hardware import ActuatorParams, HeaterDriver, MonitorADC, SensorParams


def test_heater_latency_crossing_variable_steps_and_exact_energy():
    params = ActuatorParams(latency_s=3e-6, settling_time_s=2e-6, max_power_w=0.1)
    driver = HeaterDriver(params)
    driver.step(2e-6, [4095], [300])
    assert driver.voltage_v[0] == 0
    driver.step(4e-6, [4095], [300])
    expected_v = 3.3 * (1 - np.exp(-3e-6 / 2e-6))
    np.testing.assert_allclose(driver.voltage_v, [expected_v])
    energy = quad(lambda t: (3.3 * (1 - np.exp(-t / 2e-6))) ** 2 / 500, 0, 3e-6)[0]
    np.testing.assert_allclose(driver.interval_average_power_w, [energy / 4e-6], rtol=1e-10)
    # Command at 6 us arrives at 9 us, even when the next call spans that event.
    driver.step(6e-6, [0], [300])
    expected_peak = 3.3 * (1 - np.exp(-6e-6 / 2e-6))
    np.testing.assert_allclose(driver.voltage_v, [expected_peak * np.exp(-3e-6 / 2e-6)])


def test_slew_limited_settling_and_tcr_compliance():
    driver = HeaterDriver(
        ActuatorParams(settling_time_s=1e-6, slew_rate_v_per_s=1e5, max_current_a=0.001, max_power_w=0.0001)
    )
    power = driver.step(1e-6, [4095], [350])
    np.testing.assert_allclose(driver.voltage_v, [0.1])
    np.testing.assert_allclose(driver.resistance_ohm, [505])
    assert power[0] <= 0.0001
    driver.step(0.001, [4095], [350])
    assert driver.current_a[0] <= 0.001
    np.testing.assert_allclose(driver.power_w, [0.0001], rtol=1e-8)


def test_instantaneous_driver_and_code_clipping():
    driver = HeaterDriver(ActuatorParams(settling_time_s=0, max_power_w=0.1), 2)
    np.testing.assert_allclose(driver.step(1e-6, [-100, 9000], [300, 300]), [0, 3.3**2 / 500])
    np.testing.assert_array_equal(driver.codes, [0, 4095])
    driver.step(0, [4095, 0], [300, 300])
    np.testing.assert_allclose(driver.voltage_v, [3.3, 0])


def test_existing_initial_command_holds_before_next_command_latency():
    driver = HeaterDriver(ActuatorParams(latency_s=4e-6, settling_time_s=1e-6, max_power_w=0.1))
    driver.prime([4095], [300])
    np.testing.assert_allclose(driver.voltage_v, [3.3])
    driver.step(3e-6, [0], [300])
    np.testing.assert_allclose(driver.voltage_v, [3.3])
    driver.step(2e-6, [0], [300])
    np.testing.assert_allclose(driver.voltage_v, [3.3 * np.exp(-1)])
    with pytest.raises(ValueError, match="before time advances"):
        driver.initialize([0], [300])


def test_sensor_exact_filter_quantization_and_delay_timestamp():
    params = SensorParams(
        input_noise_density_a_per_sqrt_hz=0,
        shot_noise=False,
        bandwidth_hz=1000,
        dark_current_a=0,
        latency_s=3e-4,
    )
    sensor = MonitorADC(params)
    sensor.step(1e-4, [1e-3])
    expected = 0.8e-3 * (1 - np.exp(-2 * np.pi * 0.1))
    np.testing.assert_allclose(sensor.filtered_current_a, [expected])
    assert not sensor.sample().valid
    sensor.step(2e-4, [1e-3])
    assert not sensor.sample().valid
    sensor.step(1e-4, [1e-3])
    observed = sensor.sample()
    assert observed.valid
    assert observed.timestamp_s == pytest.approx(1e-4)
    assert observed.delivered_time_s == pytest.approx(4e-4)
    assert abs(observed.current_a[0] - expected) <= 0.5 * params.adc_full_scale_a / (2**params.adc_bits - 1)


def test_sensor_zero_latency_saturation_and_snapshot_copy():
    sensor = MonitorADC(SensorParams(shot_noise=False, input_noise_density_a_per_sqrt_hz=0))
    sensor.step(0.01, [1.0])
    observation = sensor.sample()
    assert observation.valid and observation.clipped[0]
    assert observation.codes[0] == 16383
    observation.codes[:] = 0
    assert sensor.latest_observation.codes[0] == 16383


def test_filtered_noise_matches_one_sided_psd_variance():
    bandwidth = 1000.0
    noise_density = 1e-6
    # Independent stationary draws by evolving many channels beyond correlation.
    sensor = MonitorADC(
        SensorParams(
            shot_noise=False,
            dark_current_a=0,
            input_noise_density_a_per_sqrt_hz=noise_density,
            bandwidth_hz=bandwidth,
        ),
        30000,
        np.random.default_rng(42),
    )
    sensor.step(0.01, np.zeros(30000))
    expected_variance = noise_density**2 * np.pi * bandwidth / 2
    assert np.var(sensor.filtered_current_a) == pytest.approx(expected_variance, rel=0.025)


@pytest.mark.parametrize(
    "constructor,kwargs",
    [
        (ActuatorParams, {"dac_bits": 0}),
        (ActuatorParams, {"heater_resistance_ohm": 0}),
        (ActuatorParams, {"latency_s": -1}),
        (SensorParams, {"bandwidth_hz": 0}),
        (SensorParams, {"adc_gain": -1}),
    ],
)
def test_invalid_hardware_parameters(constructor, kwargs):
    with pytest.raises(ValueError):
        constructor(**kwargs)
