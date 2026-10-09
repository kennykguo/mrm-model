from dataclasses import replace
from warnings import catch_warnings, simplefilter

import numpy as np
import pytest

from mrm_model.config import default_config
from mrm_model.hardware import ActuatorParams, HeaterDriver
from mrm_model.simulation import (
    _actuator_operating_derivatives,
    _steady_state,
    bus_response,
    nominal_context,
)


def settled_driver(params, codes, temperature):
    driver = HeaterDriver(
        replace(
            params,
            settling_time_s=0,
            latency_s=0,
            slew_rate_v_per_s=float("inf"),
            voltage_noise_std_v=0,
        ),
        len(codes),
    )
    driver.prime(codes, temperature)
    return driver


@pytest.mark.parametrize(
    "regime,overrides,tcr_sign",
    [
        ("interior", {}, -1),
        ("current", {"max_current_a": 0.002}, 1),
        ("power", {"max_power_w": 0.0005}, 0),
        ("voltage", {"max_voltage_v": 0.2}, -1),
        ("minimum", {"min_voltage_v": 0.6}, -1),
    ],
)
def test_local_actuator_derivatives_match_actual_settled_driver(regime, overrides, tcr_sign):
    params = replace(
        ActuatorParams(
            dac_bits=16,
            dac_reference_v=1.6,
            max_voltage_v=1.6,
            heater_resistance_ohm=56,
            resistance_tcr_per_k=0.001,
            max_current_a=0.03,
            max_power_w=0.04,
            settling_time_s=0,
        ),
        **overrides,
    )
    codes, temperature = np.array([20000.0]), np.array([330.0])
    derivatives = _actuator_operating_derivatives(params, codes, temperature)
    lower = settled_driver(params, codes - 1, temperature)
    upper = settled_driver(params, codes + 1, temperature)
    np.testing.assert_allclose(derivatives["dv_du"], (upper.voltage_v - lower.voltage_v) / 2, atol=1e-15)
    np.testing.assert_allclose(derivatives["dp_du"], (upper.power_w - lower.power_w) / 2, atol=1e-15)
    epsilon = 0.01
    colder = settled_driver(params, codes, temperature - epsilon)
    warmer = settled_driver(params, codes, temperature + epsilon)
    np.testing.assert_allclose(
        derivatives["dv_dt"], (warmer.voltage_v - colder.voltage_v) / (2 * epsilon), rtol=1e-7, atol=1e-15
    )
    np.testing.assert_allclose(
        derivatives["dp_dt"], (warmer.power_w - colder.power_w) / (2 * epsilon), rtol=1e-7, atol=1e-15
    )
    if regime == "interior":
        assert derivatives["dp_du"][0] > 0
    else:
        assert derivatives["dp_du"][0] == 0
    if tcr_sign:
        assert np.sign(derivatives["dp_dt"][0]) == tcr_sign
    else:
        assert abs(derivatives["dp_dt"][0]) < 1e-20


@pytest.mark.parametrize("settling_time", [0, 2e-6])
@pytest.mark.parametrize(
    "overrides,initial_codes",
    [
        ({}, (20000, 1000)),
        ({"max_current_a": 0.002}, (20000, 1000)),
        ({"max_power_w": 0.0005}, (20000, 1000)),
        ({"max_voltage_v": 0.2}, (20000, 1000)),
        ({"min_voltage_v": 0.05}, (1000, 20000)),
    ],
)
def test_nominal_dc_gain_matches_equilibrium_with_compliance_and_crosslane_tcr(
    overrides, initial_codes, settling_time
):
    config = default_config(2)
    config = replace(
        config,
        actuator=replace(config.actuator, settling_time_s=settling_time, **overrides),
        run=replace(config.run, optical_heating=False, initial_codes=initial_codes),
        disturbance=replace(config.disturbance, kind="constant"),
    )
    state, _ = _steady_state(config)
    # Put each otherwise-assumed ring on a useful flank without changing the
    # independently calculated thermal operating point.
    optical = tuple(
        replace(
            params,
            reference_frequency_hz=config.run.laser_frequency_hz[lane]
            - 25e9
            - params.df_dtemperature_hz_per_k
            * (state.ring_temperatures_k[lane] - params.reference_temperature_k),
        )
        for lane, params in enumerate(config.optical)
    )
    config = replace(config, optical=optical)
    with catch_warnings():
        simplefilter("ignore", RuntimeWarning)
        context = nominal_context(config)
    numerical = np.zeros((2, 2))
    for lane in range(2):
        offset = np.eye(2, dtype=int)[lane]
        lower = replace(
            config, run=replace(config.run, initial_codes=tuple(np.array(initial_codes) - offset))
        )
        upper = replace(
            config, run=replace(config.run, initial_codes=tuple(np.array(initial_codes) + offset))
        )
        lower_state, _ = _steady_state(lower)
        upper_state, _ = _steady_state(upper)
        lower_monitor = bus_response(lower, lower_state.ring_temperatures_k).monitor_power_w
        upper_monitor = bus_response(upper, upper_state.ring_temperatures_k).monitor_power_w
        numerical[:, lane] = (
            (upper_monitor - lower_monitor)
            / 2
            * config.sensor.adc_gain
            * config.sensor.responsivity_a_per_w
            / config.sensor.adc_full_scale_a
        )
    np.testing.assert_allclose(context.jacobian_per_code, numerical, rtol=5e-4, atol=2e-11)
    discrete_dc = context.c @ np.linalg.solve(np.eye(len(context.a)) - context.a, context.b)
    np.testing.assert_allclose(discrete_dc, context.jacobian_per_code, rtol=1e-8, atol=1e-13)
    if overrides:
        np.testing.assert_array_equal(context.jacobian_per_code[:, 0], 0)


def test_nonsmooth_threshold_is_explicitly_flagged():
    params = ActuatorParams(settling_time_s=0, dac_bits=16)
    threshold_voltage = params.dac_reference_v * 20000 / (2**params.dac_bits - 1)
    params = replace(params, max_voltage_v=threshold_voltage)
    with pytest.warns(RuntimeWarning, match="nonsmooth compliance threshold"):
        derivative = _actuator_operating_derivatives(params, [20000], [300])
    assert derivative["dp_du"][0] == 0


def test_compliant_dynamic_projection_is_not_claimed_globally_linear():
    params = ActuatorParams(max_power_w=1e-5, settling_time_s=1e-6)
    with pytest.warns(RuntimeWarning, match="nonlinear plant constraint"):
        _actuator_operating_derivatives(params, [2000], [300])
