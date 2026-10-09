from dataclasses import replace

import numpy as np
import pytest

from mrm_model.config import default_config
from mrm_model.simulation import nominal_context, run_experiment


@pytest.mark.parametrize("rail", ("upper", "lower"))
def test_saturated_adc_has_no_fictitious_nominal_observability(rail):
    config = default_config()
    config = replace(
        config,
        run=replace(config.run, input_power_w=0.01, optical_heating=False, duration_s=0.0002),
        disturbance=replace(config.disturbance, kind="constant"),
        sensor=replace(
            config.sensor,
            adc_offset_a=0 if rail == "upper" else -0.01,
            shot_noise=False,
            input_noise_density_a_per_sqrt_hz=0,
        ),
    )
    with pytest.warns(RuntimeWarning, match="Initial ADC operating point"):
        context = nominal_context(config)
    assert context.target_fraction[0] == (1 if rail == "upper" else 0)
    np.testing.assert_array_equal(context.jacobian_per_code, 0)
    np.testing.assert_array_equal(context.c, 0)
    # An open-loop or independently identified/custom policy can still exercise
    # the real clipped plant; the built-in flank PI must reject no local slope.
    hold = replace(config, controller=replace(config.controller, kind="open_loop"))
    result = run_experiment(hold, context=context)
    expected_code = context.adc_max_code if rail == "upper" else 0
    np.testing.assert_array_equal(result.traces["adc_codes"], expected_code)
    assert result.metrics["adc_clipped_fraction_per_lane"][0] == pytest.approx(1)
    with pytest.raises(ValueError, match="observable nonzero slope"):
        run_experiment(config, context=context)
