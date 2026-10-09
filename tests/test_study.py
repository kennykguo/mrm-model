from dataclasses import replace

import pytest

from mrm_model.config import default_config
from mrm_model.study import apply_overrides, corner_study


def test_corner_paths_validated_and_original_preserved():
    nominal = default_config()
    corner = apply_overrides(nominal, {"optical.0.coupling_drop_power": 0.02})
    assert corner.optical[0].coupling_drop_power == 0.02
    assert nominal.optical[0].coupling_drop_power == 0.01
    with pytest.raises(ValueError, match="path"):
        apply_overrides(nominal, {"optical.0.imaginary_parameter": 1})
    with pytest.raises(ValueError, match="coupling"):
        apply_overrides(nominal, {"optical.0.coupling_drop_power": 1.5})


def test_corner_initial_resonance_error_not_rezeroed_and_calibration_not_refit():
    config = default_config()
    config = replace(config, run=replace(config.run, duration_s=0.0001))
    f0 = config.optical[0].reference_frequency_hz
    summaries = list(
        corner_study(
            config,
            [
                {"name": "nominal", "overrides": {}},
                {"name": "offset", "overrides": {"optical.0.reference_frequency_hz": f0 + 100e9}},
            ],
            kinds=("open_loop",),
        )
    )
    base = summaries[0]["metrics"]["open_loop"]
    shifted = summaries[1]["metrics"]["open_loop"]
    assert base["target_detuning_hz_per_lane"] == shifted["target_detuning_hz_per_lane"]
    assert shifted["tracking_rms_hz_per_lane"][0] > 50e9
    assert "fixed nominal" in summaries[1]["calibration_policy"]
