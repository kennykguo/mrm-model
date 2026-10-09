import pytest

from ringlab.config import load


def test_resolved_round_trip_keeps_values_and_evidence():
    cfg = load(overrides={"ring.absorption_fraction": 0.65, "simulation.seed": 19})
    other = load(cfg.resolved())
    assert other.digest() == cfg.digest()
    assert other.evidence["ring.absorption_fraction"]["status"] == "override"
    assert other.ring.absorption_fraction == .65


@pytest.mark.parametrize("override", [
    {"heater.bits": 12.5}, {"array.count": 1.5}, {"data.swing_v": 20},
    {"simulation.duration_s": -1}, {"sensor.sample_period_s": 5.5e-6},
    {"simulation.duration_s": 1e-20}, {"sensor.sample_period_s": 1e-20},
    {"ring.misspelled_parameter": 10}, {"scenario.kind": "unknown"},
    {"laser.power_per_line_w": -1},
    {"array.count": 1.0}, {"array.spacing_hz": 0}, {"array.bus_loss_db": -1},
    {"heater.bits": True}, {"sensor.noise_enabled": "false"},
    {"simulation.initial_heater_code": .5}, {"simulation.seed": 1.0},
    {"laser.power_per_line_w": True}, {"scenario.aggressor_ring": .5},
    {"data.bit_rate_hz": "nan"}, {"array.resonance_errors_hz": "oops"},
    {"ring.intrinsic_loss_per_m": 0}, {"ring.beta2_s2_per_m": 1e-25},
    {"scenario.dropout_start_s": 1.5e-6},
    {"scenario.kind": "local_triangle", "scenario.aggressor_period_s": 3e-6},
    {"controller": "pid"}, {"controller.parameters": "oops"},
])
def test_invalid_experiments_fail_before_running(override):
    with pytest.raises((ValueError, TypeError)):
        load(overrides=override)


def test_same_parameter_path_overrides_any_model():
    cfg = load("examples/models/reference.yaml", ["heater.bits=14", "simulation.seed=42"])
    assert cfg.heater.bits == 14
    assert cfg.simulation.seed == 42


def test_scientific_notation_and_numeric_tuple_are_normalized():
    cfg = load(overrides={"data.bit_rate_hz": "50e9", "array.resonance_errors_hz": [0]})
    assert cfg.data.bit_rate_hz == 50e9
    assert cfg.array.resonance_errors_hz == (0.,)


def test_long_duration_does_not_hide_fractional_tick():
    with pytest.raises(ValueError, match="integer multiple"):
        load(overrides={"simulation.duration_s": 1000 + .5e-6})
    assert load(overrides={"simulation.duration_s": 1000.}).ticks(1000.) == 1_000_000_000
