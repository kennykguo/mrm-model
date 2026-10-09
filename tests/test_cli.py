import numpy as np

from mrm_model.cli import main, prbs9
from mrm_model.config import ExperimentConfig


def test_prbs9_maximal_period_and_balance():
    bits = prbs9(1022)
    np.testing.assert_array_equal(bits[:511], bits[511:])
    assert bits[:511].sum() == 256
    cyclic = np.r_[bits[:511], bits[:8]]
    words = {tuple(cyclic[i : i + 9]) for i in range(511)}
    assert len(words) == 511
    assert (0,) * 9 not in words


def test_cli_preset_and_spectrum(tmp_path):
    profile = tmp_path / "profile.json"
    assert main(["preset", "--rings", "1", "--output", str(profile)]) == 0
    assert len(ExperimentConfig.load(profile).optical) == 1
    assert main(["spectrum", "--config", str(profile), "--output", str(tmp_path / "spectrum")]) == 0
    data = np.genfromtxt(tmp_path / "spectrum/spectrum.csv", delimiter=",", names=True)
    assert len(data) == 8001
    assert np.all(data["through_power_w"] >= 0)


def test_cli_unknown_config_fields_reports_error(tmp_path, capsys):
    profile = tmp_path / "invalid.json"
    profile.write_text('{"surprise":true}')
    assert main(["run", "--config", str(profile)]) == 2
    assert "Unknown configuration fields" in capsys.readouterr().err
