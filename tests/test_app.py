"""Exercise the actual Streamlit UI callbacks without an external browser service."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_lab_starts_and_runs_comparison_and_waveform():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=60).run()
    assert not app.exception
    next(button for button in app.button if button.label == "Run comparison").click().run()
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) == 1
    results = app.session_state["results"]
    assert set(results) == {"open_loop", "pi"}
    next(button for button in app.button if button.label == "Simulate waveform").click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["waveform"]["samples_per_bit"] == 32
    assert app.session_state["waveform_inputs"]["config"]["run"]["input_power_w"] == 100e-6
    next(widget for widget in app.number_input if widget.label == "Data rate (Gb/s)").set_value(60.0).run()
    assert any("waveform belongs to previous" in warning.value for warning in app.warning)
    next(button for button in app.button if button.label == "Simulate waveform").click().run()
    assert not app.exception
    assert not any("waveform belongs to previous" in warning.value for warning in app.warning)
    app.text_area(key="config_text").set_value(
        app.session_state["config_text"].replace('"input_power_w": 0.0001', '"input_power_w": 0.0002')
    ).run()
    assert any("waveform belongs to previous" in warning.value for warning in app.warning)


def test_lab_reports_invalid_json_and_loads_preset():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=60).run()
    app.text_area(key="config_text").set_value('{"invalid": true}').run()
    assert app.error
    assert not app.exception
    next(button for button in app.button if button.label == "Load these settings").click().run()
    assert not app.error
    assert not app.exception
