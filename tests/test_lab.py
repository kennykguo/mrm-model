"""Fixed-profile identity, real trace provenance, and local HTTP boundaries."""
import copy
from http.client import HTTPConnection
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from ringlab.config import load
from ringlab import lab
from ringlab.plant import simulate


def test_model_hash_excludes_controller_and_experiment_but_tracks_physics():
    state = lab.LabState()
    baseline = state.configuration(lab.DEFAULTS, "step")
    altered = {**lab.DEFAULTS, "initial_code": 2500, "step_codes": 8, "ambient_step_k": -0.5}
    hold = state.configuration(altered, "hold")
    assert baseline.digest() != hold.digest()
    assert lab.model_sha256(baseline) == lab.model_sha256(hold)
    changed = load(hold, {"heater.resistance_ohm": 501.0})
    assert lab.model_sha256(changed) != lab.model_sha256(baseline)
    model = state.metadata()
    assert set(model["model"]) == {"version", *lab.PHYSICAL_SECTIONS}
    assert model["provenance"]["source_sha256"]
    assert model["provenance"]["environment"]["numpy"]
    assert model["provenance"]["evidence"]["ring.reference_wavelength_m"]["status"] == "published_context"


def test_profile_is_fully_resolved_and_metadata_is_a_copy():
    raw = yaml.safe_load(lab.PROFILE_PATH.read_text())
    resolved = load(raw).resolved()
    for section in (*lab.PHYSICAL_SECTIONS, "simulation", "scenario"):
        assert set(raw[section]) == set(resolved[section])
    state = lab.LabState()
    metadata = state.metadata()
    metadata["model"]["heater"]["resistance_ohm"] = 1
    metadata["defaults"]["step_codes"] = 64
    assert state.metadata()["model"]["heater"]["resistance_ohm"] == 500
    assert state.metadata()["defaults"]["step_codes"] == 4


def test_controller_overrides_never_mutate_frozen_profile_or_future_digests():
    state = lab.LabState()
    frozen = copy.deepcopy(state._profile.resolved())
    first = state.configuration(lab.DEFAULTS, "step")
    state.configuration({**lab.DEFAULTS, "initial_code": 2000}, "hold")
    second = state.configuration(lab.DEFAULTS, "step")
    assert state._profile.resolved() == frozen
    assert first.resolved() == second.resolved()
    assert first.digest() == second.digest()


def test_warmup_uses_canonical_ticks_and_accepts_boundary_sample():
    from ringlab.controllers import Interface, Observation, create
    state = lab.LabState()
    controls = {**lab.DEFAULTS, "warmup_ms": 0.515}
    configuration = state.configuration(controls, "step")
    expected = 515*configuration.simulation.tick_s
    assert configuration.controller["parameters"]["warmup_s"] == expected
    assert state._experiment(configuration, controls)["warmup_s"] == expected
    controller = create(**configuration.controller)
    controller.reset(Interface(1, 4095, 4095, 20e-6))
    controller.step(Observation(0.0, -1e-6, np.array([0]), np.array([False]), np.array([2660]), 0, False))
    action = controller.step(Observation(520e-6, expected, np.array([300]), np.array([False]),
                                         np.array([2660]), 6000, True))
    assert action.tolist() == [2664]


@pytest.mark.parametrize("key,value", [
    ("target_code", 0), ("target_code", 4095), ("target_code", 260.0),
    ("initial_code", -1), ("initial_code", 4096), ("step_codes", 0),
    ("step_codes", 65), ("deadband_codes", -1), ("deadband_codes", 51),
    ("warmup_ms", 0), ("warmup_ms", 1.001), ("warmup_ms", 0.5005),
    ("ambient_step_k", -1.01), ("ambient_step_k", 1.01),
    ("ambient_step_k", float("nan")), ("ambient_step_k", float("inf")),
    ("ambient_step_k", -float("inf")), ("target_code", 10**1000),
    ("step_codes", True), ("ambient_step_k", False), ("warmup_ms", "0.5"),
    ("target_code", None), ("initial_code", [2660]),
])
def test_control_bounds_reject_nonfinite_boolean_and_wrong_types(key, value):
    with pytest.raises(ValueError):
        lab.validate_controls({**lab.DEFAULTS, key: value})


@pytest.mark.parametrize("payload", [None, [], {}, {**lab.DEFAULTS, "configuration": "evil.yaml"},
                                     {**lab.DEFAULTS, "controller": "module:Class"}])
def test_only_exact_control_fields_are_accepted(payload):
    with pytest.raises(ValueError):
        lab.validate_controls(payload)


def test_control_bounds_accept_edges_and_json_rejects_duplicate_fields():
    for edge in ("min", "max"):
        payload = {key: limits[edge] for key, limits in lab.LIMITS.items()}
        assert lab.validate_controls(payload) == payload
    with pytest.raises(ValueError, match="duplicate"):
        lab.decode_controls(b'{"step_codes": 1, "step_codes": 2}')
    with pytest.raises(ValueError, match="finite"):
        lab.decode_controls(b'{"target_code": NaN}')
    with pytest.raises(ValueError):
        lab.decode_controls(b"\xff")


def test_pair_uses_same_frozen_physics_seed_and_experiment(monkeypatch):
    state = lab.LabState()
    seen = []

    def fake_simulate(configuration):
        seen.append(configuration)
        return configuration

    monkeypatch.setattr(lab, "simulate", fake_simulate)
    monkeypatch.setattr(state, "_run_result", lambda configuration, _: {"controller": configuration.controller})
    response = state.run({**lab.DEFAULTS, "initial_code": 2500, "ambient_step_k": -0.5})
    assert [c.controller["name"] for c in seen] == ["step", "hold"]
    for configuration in seen:
        assert configuration.simulation.duration_s == 0.004
        assert configuration.simulation.initial_heater_code == 2500
        assert configuration.simulation.seed == 7
        assert configuration.sensor.noise_enabled
        assert configuration.scenario.start_s == 0.002
        assert configuration.scenario.ambient_step_k == -0.5
        assert lab.model_sha256(configuration) == response["model_sha256"]
    assert response["experiment"]["simulation"]["initial_heater_code"] == 2500
    assert response["model"]["heater"]["resistance_ohm"] == 500
    assert seen[1].controller["parameters"]["code"] == 2500
    json.dumps(response, allow_nan=False)


def test_changed_loaded_sources_prevent_run(monkeypatch):
    def changed():
        raise ValueError("source changed")

    monkeypatch.setattr(lab, "require_unchanged_sources", changed)
    with pytest.raises(ValueError, match="source changed"):
        lab.LabState().run(lab.DEFAULTS)


def test_run_metrics_follow_physical_sample_time_and_warnings_report_limits():
    configuration = lab.LabState().configuration(lab.DEFAULTS, "step")
    result = SimpleNamespace(
        configuration=configuration,
        trace={"time_s": np.array([0.0, 0.002, 0.00202, 0.004]),
               "sample_time_s": np.array([-1e-6, 0.001995, 0.002015, 0.003995]),
               "monitor_code": np.array([[0], [100], [258], [262]]),
               "detuning_hz": np.array([[1e9], [-1e9], [-1e9], [1e9]])},
        metrics={"heater_rail_time_s": np.array([20e-6])},
        diagnostics={"sensor_clipped_samples": np.array([1])},
        controller_spec={"name": "step", "parameters": {}},
    )
    output = lab.LabState._run_result(result, lab.DEFAULTS)
    control = output["metrics"]["control"]
    assert control["post_disturbance_rms_error_code"] == 2
    assert control["post_disturbance_mean_abs_error_code"] == 2
    assert control["within_deadband_fraction"] == 1
    assert control["final_error_code"] == 2
    warnings = output["diagnostics"]["warnings"]
    assert len(warnings) == 4
    assert any("after warm-up" in warning for warning in warnings)
    assert any("final state" in warning for warning in warnings)
    assert any("code rail" in warning for warning in warnings)
    assert any("ADC sample clipped" in warning for warning in warnings)


def test_cli_routes_lab_to_loopback_server_and_bounds_port(monkeypatch):
    from ringlab.cli import main
    calls = []
    monkeypatch.setattr(lab, "serve", lambda port: calls.append(port) or 0)
    assert main(["lab"]) == 0
    assert main(["lab", "--port", "8766"]) == 0
    assert calls == [8765, 8766]
    for port in ("0", "65536", "-1"):
        with pytest.raises(SystemExit):
            main(["lab", "--port", port])


@pytest.fixture
def server():
    with lab.LabHTTPServer(0) as instance:
        thread = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            yield instance
        finally:
            instance.shutdown()
            thread.join(timeout=3)


def request(server, method, path, body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_http_serves_model_only_on_local_host_and_origin(server):
    status, headers, body = request(server, "GET", "/api/model")
    assert status == 200
    assert json.loads(body)["model_id"] == lab.MODEL_ID
    assert "Access-Control-Allow-Origin" not in headers
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert headers["Cache-Control"] == "no-store"
    for extra in ({"Host": "evil.example"}, {"Origin": "https://evil.example"},
                  {"Origin": "null"}, {"Origin": "http://127.0.0.1:1"},
                  {"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"}):
        status, _, body = request(server, "GET", "/api/model", headers=extra)
        assert status == 403
        assert "error" in json.loads(body)
    origin = f"http://127.0.0.1:{server.server_port}"
    assert request(server, "GET", "/api/model", headers={"Origin": origin})[0] == 200


def test_http_whitelist_never_exposes_arbitrary_files(server, tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("local UI")
    (tmp_path / "secret.txt").write_text("not an asset")
    monkeypatch.setattr(lab, "STATIC_PATH", tmp_path)
    assert request(server, "GET", "/")[2] == b"local UI"
    for path in ("/secret.txt", "/../lab_model.yaml", "/%2e%2e/lab_model.yaml", "/api/run", "/?path=secret.txt"):
        assert request(server, "GET", path)[0] == 404
    assert request(server, "OPTIONS", "/api/run")[0] == 405


def test_http_run_validates_request_and_returns_json(server, monkeypatch):
    seen = []

    def run(controls):
        seen.append(controls)
        return {"runs": {"step": {"trace": {"time_s": np.array([0.0])}}}}

    monkeypatch.setattr(server.lab, "run", run)
    headers = {"Content-Type": "application/json"}
    status, _, body = request(server, "POST", "/api/run", json.dumps(lab.DEFAULTS), headers)
    assert status == 200
    assert json.loads(body)["runs"]["step"]["trace"]["time_s"] == [0.0]
    assert seen == [lab.DEFAULTS]
    for body, extra in (("{}", headers), ("not json", headers),
                        ("{}", {"Content-Type": "text/plain"}),
                        (" "*(lab.MAX_BODY_BYTES+1), headers),
                        (json.dumps({**lab.DEFAULTS, "warmup_ms": True}), headers),
                        ("{}", {**headers, "Transfer-Encoding": "chunked"})):
        assert request(server, "POST", "/api/run", body, extra)[0] == 400
    assert len(seen) == 1


def test_http_one_inflight_run_and_sanitized_errors(server, monkeypatch):
    headers = {"Content-Type": "application/json"}
    body = json.dumps(lab.DEFAULTS)
    with server.lab.run_lock:
        assert request(server, "POST", "/api/run", body, headers)[0] == 409

    def fail(_):
        raise RuntimeError("private filesystem path must not be returned")

    monkeypatch.setattr(server.lab, "run", fail)
    status, _, response = request(server, "POST", "/api/run", body, headers)
    assert status == 500
    assert b"private filesystem" not in response
    assert "error" in json.loads(response)
    assert not server.lab.run_lock.locked()


def test_server_bind_is_loopback_only():
    with lab.LabHTTPServer(0) as instance:
        assert instance.server_address[0] == "127.0.0.1"
    for invalid in (True, "8765", -1, 65536):
        with pytest.raises(ValueError):
            lab.LabHTTPServer(invalid)


def test_real_short_trace_temperature_fields_and_fixed_seed_replay():
    configuration = lab.LabState().configuration(lab.DEFAULTS, "hold")
    configuration = load(configuration, {"simulation.duration_s": 40e-6,
                                          "simulation.record_period_s": 1e-6,
                                          "scenario.start_s": 20e-6})
    first, second = simulate(configuration), simulate(configuration)
    for key in first.trace:
        np.testing.assert_array_equal(first.trace[key], second.trace[key])
    count = len(first.trace["time_s"])
    assert first.trace["heater_temperature_k"].shape == (count, 1)
    assert first.trace["package_temperature_k"].shape == (count,)
    assert first.trace["ambient_temperature_k"].shape == (count,)
    assert first.trace["heater_temperature_k"][0, 0] == 300
    assert first.trace["package_temperature_k"][0] == 300
    np.testing.assert_array_equal(first.trace["ambient_temperature_k"][:20], 300)
    np.testing.assert_array_equal(first.trace["ambient_temperature_k"][20:], 300.25)
    assert first.diagnostics["trace_bytes"] == count*(11+4)*8
    encoded = lab.LabState._run_result(first, lab.DEFAULTS)
    json.dumps(encoded, allow_nan=False)


@pytest.mark.slow
def test_real_fixed_lab_pair_produces_matching_experiments_and_controller_effect():
    result = lab.LabState().run(lab.DEFAULTS)
    step, hold = (result["runs"][name] for name in ("step", "hold"))
    for run in (step, hold):
        trace = run["trace"]
        assert len(trace["time_s"]) == 201
        assert trace["time_s"][0] == 0
        assert trace["time_s"][-1] == pytest.approx(0.004)
        assert trace["heater_code"][0] == [2660]
        assert run["diagnostics"]["numerical_method"] == "Radau"
        assert run["diagnostics"]["rhs_calls"] > 0
        assert run["diagnostics"]["sensor_clipped_samples"] == [0]
        assert run["metrics"]["control"]["post_disturbance_rms_error_code"] is not None
    assert step["trace"]["ambient_temperature_k"] == hold["trace"]["ambient_temperature_k"]
    assert step["trace"]["monitor_code"][:25] == hold["trace"]["monitor_code"][:25]
    assert all(code == [2660] for code in hold["trace"]["heater_code"])
    assert any(code != [2660] for code in step["trace"]["heater_code"])
    assert result["controls"] == lab.DEFAULTS
    json.dumps(result, allow_nan=False)
