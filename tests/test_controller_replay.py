import importlib
import json
import sys

import numpy as np
import pytest
import yaml

from ringlab import provenance
from ringlab.config import load
from ringlab.controllers import DitherLock, Hold, PID, ScanPI, snapshot
from ringlab.experiments import compare, draw_uncertainty, rerun, save
from ringlab.plant import simulate


def configuration(**overrides):
    settings = {"simulation.duration_s": 60e-6, "simulation.record_period_s": 5e-6,
                "data.averaging": "cw", "controller.name": "hold",
                "controller.parameters": {"code": 500}, "sensor.noise_enabled": False}
    settings.update(overrides)
    return load(overrides=settings)


def assert_same_trace(first, second):
    with np.load(first / "trace.npz") as a, np.load(second / "trace.npz") as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("controller", [
    Hold(np.array([777])),
    PID(target_code=220, initial_code=900, kp=.1, ki_per_s=40, kd_s=1e-7, sign=-1),
    ScanPI(scan_steps=8, settle_frames=1, target_fraction=.7, kp=.2, ki_per_s=30, minimum_peak_code=3),
    DitherLock(initial_code=850, amplitude_codes=7, gain=.2, dwell_frames=1),
])
def test_actual_injected_builtin_constructor_is_saved_before_reset(controller, tmp_path):
    expected = snapshot(controller)
    result = simulate(configuration(), controller=controller)
    # Runtime mutations after simulate must not affect the captured settings.
    if isinstance(controller, Hold):
        controller.code[0] = 999
    if isinstance(controller, ScanPI):
        assert isinstance(controller.target, np.ndarray)
    path = save(result, tmp_path, plot=False)
    saved = json.loads((path / "controller.json").read_text())
    assert saved["name"] == expected["name"]
    for key, value in expected["parameters"].items():
        np.testing.assert_array_equal(saved["parameters"][key], value)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["source_sha256"] == result.source_sha256
    replay = rerun(path, tmp_path)
    assert_same_trace(path, replay)


class UnknownController:
    def reset(self, interface):
        self.action = np.full(interface.rings, 123, np.int64)

    def step(self, observation):
        return self.action.copy()


def test_unknown_injected_controller_runs_but_cannot_silently_save_config_controller(tmp_path):
    result = simulate(configuration(), controller=UnknownController())
    assert result.controller_spec is None
    with pytest.raises(ValueError, match="unknown injected controller"):
        save(result, tmp_path / "output", plot=False)
    assert not (tmp_path / "output").exists()


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    name = "replay_test_custom_controller"
    file = tmp_path / f"{name}.py"
    file.write_text("""import numpy as np
class CustomController:
    def __init__(self, code=73):
        self.code = code
    def reset(self, interface):
        self.action = np.full(interface.rings, self.code, np.int64)
        self.code = -1  # original constructor input is no longer recoverable
    def step(self, observation):
        return self.action.copy()
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    from ringlab import controllers
    monkeypatch.setattr(controllers, "_PLUGIN_OBSERVED_SOURCE", {})
    importlib.invalidate_caches()
    yield f"{name}:CustomController", file
    sys.modules.pop(name, None)


def test_configured_plugin_parameters_and_source_are_saved(plugin, tmp_path):
    name, file = plugin
    cfg = configuration(**{"controller.name": name, "controller.parameters": {"code": 81}})
    result = simulate(cfg)
    assert result.controller.code == -1
    path = save(result, tmp_path / "runs", plot=False)
    spec = json.loads((path / "controller.json").read_text())
    assert spec["parameters"] == {"code": 81}
    assert spec["source_sha256"][name.split(":")[0]]
    assert "external resources" in spec["capture_note"]
    assert_same_trace(path, rerun(path, tmp_path / "runs"))
    file.write_text(file.read_text() + "\n# changed implementation\n")
    with pytest.raises(ValueError, match="plugin source changed"):
        rerun(path, tmp_path / "runs")
    with pytest.raises(ValueError, match="plugin source changed"):
        save(result, tmp_path / "after_change", plot=False)


def test_injected_custom_controller_needs_explicit_importable_constructor_spec(plugin, tmp_path):
    name, _ = plugin
    module = importlib.import_module(name.split(":")[0])
    instance = module.CustomController(code=92)
    result = simulate(configuration(), controller=instance,
                      controller_configuration={"name": name, "parameters": {"code": 92}})
    path = save(result, tmp_path / "runs", plot=False)
    assert_same_trace(path, rerun(path, tmp_path / "runs"))
    with pytest.raises(ValueError, match="actual controller class"):
        simulate(configuration(), controller=UnknownController(),
                 controller_configuration={"name": name, "parameters": {"code": 92}})


def test_controller_file_tampering_is_rejected(tmp_path):
    path = save(simulate(configuration(), controller=Hold(321)), tmp_path, plot=False)
    spec_file = path / "controller.json"
    spec = json.loads(spec_file.read_text())
    spec["parameters"]["code"] = 322
    spec_file.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="controller specification changed"):
        rerun(path, tmp_path)


@pytest.mark.parametrize("field,changed,match", [("python", "0.0.0", "Python version"),
                                               ("platform", "different-platform", "platform")])
def test_replay_requires_explicit_opt_in_for_runtime_environment_changes(tmp_path, field, changed, match):
    path = save(simulate(configuration()), tmp_path, plot=False)
    manifest_file = path / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest[field] = changed
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        rerun(path, tmp_path)
    changed_path = rerun(path, tmp_path, allow_changed_model=True)
    new_manifest = json.loads((changed_path / "manifest.json").read_text())
    assert match in new_manifest["diagnostics"]["replay_changes"]


@pytest.mark.parametrize("change_before_run", [False, True])
def test_disk_source_edits_cannot_be_relabelled_as_loaded_source(tmp_path, monkeypatch, change_before_run):
    project = tmp_path / "source"
    (project / "src").mkdir(parents=True)
    file = project / "src" / "model.py"
    file.write_text("VERSION = 1\n")
    monkeypatch.setattr(provenance, "PROJECT", project)
    initial_hashes = provenance.source_hashes()
    monkeypatch.setattr(provenance, "_LOADED_SOURCE_HASHES", initial_hashes)
    if change_before_run:
        file.write_text("VERSION = 2\n")
    result = simulate(configuration())
    assert result.source_sha256 == initial_hashes
    if not change_before_run:
        file.write_text("VERSION = 2\n")
    with pytest.raises(ValueError, match="source changed after package import"):
        save(result, tmp_path / "output", plot=False)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("field,value", [("samples", 2.5), ("samples", True), ("samples", -1),
                                          ("seed", 2.5), ("seed", True), ("seed", -1)])
def test_study_counts_and_seeds_are_not_silently_truncated(tmp_path, field, value):
    model = tmp_path / "model.yaml"
    model.write_text(yaml.safe_dump(configuration().resolved()))
    study = {"model": "model.yaml", "controllers": [{"name": "hold"}], field: value}
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(study))
    with pytest.raises(ValueError, match="nonnegative integer"):
        compare(path, preview=True)


def test_uncertainty_cases_do_not_overwrite_named_cases(tmp_path):
    (tmp_path / "model.yaml").write_text(yaml.safe_dump(configuration().resolved()))
    study = {"model": "model.yaml", "controllers": [{"name": "hold"}], "samples": 1,
             "cases": {"sample_0000": {"controller.parameters": {"code": 12}}}}
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(study))
    with pytest.raises(ValueError, match="conflicts with a supplied case"):
        compare(path, preview=True)
    with pytest.raises(ValueError, match="nonnegative integer"):
        draw_uncertainty({}, 1.5, 2)
