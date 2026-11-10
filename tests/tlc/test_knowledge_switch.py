"""tools/tlc_run.py knowledge switch for corner runs: ``corner`` (the default) gives the plant and the controller the
corner's design; ``nominal`` (model mismatch) runs the plant on the corner and gives the controller the experiment's
own design without the corner alternatives. Thermal time constants of the nominal model tell the two designs apart."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from mrm.control import models as M
from mrm.control.api import ControllerSpec
from mrm.control.tlc.priors import g26_priors
from mrm.sim import experiment as E

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("tlc_run", ROOT / "tools" / "tlc_run.py")
tlc_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tlc_run)

STEADY = "experiments/tlc/steady_tlc_faithful.yaml"
SLOW_CORNER = {"uncertainty.thermal.time_scale_fast": "high", "uncertainty.thermal.time_scale_slow": "high"}
PI_LOCK = ["controller.name=pi_lock_c", "controller.params={}"]
SHORT_RUN = ["experiment.duration_s=0.002", "experiment.score_from_s=0.001",
             "experiment.self_reference={duration_s: 0.001, window_s: 0.0005}"]


# thermal time constants (s) of a design's nominal model - float64 (n_lags,)
def _thermal_taus_s(design) -> np.ndarray:
    cfg = design.model_config() if hasattr(design, "model_config") else design
    return np.asarray(M.nominal_plant_params(cfg).thermal.system.taus_s)[0]


class _SpecRecorder:
    # stands in for a controller - keeps the spec its reset receives
    def __init__(self):
        self.spec = None

    def reset(self, spec):
        self.spec = spec


@pytest.fixture(scope="module")
def designs():
    exp = E.load(STEADY, overrides=PI_LOCK)
    return {"nominal": exp.cfg, "corner": exp.config((), SLOW_CORNER)}


def test_corner_and_nominal_designs_differ(designs):
    assert not np.allclose(_thermal_taus_s(designs["nominal"]), _thermal_taus_s(designs["corner"]))


def test_knowledge_config_is_the_experiment_design_without_corner_alternatives(designs):
    assert tlc_run.knowledge_config(STEADY, PI_LOCK, "corner") is None
    cfg = tlc_run.knowledge_config(STEADY, PI_LOCK, "nominal")
    assert cfg.hash() == designs["nominal"].hash()
    with pytest.raises(ValueError, match="knowledge"):
        tlc_run.knowledge_config(STEADY, PI_LOCK, "truth")


def test_controller_receives_the_nominal_design_and_keeps_every_other_fact(designs):
    corner_spec = ControllerSpec.from_config(designs["corner"], n_batch=1, nominal=designs["corner"])
    ctrl = tlc_run.with_nominal_knowledge(_SpecRecorder(), designs["nominal"])
    ctrl.reset(corner_spec)
    assert np.allclose(_thermal_taus_s(ctrl.spec.nominal), _thermal_taus_s(designs["nominal"]))
    assert not np.allclose(_thermal_taus_s(ctrl.spec.nominal), _thermal_taus_s(corner_spec.nominal))
    assert ctrl.spec.knowledge == corner_spec.knowledge
    assert ctrl.spec.layout == corner_spec.layout and ctrl.spec.period_ticks == corner_spec.period_ticks


def test_tlc_priors_under_mismatch_are_the_nominal_priors(designs):
    corner_spec = ControllerSpec.from_config(designs["corner"], n_batch=1, nominal=designs["corner"])
    nominal_spec = ControllerSpec.from_config(designs["nominal"], n_batch=1, nominal=designs["nominal"])
    ctrl = tlc_run.with_nominal_knowledge(_SpecRecorder(), designs["nominal"])
    ctrl.reset(corner_spec)
    mismatch, nominal, corner = (g26_priors(s).pp for s in (ctrl.spec, nominal_spec, corner_spec))
    assert mismatch.tau_hf == pytest.approx(nominal.tau_hf) and mismatch.tau_hs == pytest.approx(nominal.tau_hs)
    assert mismatch.tau_hf != pytest.approx(corner.tau_hf)


@pytest.fixture
def created(monkeypatch):
    # every controller the experiment builds, in creation order
    made = []
    make = E._make_controller

    def spy(entry):
        ctrl = make(entry)
        made.append(ctrl)
        return ctrl

    monkeypatch.setattr(E, "_make_controller", spy)
    return made


@pytest.mark.parametrize("knowledge", ["corner", "nominal"])
def test_run_plant_follows_the_corner_and_knowledge_follows_the_switch(tmp_path, created, designs, knowledge):
    # the corner case uses the default, so it also checks that the default is the corner's own design
    switch = {} if knowledge == "corner" else {"knowledge": knowledge}
    run_dir = tlc_run.run_with_telemetry(STEADY, str(tmp_path), PI_LOCK + SHORT_RUN, SLOW_CORNER,
                                         {"corner": "g26_slow_thermal"}, **switch)
    info = json.loads((run_dir / "tlc_run.json").read_text())
    assert info["alts"] == SLOW_CORNER
    assert info.get("knowledge") == switch.get("knowledge")             # the default run record is unchanged
    world = json.loads((run_dir / "manifest.json").read_text())["cases"]["main"]["manifest_inputs"]["world"]
    assert world["thermal.time_scale_fast"] == [2.0] and world["thermal.time_scale_slow"] == [2.0]   # slow plant
    assert created and all(c.spec is not None for c in created)          # pre-run and main run
    want = _thermal_taus_s(designs[knowledge])
    for ctrl in created:
        assert np.allclose(_thermal_taus_s(ctrl.spec.nominal), want)
