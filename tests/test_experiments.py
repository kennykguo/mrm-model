import json

import numpy as np
import pytest

from ringlab.config import load
from ringlab.experiments import draw_uncertainty, rerun, save
from ringlab.plant import simulate
from ringlab.thermal import AssumedThermalParameters, default_ring_network


def test_saved_run_replays_injected_physics_and_rejects_tampering(tmp_path):
    cfg = load(overrides={"simulation.duration_s": 10e-6, "data.averaging": "cw",
                          "controller.name": "hold", "controller.parameters": {"code": 500}})
    custom = default_ring_network(1, AssumedThermalParameters(heater_capacity_j_per_k=9e-10))
    result = simulate(cfg, thermal_network=custom)
    path = save(result, tmp_path, plot=False)
    new = rerun(path, tmp_path)
    with np.load(path/"trace.npz") as a, np.load(new/"trace.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    data = json.loads((path/"junction_model.json").read_text())
    data["parameters"]["frequency_shift_scale_hz"] *= 2
    (path/"junction_model.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="physical model changed"):
        rerun(path, tmp_path)


def test_uncertainty_is_stratified_repeatable_and_shared_by_controller_cases():
    spec = {"ring.absorption_fraction": {"kind": "uniform", "low": .2, "high": .8}}
    a = draw_uncertainty(spec, 8, 4)
    assert a == draw_uncertainty(spec, 8, 4)
    values = np.array([v["ring.absorption_fraction"] for v in a])
    assert sorted(np.floor((values-.2)/.6*8).astype(int)) == list(range(8))


def test_memory_budget_is_checked_before_record_allocations():
    cfg = load(overrides={"simulation.duration_s": 1e6})
    with pytest.raises(ValueError, match="trace memory"):
        simulate(cfg)
