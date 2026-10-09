"""The controllers_a baselines through the compiled engine (mrm/sim/runner.py): an adapter check, slow (the first
engine compile takes minutes per path on a loaded machine, and ~2 GB: each path runs in its own subprocess so the
whole stays under the 3 GB memguard limit).

- The engine plant and the test harness agree open loop (same warm-start codes, same detuning trajectory).
- PiLock on the engine's Python path and PiLockCompiled on its compiled path give identical engine metrics, and lock.
  The engine starts the thermal lags cold (the harness settles them), so the controller holds the hand-off code for
  ``start_delay_s`` while the ms crosstalk and package-tail lags warm the ring; a ring that ends that warm-up on the
  cold branch violates the hand-off precondition (measured: 1 of 8).
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import test_ctrl_a_harness as H

runner = pytest.importorskip("mrm.sim.runner")
pytestmark = pytest.mark.slow
HERE = Path(__file__).resolve().parent

_SCRIPT = """
import sys, warnings, numpy as np
sys.path.insert(0, {here!r})
import test_ctrl_a_harness as H
from mrm.sim import runner
from mrm.sim.metrics import RecordSpec
from mrm.control.baselines.pi_lock import PiLock, PiLockCompiled
cfg, pp, spec, plant = H.placed()
codes = plant.lat_heater.copy()
ctrl = PiLock(start_delay_s={delay}) if {path!r} == "python" else PiLockCompiled(start_delay_s={delay})
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = runner.run(cfg, controller=ctrl, duration_s={dur}, uncertainty="nominal", variation="off",
                     heater_code=codes, reset_protocol="branch_high", record=[RecordSpec("detuning_m", 100, "last")])
tel = ctrl.telemetry()
keys = ("mean_detuning_pm", "rms_detuning_pm", "max_abs_detuning_pm", "excursion_count")
np.savez({out!r}, state=tel["state"], det=np.asarray(res.trace["detuning_m@100:last"]["value"]),
         wall=res.wall_s, sps=res.sim_s_per_wall_s, **{{k: np.asarray(res.metrics[k], float) for k in keys}})
"""


def _engine(path: str, out: Path, delay=0.03, dur=0.04):
    code = _SCRIPT.format(here=str(HERE), path=path, delay=delay, dur=dur, out=str(out))
    subprocess.run([sys.executable, "-c", code], check=True, timeout=7200)
    return dict(np.load(out))


def test_engine_matches_the_harness_open_loop():
    from mrm.sim.metrics import RecordSpec
    cfg, pp, spec, plant = H.placed()
    codes = plant.lat_heater.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = runner.run(cfg, controller=None, duration_s=2e-3, uncertainty="nominal", variation="off",
                         heater_code=codes, reset_protocol="branch_high",
                         record=[RecordSpec("detuning_m", 100, "last")])
    eng = np.asarray(res.trace["detuning_m@100:last"]["value"])[:, 0, :] * 1e12      # (S, N) pm
    lp = H.LoopPlant(pp, cfg, period_ticks=spec.period_ticks)
    lp.lat_heater[:] = codes
    lp.force_hot = True
    har = []
    for k in range(2000):
        lp.tick(k)
        if k % 100 == 99:
            har.append(lp.delta_pm[0].copy())
    har = np.array(har)
    n = min(eng.shape[0], har.shape[0])
    # the thermal states agree to ~0.01 K; the engine's detuning is ~1 % larger in magnitude than the harness's
    # rigid-translation Truth (lam_res + shift - lam_line) at ~1-2 nm detuning (measured: -1136.7 vs -1124.9 pm):
    # a conversion difference to settle with the engine owner (docs/handoff/controllers_a.md)
    # measured on 16 records x 8 rings: 159 of 160 within 2 %; the exception (ring 7 at 0.5 ms, 124 pm) is a ring
    # moving between branches during the cold-lag warm-up, where a sub-tick timing difference shows
    close = np.isclose(eng[:n], har[:n], rtol=0.02, atol=2.0)
    assert close.mean() >= 0.98, (close.mean(), eng[:n][~close], har[:n][~close])


def test_python_and_compiled_paths_agree_and_lock(tmp_path):
    from mrm.control.baselines.pi_lock import ST_LOCK
    a = _engine("python", tmp_path / "py.npz")
    b = _engine("compiled", tmp_path / "c.npz")
    for k in ("mean_detuning_pm", "rms_detuning_pm", "max_abs_detuning_pm", "excursion_count", "state"):
        np.testing.assert_array_equal(a[k], b[k])
    d = a["det"][:, 0, :] * 1e12
    final = np.median(d[-20:], axis=0)
    print("CTRL_A_METRIC pi_lock engine (python & compiled paths identical)",
          json.dumps({"state": a["state"].tolist(), "final_detuning_pm": np.round(final, 1).tolist(),
                      "sim_s_per_wall_s_python": float(a["sps"]), "sim_s_per_wall_s_compiled": float(b["sps"])}))
    # measured: 7 of 8 rings lock at 126-130 pm; ring 3 starts on the cold branch because the engine's warm start
    # begins with cold thermal lags (the hand-off precondition is not met), maps from there and faults
    lock = a["state"] == ST_LOCK
    assert lock.sum() >= 6, a["state"]
    assert np.all((final[lock[0]] > 80.0) & (final[lock[0]] < 180.0)), final
