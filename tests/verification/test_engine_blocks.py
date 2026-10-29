"""Checks that the auto substep policy gives the same run however the runner cuts it into kernel blocks.

Read this first
---------------
Under ``substep_policy='auto'`` the end of a kernel block caps the substep in flight, and the growth limit then
restarts from the shortened step. So the trajectory could depend on how the runner cut a run into kernel calls:
chunk_ticks 13 against 1e5 once gave 11426 against 10582 substeps and monitor codes up to 9 apart (g26 nominal,
pi_lock_c, 20 ms). A controller edge always ends a substep (a step never crosses one). So the runner ends every
kernel block of a run without tick jitter on a controller edge, and chunk_ticks only sets how often the kernel
returns. The scoring split (``score_from_s``) must also end a substep at its own tick. Under the auto policy it
changes nothing only when it falls on a controller edge, and the runner warns for any other tick. See
docs/ENGINE.md, "Metrics and when scoring begins" and "Fixed and automatic substeps". ~30 s with a warm numba cache.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mrm import params
from mrm.sim import runner
from mrm.sim.metrics import RecordSpec

RECORDS = [RecordSpec("detuning_m", 25, "mean"), RecordSpec("monitor_code", 25, "last"),
           RecordSpec("heater_code", 25, "last")]       # every 25 ticks


@pytest.fixture(scope="module")
def cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26")


def _run(cfg, controller="pi_lock_c", **kw):
    """A 4 ms nominal run from the locked steady state under the auto policy. Keyword arguments override the run
    options. res.split_warnings lists the ScoreSplitWarning messages."""
    opts = dict(duration_s=4e-3, uncertainty="nominal", variation="off", heater_code="locked_steady_state", settle=True,
                substep_policy="auto", record=RECORDS)
    opts.update(kw)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = runner.run(cfg, controller, None, **opts)
    res.split_warnings = [str(x.message) for x in w if issubclass(x.category, runner.ScoreSplitWarning)]
    return res


def _assert_same_run(a, b, metrics=True):
    """Assert bit-identical traces, substep counts and final detuning (and the main metrics if metrics is true)."""
    for key in a.trace:
        np.testing.assert_array_equal(a.trace[key]["value"], b.trace[key]["value"], err_msg=key)
    for key in ("max_abs_detuning_pm", "rms_detuning_pm", "mean_detuning_pm", "heater_energy_j") if metrics else ():
        np.testing.assert_array_equal(a.metrics[key], b.metrics[key], err_msg=key)
    np.testing.assert_array_equal(a.diagnostics["substeps"], b.diagnostics["substeps"])
    np.testing.assert_array_equal(a.truth_final.fields["detuning_m"], b.truth_final.fields["detuning_m"])


@pytest.mark.parametrize("controller", ["pi_lock_c", None], ids=["compiled", "open_loop"])
def test_auto_policy_results_do_not_depend_on_chunk_ticks(cfg, controller):
    ref = _run(cfg, controller)
    for chunk in (13, 777):
        res = _run(cfg, controller, chunk_ticks=chunk)
        _assert_same_run(ref, res)
        assert res.manifest_inputs["chunk_ticks"] == chunk
    assert ref.manifest_inputs["chunk_ticks"] == 100_000


def test_score_split_on_an_edge_keeps_the_auto_trajectory(cfg):
    """A split on a controller edge (g26: every 10 ticks) ends no substep that the unsplit run takes: plant,
    controller and records are bit for bit those of the unscored run. A split elsewhere warns under the auto policy
    (it ends a substep there) and not under the fixed policy (nothing changes)."""
    ref = _run(cfg)
    on_edge = _run(cfg, score_from_s=2e-3)
    _assert_same_run(ref, on_edge, metrics=False)
    assert on_edge.manifest_inputs["score_from_tick"] == 2000 and not on_edge.split_warnings
    off_edge = _run(cfg, score_from_s=2.003e-3)
    assert len(off_edge.split_warnings) == 1 and "controller edge" in off_edge.split_warnings[0]
    fixed = [_run(cfg, substep_policy="fixed", score_from_s=s) for s in (None, 2.003e-3)]
    assert not fixed[1].split_warnings
    for key in fixed[0].trace:
        np.testing.assert_array_equal(fixed[0].trace[key]["value"], fixed[1].trace[key]["value"], err_msg=key)
