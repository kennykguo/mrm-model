"""The g26 aggressor sites (docs/handoff/integrator.md section 7).

The default site ``aggressor`` (0, -30) um is the mirror image of the former default (0, 30) um, which is kept as the
stress site ``aggressor_pd_adjacent``: both are 30 um from ring 0 and the same distance from every ring, so the
aggressor -> ring and -> heater-resistor paths, the A1/A2 power calibration and the rings' trajectories are the same,
bit for bit. Only the monitor-PD paths differ (PD0 55 um from the default site, 5 um from the stress site); the die
temperature sensor (~1 mm away) sees both sites alike.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mrm import params
from mrm.scenarios import library as lib


@pytest.fixture(scope="module")
def g26():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26")
        pp = params.realize(cfg, 1, seeds=0, uncertainty="nominal", variation="off")
    return cfg, pp


def _lags(sysm, src, obs):
    m = np.flatnonzero((sysm.lag_src == src) & (sysm.lag_obs == obs))
    return sysm.taus_s[:, m], sysm.res_k_per_w[:, m]


def test_g26_sites_and_provenance(g26):
    cfg, _ = g26
    sites = cfg.get("layout.sites")
    assert (sites["aggressor"]["x_um"], sites["aggressor"]["y_um"], sites["aggressor"]["kind"]) == (0.0, -30.0,
                                                                                                   "aggressor")
    st = sites[lib.AGGRESSOR_PD_ADJACENT_SITE]
    assert (st["x_um"], st["y_um"], st["kind"]) == (0.0, 30.0, "aggressor") and "stress" in st["note"]
    leaf = cfg.leaves["layout.sites"]
    assert leaf.provenance == "assumed"
    pd = np.asarray(cfg.get("ring.thermal.pd_offset_um"), float)
    assert np.hypot(*(pd - (0.0, -30.0))) == pytest.approx(55.0)          # default site to ring 0's monitor PD
    assert np.hypot(*(pd - (0.0, 30.0))) == pytest.approx(5.0)            # stress site to ring 0's monitor PD


def test_ring_and_sensor_paths_are_identical_only_the_pd_paths_differ(g26):
    _, pp = g26
    sysm = pp.thermal.system
    names = [sysm.src_names[i] for i in sysm.aggressor_src]
    assert names == [lib.AGGRESSOR_SITE, lib.AGGRESSOR_PD_ADJACENT_SITE]
    a_def, a_pdj = (int(i) for i in sysm.aggressor_src)
    same = [*sysm.ring_obs, *sysm.heater_obs, *sysm.temp_obs]
    assert len(sysm.temp_obs) >= 1                                       # the die sensor (feedforward_pi reads it)
    for o in same:
        t1, r1 = _lags(sysm, a_def, int(o))
        t2, r2 = _lags(sysm, a_pdj, int(o))
        assert r1.size and np.array_equal(t1, t2) and np.array_equal(r1, r2), sysm.obs_names[int(o)]
    dc = [[float(_lags(sysm, a, int(o))[1].sum()) for a in (a_def, a_pdj)] for o in sysm.pd_obs]
    assert dc[0][0] < 0.1 * dc[0][1]                                     # PD0: 55 um vs 5 um (interim site law)
    assert all(d[0] <= d[1] for d in dc)                                 # every PD is at least as far as before


def test_a1_rings_identical_on_both_sites_only_pd_differs(g26):
    """Engine, A1 near its hot cusp, 3 ms from a settled start: the calibrated power is the same on both sites and
    the rings' detuning and temperatures are identical bit for bit; the PD temperatures differ."""
    runner = pytest.importorskip("mrm.sim.runner")
    from mrm.sim.metrics import RecordSpec
    cfg, pp = g26
    rec = [RecordSpec("detuning_m", 10, "mean"), RecordSpec("t_ring_k", 10, "mean"), RecordSpec("t_pd_k", 10, "max")]
    out = {}
    for key, sc in (("default", lib.a1_slide10_aggressor(phase_s=0.0)),
                    ("stress", lib.a1_slide10_aggressor_pd_adjacent(phase_s=0.0))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = runner.run(pp, None, sc, duration_s=3e-3, cfg=cfg, heater_code="quasi_static", settle=True,
                           reset_protocol="branch_high", record=rec)
        cal = r.manifest_inputs["aggressor_calibration"]
        out[key] = ({k: np.asarray(v["value"]) for k, v in r.trace.items()},
                    float(np.max(next(iter(cal.values()))["peak_power_w"])))
    (tr_d, p_d), (tr_s, p_s) = out["default"], out["stress"]
    assert p_d == p_s > 0.1                                              # same calibrated power, W
    for k in ("detuning_m@10:mean", "t_ring_k@10:mean"):
        assert np.array_equal(tr_d[k], tr_s[k]), k
    pd0_d, pd0_s = tr_d["t_pd_k@10:max"][:, 0, 0], tr_s["t_pd_k@10:max"][:, 0, 0]
    assert np.max(pd0_s - pd0_d) > 100.0                                 # K: the stress site heats PD0 far more
