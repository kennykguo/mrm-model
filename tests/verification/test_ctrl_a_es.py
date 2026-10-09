"""Extremum seeking, perturb-and-observe and sinusoidal (mrm/control/baselines/extremum_seeking.py): convergence to
the drop peak on a toy ring and on g26 at low laser power (no bistability), and the fold cycling at g26 nominal
power (CTRL_A_METRIC lines; run with -s)."""
from __future__ import annotations

import numpy as np
import pytest

import test_ctrl_a_harness as H
from mrm.control import registry
from mrm.control.baselines.extremum_seeking import E_FAULT, ExtremumSeekingPO, ExtremumSeekingSine

LOW_DB = -14.0          # +4 -> -10 dBm at the ring: bistability ratio ~6 -> ~0.24 (no fold)


def report(tag, m: H.LockMetrics, **extra):
    s = m.summary()
    s.update(extra)
    print("CTRL_A_METRIC", tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


@pytest.mark.parametrize("cls,kw", [(ExtremumSeekingPO, {}), (ExtremumSeekingPO, {"mode": "argmax"}),
                                    (ExtremumSeekingSine, {})])
def test_toy_converges_to_the_peak(cls, kw):
    spec = H.toy_spec(1, 3)
    u0 = np.array([[30000, 30500, 31000]], float)
    toy = H.ToyRings(spec, u0=u0, g=np.array([[0.3, 0.8, 1.2]]), u_init=u0[0] + 300, x0=3.0, noise=1.0)
    c = cls(init_codes=u0[0] + 300, **kw)
    c.reset(spec)
    toy.run(c, 3000)
    np.testing.assert_allclose(toy.x, 0.0, atol=0.06)
    assert np.all(c.telemetry()["fall_count"] == 0)


@pytest.mark.parametrize("cls", [ExtremumSeekingPO, ExtremumSeekingSine])
def test_g26_low_power_reaches_the_peak(cls):
    cfg, pp, spec, plant = H.placed(delta_pm=60.0, power_delta_db=LOW_DB)
    c = cls()
    loop = H.ClosedLoop(plant, spec, c)
    rec = loop.run(8000)
    d = rec["delta_pm"].astype(float)
    final = np.median(d[-1500:], axis=0)
    assert np.all(np.abs(final) < 15.0), final
    m = H.lock_metrics(rec, slice(-3000, -1500), slice(-1500, None), delta_ref=np.zeros_like(final))
    report(f"{c.name} g26 at -10 dBm per ring (peak lock)", m, centre_detuning_pm=float(np.mean(final)))


@pytest.mark.parametrize("cls", [ExtremumSeekingPO, ExtremumSeekingSine])
def test_g26_nominal_power_cycles_at_the_fold(cls):
    """At +4 dBm per ring the fold sits ~9 pm blue of the peak: the seekers fall off and recapture (07 P0-1)."""
    cfg, pp, spec, plant = H.placed(delta_pm=60.0)
    c = cls()
    loop = H.ClosedLoop(plant, spec, c)
    rec = loop.run(8000)
    tel = c.telemetry()
    assert np.all(tel["state"] != E_FAULT)
    d = rec["delta_pm"].astype(float)
    m = H.lock_metrics(rec, slice(0, 500), slice(2000, None), delta_ref=np.zeros(d.shape[1:]))
    report(f"{c.name} g26 nominal power (peak seeking)", m, falls=float(tel["fall_count"].sum()),
           min_detuning_pm=float(d.min()))


def test_registry():
    assert isinstance(registry.create("es_po", {"delta_codes": 8.0}), ExtremumSeekingPO)
    assert isinstance(registry.create("es_sine", {"dither_hz": 1000.0}), ExtremumSeekingSine)
    with pytest.raises(ValueError):
        ExtremumSeekingPO(mode="nope")
    with pytest.raises(ValueError):
        ExtremumSeekingSine(dither_hz=40e3).reset(H.toy_spec(1, 1))
