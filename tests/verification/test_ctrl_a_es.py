"""Checks the extremum-seeking controllers (mrm/control/baselines/extremum_seeking.py).

Read this first
---------------
es_po (perturb and observe) and es_sine (sinusoidal dither) climb to the drop peak. The tests check:

- convergence to the drop peak on a toy ring, and on g26 at low laser power (no bistability);
- at g26 nominal power: es_po falls off the self-heating fold and recaptures; es_sine never falls and approaches the
  peak slowly (CTRL_A_METRIC lines; run with -s);
- on the engine, es_sine parks at the drop peak after ~150 ms (slow).
"""
from __future__ import annotations

import numpy as np
import pytest

import test_ctrl_a_harness as harness
from mrm.control import registry
from mrm.control.baselines.extremum_seeking import E_FAULT, ExtremumSeekingPO, ExtremumSeekingSine

LOW_DB = -14.0          # +4 -> -10 dBm at the ring: bistability ratio ~6 -> ~0.24 (no fold)


def report(tag, m: harness.LockMetrics, **extra):
    s = m.summary()
    s.update(extra)
    print("CTRL_A_METRIC", tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


@pytest.mark.parametrize("cls,kw", [(ExtremumSeekingPO, {}), (ExtremumSeekingPO, {"mode": "argmax"}),
                                    (ExtremumSeekingSine, {})])
def test_toy_converges_to_the_peak(cls, kw):
    spec = harness.toy_spec(1, 3)
    u0 = np.array([[30000, 30500, 31000]], float)
    toy = harness.ToyRings(spec, u0=u0, g=np.array([[0.3, 0.8, 1.2]]), u_init=u0[0] + 300, x0=3.0, noise=1.0)
    c = cls(init_codes=u0[0] + 300, **kw)
    c.reset(spec)
    toy.run(c, 3000)
    np.testing.assert_allclose(toy.x, 0.0, atol=0.06)
    assert np.all(c.telemetry()["fall_count"] == 0)


@pytest.mark.parametrize("cls", [ExtremumSeekingPO, ExtremumSeekingSine])
def test_g26_low_power_reaches_the_peak(cls):
    cfg, pp, spec, plant = harness.placed(delta_pm=60.0, power_delta_db=LOW_DB)
    c = cls()
    loop = harness.ClosedLoop(plant, spec, c)
    rec = loop.run(8000)
    d = rec["delta_pm"].astype(float)
    final = np.median(d[-1500:], axis=0)
    assert np.all(np.abs(final) < 15.0), final
    m = harness.lock_metrics(rec, slice(-3000, -1500), slice(-1500, None), delta_ref=np.zeros_like(final))
    report(f"{c.name} g26 at -10 dBm per ring (peak lock)", m, centre_detuning_pm=float(np.mean(final)))


def _g26_nominal_power(cls, **kw):
    cfg, pp, spec, plant = harness.placed(delta_pm=60.0)
    c = cls(**kw)
    loop = harness.ClosedLoop(plant, spec, c)
    rec = loop.run(8000)                          # 8 ms
    tel = c.telemetry()
    assert np.all(tel["state"] != E_FAULT)
    d = rec["delta_pm"].astype(float)
    m = harness.lock_metrics(rec, slice(0, 500), slice(2000, None), delta_ref=np.zeros(d.shape[1:]))
    report(f"{c.name}{kw or ''} g26 nominal power (peak seeking)", m, falls=float(tel["fall_count"].sum()),
           min_detuning_pm=float(d.min()))
    return d, tel


def test_g26_nominal_power_po_falls_at_the_fold():
    """At +4 dBm per ring the self-heating fold sits a few pm blue of the peak. A probe that reaches past it makes
    es_po fall off and recapture; it never ends in FAULT.

    The default +-16-code probe (~6 pm at the peak) reaches the fold only marginally. On the current plant (secant
    beta, heater_self_ratio 1.132) it does not fall in this 8 ms run, and it falls once on rings 4-7 in 0.4 s on the
    engine. With the tangent beta or with the ratio 1.158 (either change alone) it falls once on every ring here. So
    the fall path is exercised with a +-24-code probe (~9 pm), and the default parameters are checked for FAULT
    only."""
    d, tel = _g26_nominal_power(ExtremumSeekingPO, delta_codes=24.0)
    assert tel["fall_count"].sum() > 0, tel["fall_count"]
    assert d.min() < -100.0                       # a fall: the ring dropped onto its cold branch
    _g26_nominal_power(ExtremumSeekingPO)         # default parameters: asserts no FAULT


def test_g26_nominal_power_sine_approaches_without_a_fall():
    """es_sine's dither does not reach the fold: no fall, and the rings move toward the peak (blue, d decreasing)
    but are still well red of it after 8 ms (a slow approach, not a lock)."""
    d, tel = _g26_nominal_power(ExtremumSeekingSine)
    assert np.all(tel["fall_count"] == 0), tel["fall_count"]
    early, late = np.median(d[:500], axis=0), np.median(d[-500:], axis=0)
    assert np.all(late < early - 1.0), (early, late)
    assert np.all(late > 20.0), late              # not at the peak yet
    assert d.min() > 20.0


@pytest.mark.slow
def test_engine_sine_parks_at_the_drop_peak():
    """On the engine (g26 nominal, settled warm start, 0.2 s): es_sine drifts toward the drop peak at ~0.8 pm/ms
    without a fall and parks there (|d| < 3 pm), where the OMA penalty is ~13 dB."""
    import warnings
    from mrm import params
    from mrm.scenarios import library
    from mrm.sim import runner
    from mrm.sim.metrics import RecordSpec
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26")
        c = ExtremumSeekingSine()
        r = runner.run(cfg, c, library.static(), duration_s=0.2, uncertainty="nominal", variation="off", seeds=0,
                       heater_code="locked_steady_state", settle=True, reset_protocol="branch_high",
                       record=[RecordSpec("detuning_m", 10, "mean"), RecordSpec("reloma", 10, "mean"),
                               RecordSpec("oracle_reloma", 10, "mean")])
    d = np.asarray(r.trace["detuning_m@10:mean"]["value"])[:, 0, :] * 1e12      # 10 us samples
    assert np.all(c.telemetry()["fall_count"] == 0)
    drift = (np.median(d[3000:4000], axis=0) - np.median(d[2000:3000], axis=0)) / 10.0     # pm/ms at 20-40 ms
    assert np.all((drift < -0.5) & (drift > -1.2)), drift
    end = np.median(d[-1000:], axis=0)
    assert np.all(np.abs(end) < 3.0), end
    rel = np.asarray(r.trace["reloma@10:mean"]["value"])[-1000:, 0, :].mean(axis=0)
    orc = np.asarray(r.trace["oracle_reloma@10:mean"]["value"])[-1000:, 0, :].mean(axis=0)
    pen = 10.0 * np.log10(orc / rel)
    assert np.all((pen > 11.0) & (pen < 15.0)), pen


def test_registry():
    assert isinstance(registry.create("es_po", {"delta_codes": 8.0}), ExtremumSeekingPO)
    assert isinstance(registry.create("es_sine", {"dither_hz": 1000.0}), ExtremumSeekingSine)
    with pytest.raises(ValueError):
        ExtremumSeekingPO(mode="nope")
    with pytest.raises(ValueError):
        ExtremumSeekingSine(dither_hz=40e3).reset(harness.toy_spec(1, 1))
