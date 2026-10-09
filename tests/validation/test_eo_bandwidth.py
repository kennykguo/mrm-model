"""EO bandwidth of the waveform mode against CAL-10 (calibration) and VAL-3a/VAL-3b (held out).

Replica ``eo_bandwidth_at_bias`` (mrm/validation/targets/README.md): the small-signal EO response of the G25 test
vehicle (preset g25_test_vehicle, nominal world) through the driver output resistance, the junction R_s and C_j(V)
(mrm.waveform.driver) and the exact TCMT linearization (mrm.waveform.tcmt_time; Sacher & Poon 2008 poles at
-kappa/2 +- i Delta, zero at -(kappa_i + kappa_d)); 3 dB point |H(f)/H(0)|^2 = 1/2 (docs/research/03 section 5.6).

The measurement detuning is not printed (CAL-10/VAL-3 ``detuning: hypothesis``). Hypotheses, chosen before any
comparison from 03 section 5.6 (which states the CAL-10 inversion at both): the laser at the maximum static slope
x = 2 Delta/kappa = 1/sqrt(3) (PRIMARY: the usual small-signal bias point) or at x = 0.813 (the -4 dB through point);
and at 2 V either re-centred to the same x (PRIMARY: the laser re-tuned at each bias) or held at the -1 V wavelength
(the resonance then moves by the junction shift). R_s C is fitted to CAL-10 only (calibration rule: R_s scaled, C_j(V)
and R_drv as in the preset; only the product is identified); VAL-3 is then predicted, not tuned. The scored
assertions use the primary hypotheses; every hypothesis is printed (pytest -s) and in the handoff. Validation tests
report outcomes and do not block CI (07 section 4.1).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq

from mrm.validation.targets_io import evaluate, load_targets
from mrm.waveform import DriverJunction, RingTCMT, eo_bandwidth

X_PRIMARY = 1.0 / math.sqrt(3.0)
X_HYPOTHESES = (X_PRIMARY, 0.813)


@pytest.fixture(scope="module")
def pp():
    from mrm import params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # artefact fallbacks warn until the extraction lands
        cfg = params.load("g25_test_vehicle")
        return params.realize(cfg, 1, seeds=0, uncertainty="nominal", variation="off")


@pytest.fixture(scope="module")
def targets():
    cal = {t.id: t for t in load_targets("calibration")}
    val = {t.id: t for t in load_targets("validation")}
    return cal["CAL-10"], val["VAL-3a"], val["VAL-3b"]


def _f3db(pp, v_bias, x_bias=None, t_ring_k=None, r_s_scale=1.0):
    ring = RingTCMT.from_plant_params(pp, 0, 0, v_bias=v_bias, x_bias=x_bias, t_ring_k=t_ring_k)
    dj = DriverJunction.from_plant_params(pp, 0, 0, rise_time_s=0.0, r_s_scale=r_s_scale)
    r = eo_bandwidth(ring, dj, v_bias)
    return r, ring, dj


@pytest.fixture(scope="module")
def fits(pp, targets):
    """R_s scale fitted to CAL-10 (30.9 GHz at 1 V reverse) under each detuning hypothesis, and the predictions."""
    cal10 = targets[0]
    out = {}
    for x in X_HYPOTHESES:
        pre, ring1, _ = _f3db(pp, 1.0, x_bias=x)
        scale = brentq(lambda s: _f3db(pp, 1.0, x_bias=x, r_s_scale=s)[0]["f3db_hz"] - cal10.value * 1e9, 0.3, 30.0,
                       xtol=1e-7)
        b1, _, dj = _f3db(pp, 1.0, x_bias=x, r_s_scale=scale)
        b2c, _, _ = _f3db(pp, 2.0, x_bias=x, r_s_scale=scale)
        b2f, _, _ = _f3db(pp, 2.0, t_ring_k=ring1.meta["t_ring_k"], r_s_scale=scale)
        out[x] = {"preset_f3db_ghz": pre["f3db_hz"] / 1e9, "r_s_scale": scale,
                  "r_s_ohm": float(dj.r_s(1.0)), "rc_pole_1v_ghz": b1["rc_pole_hz"] / 1e9,
                  "f3db_1v_ghz": b1["f3db_hz"] / 1e9,
                  "recentred": {"f3db_2v_ghz": b2c["f3db_hz"] / 1e9, "peaking_db": b2c["peaking_db"],
                                "ratio": b2c["f3db_hz"] / b1["f3db_hz"], "x_2v": b2c["x_bias"]},
                  "fixed_laser": {"f3db_2v_ghz": b2f["f3db_hz"] / 1e9, "peaking_db": b2f["peaking_db"],
                                  "ratio": b2f["f3db_hz"] / b1["f3db_hz"], "x_2v": b2f["x_bias"]}}
    for x, d in out.items():
        print(f"\nx_bias {x:.3f}: preset {d['preset_f3db_ghz']:.2f} GHz at -1 V; CAL-10 fit R_s x{d['r_s_scale']:.3f} "
              f"= {d['r_s_ohm']:.0f} ohm (RC pole {d['rc_pole_1v_ghz']:.1f} GHz)")
        for k in ("recentred", "fixed_laser"):
            e = d[k]
            print(f"   2 V {k:11s}: x = {e['x_2v']:.3f}, f3dB {e['f3db_2v_ghz']:.2f} GHz (peaking {e['peaking_db']:.2f}"
                  f" dB), ratio 2V/1V {e['ratio']:.3f}")
    return out


def test_cal10_preset_prediction_is_reported(fits, targets):
    """The uncalibrated preset (C_j 15 fF at 0 V, R_s 100 ohm, R_drv 50 ohm: assumed) predicts a bandwidth far
    above CAL-10, so R_s C must be calibrated (CAL-10 identifies it). Recorded, not scored."""
    ev = evaluate(targets[0], fits[X_PRIMARY]["preset_f3db_ghz"])
    print(f"CAL-10 with the preset R_s C: {fits[X_PRIMARY]['preset_f3db_ghz']:.2f} GHz -> {ev.level}")
    assert math.isfinite(fits[X_PRIMARY]["preset_f3db_ghz"])
    assert fits[X_PRIMARY]["preset_f3db_ghz"] > targets[0].value


def test_cal10_rc_fit(fits, targets):
    """The fit reproduces CAL-10 (by construction); the implied total R (R_drv + R_s) and RC pole are recorded.
    Primary hypothesis: R_s stays inside the preset's range (15-390 ohm)."""
    for x, d in fits.items():
        assert d["f3db_1v_ghz"] == pytest.approx(targets[0].value, abs=1e-3)
        assert evaluate(targets[0], d["f3db_1v_ghz"]).passed
    assert 15.0 <= fits[X_PRIMARY]["r_s_ohm"] <= 390.0


def test_val3a_bandwidth_at_2v(fits, targets):
    """VAL-3a: 35.4 +- 3 GHz at 2 V reverse (primary hypotheses: x = 1/sqrt(3), laser re-centred)."""
    pred = fits[X_PRIMARY]["recentred"]["f3db_2v_ghz"]
    ev = evaluate(targets[1], pred)
    print(f"VAL-3a primary prediction {pred:.2f} GHz -> {ev.level} ({ev.detail})")
    assert ev.passed


def test_val3b_bandwidth_ratio(fits, targets):
    """VAL-3b: ratio 2 V / 1 V = 1.09 +- 0.06 (primary hypotheses). A pure abrupt-junction RC would give ~1.24;
    the photon-lifetime limit pulls it down."""
    pred = fits[X_PRIMARY]["recentred"]["ratio"]
    ev = evaluate(targets[2], pred)
    print(f"VAL-3b primary prediction {pred:.3f} -> {ev.level} ({ev.detail})")
    assert ev.passed


def test_all_hypotheses_are_reported(fits, targets):
    """Every hypothesis is evaluated and printed (no assertion on the non-primary ones); the fixed-laser reading is
    recorded as excluded by VAL-3b when it fails (the bias moves the resonance by ~9 GHz/V at 1-2 V)."""
    rows = []
    for x, d in fits.items():
        for k in ("recentred", "fixed_laser"):
            a = evaluate(targets[1], d[k]["f3db_2v_ghz"])
            b = evaluate(targets[2], d[k]["ratio"])
            rows.append((x, k, d[k]["f3db_2v_ghz"], a.level, d[k]["ratio"], b.level))
    for r in rows:
        print(f"x={r[0]:.3f} {r[1]:11s}: VAL-3a {r[2]:.2f} GHz {r[3]:5s} | VAL-3b {r[4]:.3f} {r[5]}")
    assert all(np.isfinite([r[2] for r in rows])) and all(np.isfinite([r[4] for r in rows]))
