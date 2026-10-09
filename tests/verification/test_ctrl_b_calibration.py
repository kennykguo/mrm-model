"""mrm.control.calibration: the knowledge gate on spec.nominal (07 P1-1), the nominal heater model, the line labels
from a sweep pattern (cold start, B5), and the simulated factory calibration (07 P1-1 factory_cal) on a test
station (the reference bench, or the engine's Plant when available).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mrm.control import calibration as cal
from mrm.control.calibration import (FactoryProcedure, code_of_w, factory_cal, heater_current_a, heater_power_w,
                                     label_bank_by_pattern, label_lines_by_pattern, nominal_design,
                                     predicted_pattern, w_of_code)
from mrm.plant import electrical as el

from test_ctrl_b_plant import RefBenchFactory, load_g26, make_spec, realize

ALL = ["line_grid", "ring_to_line", "heater_nominal", "optics_nominal", "dlambda_dt_nominal", "monitor_nominal"]


def test_knowledge_gate():
    """07 P1-1: nominal values reach a controller only with their token; g26's own knowledge (line_grid,
    ring_to_line) releases none of the device numbers."""
    spec = make_spec(load_g26(), 1)
    d = nominal_design(spec)
    assert d.heater_law is None and d.q_loaded is None and d.s0_m_per_k is None and d.fwhm_w is None
    assert d.lambda0_m is not None                     # the line grid is known
    spec = make_spec(load_g26(knowledge=["heater_r_nominal"]), 1)   # 07 P1-1 name: the heater resistance only
    d = nominal_design(spec)
    assert d.r_cold_ohm == pytest.approx(45.41) and d.heater_law is None and d.q_loaded is None
    # the controllers-A vocabulary (mrm.control.baselines.pi_lock.KNOWLEDGE_KEYS) releases the same fields
    spec = make_spec(load_g26(knowledge=["line_grid", "heater_law", "heater_efficiency", "linewidth"]), 1)
    d = nominal_design(spec)
    assert d.heater_law == "voltage" and d.p_fsr_w == pytest.approx(0.02545) and d.fsr_hz == pytest.approx(1.8e12)
    assert d.q_loaded == pytest.approx(3850.0) and d.offset_m is None and d.supply_current_max_a is None
    assert d.fwhm_w is not None
    spec = make_spec(load_g26(knowledge=ALL), 1)
    d = nominal_design(spec)
    assert d.fsr_hz == pytest.approx(1.8e12) and d.q_loaded == pytest.approx(3850.0)
    assert d.fwhm_m == pytest.approx(d.lambda0_m / 3850.0)
    assert d.supply_current_max_a == pytest.approx(0.15)
    # spec.nominal = None: nothing but the DAC facts
    from mrm.control.api import ControllerSpec
    bare = ControllerSpec.from_config(load_g26(knowledge=ALL), n_batch=1, nominal=None)
    assert nominal_design(bare).heater_law is None


def test_drive_coordinate_round_trip():
    for q in (1, 2):
        c = np.array([0, 1, 1000, 32768, 65535])
        np.testing.assert_array_equal(code_of_w(w_of_code(c, 65535, q), 65535, q), c)
    assert cal.drive_exponent("voltage") == 2 and cal.drive_exponent("dac_pwm") == 1


def test_nominal_heater_model_against_the_plant_driver():
    """The controller's nominal heater law (heater_nominal knowledge) against the plant's driver law at the nominal
    world, with the heater-resistor temperature from the same droop model: within 3 % (TCR droop to first order)."""
    cfg = load_g26(knowledge=ALL)
    spec = make_spec(cfg, 1)
    d = nominal_design(spec)
    pp = realize(cfg, 1, nominal=True)
    hp = el.HeaterDriverParams.from_plant_params(pp)
    codes = np.array([10000, 20000, 30000, 40000, 50000])
    p_ctrl = heater_power_w(codes, d)
    i_ctrl = heater_current_a(codes, d)
    # plant: V = V_fs code / code_max into R(T) = R0 (1 + TCR dT_h) with dT_h from the controller's own estimate
    v = hp.v_full_scale_v[0, 0] * codes / 65535.0
    r = hp.r_cold_ohm[0, 0] * (1.0 + d.y_per_w * p_ctrl)
    np.testing.assert_allclose(p_ctrl, v ** 2 / r, rtol=1e-9)
    np.testing.assert_allclose(i_ctrl, v / r, rtol=1e-9)
    # and the y used is the printed G25 droop fit (0.00904 /mW, CAL-4) within 15 %
    assert 0.85 * 9.04 < d.y_per_w < 1.15 * 9.04


def test_pattern_labels_g26_grid():
    """Synthetic sweeps: one FSR plus one crossing identifies every line (the empty gap between lines 4 and 3), for
    FSRs across window B, with a curved drive map and position noise; one crossing short of a full FSR is flagged
    as ambiguous rather than guessed."""
    f = 228.749e12 + (np.arange(8) - 3.5) * 400e9
    rng = np.random.default_rng(1)
    for fsr in (1.75e12, 1.8e12, 1.86e12):
        for first in (0, 3, 6):
            v, lines, _ = predicted_pattern(f, fsr, first, 9)
            w_lin = 0.05 + 0.5 * v
            w = w_lin / (1.0 - 0.2 * w_lin) + rng.normal(0.0, 0.002, v.size)   # g_c inverse with c = 0.2
            out = label_lines_by_pattern(w, f, fsr_range_hz=(1.53e12, 2.07e12))
            assert out["ok"], out
            np.testing.assert_array_equal(out["lines"], lines)
    v, lines, _ = predicted_pattern(f, 1.8e12, 2, 7)
    short = label_lines_by_pattern(0.05 + 0.5 * v, f)
    assert not short["ok"]


def test_pattern_labels_joint_bank():
    f = 228.749e12 + (np.arange(8) - 3.5) * 400e9
    rng = np.random.default_rng(2)
    evs, want = [], []
    for n in range(8):
        v, lines, _ = predicted_pattern(f, 1.79e12, (n + 3) % 8, 9)
        evs.append(0.04 + 0.52 * v + rng.normal(0.0, 0.003, v.size))
        want.append(lines)
    fits = label_bank_by_pattern(evs, f)
    for n in range(8):
        assert fits[n]["ok"], fits[n]
        np.testing.assert_array_equal(fits[n]["lines"], want[n])
    assert fits[0]["fsr_hz"] == pytest.approx(1.79e12, rel=0.01)


def test_factory_procedure_plan_and_scenario():
    cfg = load_g26(knowledge=ALL)
    spec = make_spec(cfg, 1)
    proc = FactoryProcedure()
    pl = proc.plan(spec)
    assert [len(g) for g in pl["groups"]] == [1] * 8                      # one ring at a time (default)
    assert [g[0] for g in pl["groups"]] == list(spec.bus_order)
    assert len(pl["sets"]) == 4 and pl["sets"][1].tolist() == [False, True] * 4
    # every all-lines sweep before any line-coding sweep
    assert [si for (_, _, _, si) in pl["schedule"][:8]] == [0] * 8
    # sized by the supply: 4 rings per group under the nominal 0.15 A at w_max = 0.85 (8 would break it)
    assert [len(g) for g in FactoryProcedure(group_size=None).plan(spec)["groups"]] == [4, 4]
    sc = proc.scenario(spec)
    arr = sc.compile(dt_s=spec.tick_s, n_rings=8, n_lines=8, batch=1, aggressor_sites=())
    cache = arr.new_cache()
    per = pl["period_ticks"]
    for (a, b, gi, si) in pl["schedule"][:4]:
        mid = (a + b) // 2 * per
        np.testing.assert_allclose(arr.evaluate(mid, cache)["laser.line_enable"][0], pl["sets"][si].astype(float))
    assert arr.evaluate(0, cache)["thermal.t_base_k"][0, 0] == pytest.approx(298.15)
    # without the supply knowledge one ring at a time
    assert all(len(g) == 1 for g in FactoryProcedure(group_size=None).plan(make_spec(load_g26(), 1))["groups"])


@pytest.mark.slow
def test_factory_cal_on_the_reference_bench():
    """The full simulated factory sweep on the reference plant (nominal g26): every ring's resonances labelled, and
    the labels agree with Truth (the line each resonance's drive puts the ring on, checked by a static run)."""
    cfg = load_g26(knowledge=ALL)
    pp = realize(cfg, 1, nominal=True)
    spec = make_spec(cfg, 1)
    tab = factory_cal(RefBenchFactory(pp, cfg, spec), FactoryProcedure())
    assert np.all(tab.ok[0]), (tab.ok, tab.notes)
    assert 24.0 < tab.t_cal_c[0] < 40.0                 # the die sensor during the sweeps (station at 25 degC)
    # the expected g26 nominal pattern: ring n meets line n + 3 first (or the next order), then the cyclic order
    # L0, L4, (empty gap), L3, L7, L2, L6, L1, L5
    cyc = [0, 4, 3, 7, 2, 6, 1, 5]
    for n in range(8):
        w, ln = tab.events(0, n)
        assert w.size >= 9
        k = cyc.index(int(ln[0]))
        np.testing.assert_array_equal(ln[:8], [cyc[(k + j) % 8] for j in range(8)])
    print("\nfactory table (nominal): " + "; ".join(f"ring {n}: {tab.events(0, n)[1][:9].tolist()}"
                                                     for n in range(8)))


# ---------------------------------------------------------------------------------------------------------------
# review additions: priors that cover the uncertainty set, the drive-law hypothesis, plain-mapping units
# ---------------------------------------------------------------------------------------------------------------

G26_GRID = 228.749e12 + (np.arange(8) - 3.5) * 400e9


def test_fsr_prior_covers_every_fsr_hypothesis():
    """07 4.8: the FSR is a hypothesis (window B 1.8 THz default, window A 1.47 THz, stress 1.53 / 1.60 / 1.72 /
    1.88). The pattern fit's window must contain all of them; the earlier nominal +- 15 % (1.53-2.07 THz) excluded
    window A, and a window-A world was then labelled wrongly AND confidently (closed loop, sampled g26 world)."""
    d = nominal_design(make_spec(load_g26(knowledge=ALL), 1))
    assert d.fsr_range_hz == pytest.approx((1.4e12, 1.9e12))
    lo, hi = cal.fsr_prior_hz(d)
    for fsr in (1.47e12, 1.53e12, 1.60e12, 1.72e12, 1.8e12, 1.88e12):
        assert lo < fsr < hi
    assert cal.fsr_prior_hz(nominal_design(make_spec(load_g26(), 1))) == (1.3e12, 2.1e12)   # no optics knowledge


def test_pattern_labels_window_a():
    """A window-A map (FSR 1.47 THz) is labelled right inside the knowledge-gated prior and never confidently wrong.
    The former prior (nominal +- 15 %, 1.53-2.07 THz) cannot hold the true FSR, and the curvature then aliases a
    window-A pattern onto a window-B one: wrong labels reported as confident (the defect this test pins)."""
    d = nominal_design(make_spec(load_g26(knowledge=ALL), 1))
    rng = np.random.default_rng(5)
    n_ok = old_wrong = 0
    for first in range(8):
        v, lines, _ = predicted_pattern(G26_GRID, 1.47e12, first, 10)
        w = 0.05 + 0.55 * v + rng.normal(0.0, 0.002, v.size)
        out = label_lines_by_pattern(w, G26_GRID, fsr_range_hz=cal.fsr_prior_hz(d))
        assert not out["ok"] or np.array_equal(out["lines"], lines), (first, out)
        n_ok += bool(out["ok"])
        old = label_lines_by_pattern(w, G26_GRID, fsr_range_hz=(1.53e12, 2.07e12), amb_min=1.5,
                                     require_interior=False)          # the former prior and acceptance rule
        old_wrong += bool(old["ok"] and not np.array_equal(old["lines"], lines))
    assert n_ok >= 6, n_ok
    assert old_wrong >= 1                     # the former prior's failure mode, for the record


def test_pattern_labels_drive_law_mismatch():
    """07 P1-11: the heater law is a hypothesis. A power-linear driver (power, dac_pwm) read in the q = 2 coordinate
    gives P ~ w^(1/2), which the one-parameter curvature cannot represent; with law_exponents the fit labels such
    maps right, and never confidently wrong over the FSR hypotheses (synthetic maps, 2 mdrive noise)."""
    d = nominal_design(make_spec(load_g26(knowledge=ALL), 1))
    prior = cal.fsr_prior_hz(d)
    ex = cal.law_exponents(d, 2)
    assert ex == (1.0, 0.5) and cal.law_exponents(d, 1) == (1.0, 2.0)
    rng = np.random.default_rng(1)
    right = wrong = 0
    for fsr in (1.47e12, 1.72e12, 1.8e12, 1.88e12):
        for first in range(0, 8, 2):
            v, lines, _ = predicted_pattern(G26_GRID, fsr, first, 10)
            p = 0.05 + 0.5 * v                                    # heater power in units of the full scale
            w = p ** 2 / 0.6 + rng.normal(0.0, 0.002, v.size)     # the q = 2 coordinate of a power-linear driver
            out = label_lines_by_pattern(w, G26_GRID, fsr_range_hz=prior, exponents=ex)
            right += bool(out["ok"] and np.array_equal(out["lines"], lines))
            wrong += bool(out["ok"] and not np.array_equal(out["lines"], lines))
            assert out["exponent"] in ex
    assert wrong == 0 and right >= 12, (right, wrong)


def test_nominal_design_from_a_plain_mapping_is_si():
    """spec.nominal may be a plain mapping in config units (mrm.control.api.nominal_line_grid_hz accepts one): the
    nominal design must still be SI (the unit suffix converts, ARCHITECTURE section 1)."""
    import dataclasses
    spec = make_spec(load_g26(knowledge=ALL), 1)
    ref = nominal_design(spec)
    plain = {"ring.heater.driver_mode": "voltage", "ring.heater.full_scale_v": {"value": 2.5},
             "ring.heater.r_cold_ohm": 45.41, "ring.heater.p_fsr_mw": 25.45, "ring.optics.fsr_thz": {"value": 1.8,
             "range": [1.4, 1.9]}, "ring.optics.q_loaded": 3850, "ring.optics.resonance_offset_nm": 7.6,
             "ring.optics.resonance_offset_ref_temp_c": 25.0}
    d = nominal_design(dataclasses.replace(spec, nominal=plain))
    assert d.p_fsr_w == pytest.approx(0.02545) and d.fsr_hz == pytest.approx(1.8e12)
    assert d.offset_m == pytest.approx(7.6e-9) and d.offset_ref_temp_k == pytest.approx(298.15)
    assert d.fsr_range_hz == pytest.approx((1.4e12, 1.9e12)) and d.fwhm_m == pytest.approx(ref.fwhm_m)


def test_public_names_resolve():
    for name in cal.__all__:
        assert hasattr(cal, name), name
