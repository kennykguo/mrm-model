"""realize(): worlds vs devices, counter-RNG keying, geometry Jacobians, shapes, timing and lag rules
(07 P0-4, P1-13, P1-16, P0-6, P0-1 item 2)."""
import copy
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from mrm.params.realize import SELFHEAT_LAG_GAIN_MAX
from mrm.params.config import load, yaml_load
from mrm.params.realize import PlantParams, RealizeError, realize

MINI = Path(__file__).resolve().parents[1] / "data" / "mini.yaml"
ROOT = Path(__file__).resolve().parents[2]


def mini():
    return copy.deepcopy(yaml_load(MINI.read_text()))


@pytest.fixture(scope="module")
def cfg():
    return load(MINI)


def _arrays(p: PlantParams):
    """Every batched numeric array of a PlantParams, by name."""
    out = {}
    for f in fields(p):
        g = getattr(p, f.name)
        if f.name == "layout" or not hasattr(g, "__dataclass_fields__"):
            continue
        for gf in fields(g):
            v = getattr(g, gf.name)
            if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[0] == p.batch:
                out[f"{f.name}.{gf.name}"] = v
    return out


def test_shapes_and_readonly(cfg):
    B, N, L = 5, cfg.n_rings, cfg.n_lines
    p = realize(cfg, B, 0, thermal_hook=None)
    o = p.optics
    assert o.q_loaded.shape == (B, N) and o.t_ref_k.shape == (B, 1) and p.laser.power_w.shape == (B, L)
    assert o.order.dtype == np.int64 and p.timing.monitor_latency_ticks.dtype == np.int64
    assert p.heater.driver_mode.shape == (B, 1) and p.heater.dac_bit_err_z.shape == (B, N, 16)
    assert p.temp_sensors.offset_k.shape == (B, 1)
    for name, a in _arrays(p).items():
        assert a.dtype in (np.float64, np.int64, np.complex128), name
        assert not a.flags.writeable, name
        assert a.shape[1] in (1, N, L, len(p.temp_sensors.names)), name
    assert p.flags.as_array().dtype == np.int32
    assert p.layout.ring_xy_m.shape == (N, 2)


def test_relations_close_after_realize(cfg):
    for mode in ("nominal", "sample", "lhs"):
        p = realize(cfg, 32, 3, uncertainty=mode, thermal_hook=None)
        assert all(v <= 1e-12 for v in p.diagnostics["relation_residuals"].values()), mode


def test_element_b_equals_single_run(cfg):
    big = realize(cfg, 12, 100, thermal_hook=None)
    for b in (0, 5, 11):
        one = realize(cfg, 1, 100 + b, thermal_hook=None)
        sel = big.select(b)
        a1, a2 = _arrays(sel), _arrays(one)
        assert a1.keys() == a2.keys()
        for k in a1:
            np.testing.assert_array_equal(a1[k], a2[k], err_msg=k)
        for k in big.world:
            np.testing.assert_array_equal(big.world[k][b:b + 1], one.world[k], err_msg=k)


def test_worlds_and_devices_use_separate_streams(cfg):
    a = realize(cfg, 8, {"device": 1, "uncertainty": 7}, thermal_hook=None)
    b = realize(cfg, 8, {"device": 2, "uncertainty": 7}, thermal_hook=None)
    c = realize(cfg, 8, {"device": 1, "uncertainty": 8}, thermal_hook=None)
    for k in a.world:
        np.testing.assert_array_equal(a.world[k], b.world[k])          # same worlds, other devices
    assert not np.array_equal(a.heater.r_cold_ohm, b.heater.r_cold_ohm)
    np.testing.assert_array_equal(a.heater.r_cold_ohm, c.heater.r_cold_ohm)   # same devices, other worlds
    assert not np.array_equal(a.world["q_loaded"], c.world["q_loaded"])


def test_uncertainty_is_per_world_variation_per_ring(cfg):
    p = realize(cfg, 16, 0, thermal_hook=None)
    q = p.optics.q_loaded                         # die-level epistemic draw: equal across rings
    assert np.all(q == q[:, :1]) and np.unique(q[:, 0]).size == 16
    r = p.heater.r_cold_ohm                       # per-device variation: differs ring to ring
    assert np.all(np.std(r, axis=1) > 0)
    np.testing.assert_allclose(p.world["q_loaded"], q[:, 0])
    assert 3500 <= q.min() and q.max() <= 4200


def test_nominal_mode_and_variation_off(cfg):
    p = realize(cfg, 3, 9, uncertainty="nominal", variation="off", thermal_hook=None)
    assert np.all(p.optics.q_loaded == 3800)
    assert np.all(p.heater.r_cold_ohm == 45.4)
    assert np.all(p.world["thermo_optic"] == 0)
    np.testing.assert_allclose(p.thermal.dlambda_dt_s0_m_per_k, 67.5e-12, rtol=1e-15)
    assert np.all(p.timing.latency_scale == 1.0)


def test_hypothesis_frequencies_follow_weights(cfg):
    p = realize(cfg, 4000, 0, variation="off", thermal_hook=None)
    idx = p.world["thermo_optic"]
    frac = np.bincount(idx, minlength=3) / idx.size
    np.testing.assert_allclose(frac, [0.5, 0.25, 0.25], atol=0.03)
    s0 = np.round(p.thermal.dlambda_dt_s0_m_per_k[:, 0] * 1e12, 9)
    beta = p.thermal.dlambda_dt_beta_per_k[:, 0]
    assert np.all(s0[idx == 1] == 86.0) and np.all(beta[idx == 1] == 0.0)     # H86 sets both leaves
    assert np.all(s0[idx == 2] == 62.5) and np.all(beta[idx == 2] == 1.4e-3)
    drv = p.heater.driver_mode[:, 0]
    assert set(np.unique(drv)) == {0, 1}


def test_pinned_and_lhs(cfg):
    pinned = load(MINI, alts={"uncertainty.thermo_optic": "H86", "uncertainty.q_loaded": "low_q"})
    p = realize(pinned, 50, 0, thermal_hook=None)
    np.testing.assert_allclose(p.thermal.dlambda_dt_s0_m_per_k, 86e-12, rtol=1e-15)
    assert np.all(p.optics.q_loaded == 2600)
    lhs = realize(cfg, 40, 0, uncertainty="lhs", variation="off", thermal_hook=None)
    u = (lhs.world["q_loaded"] - 3500) / 700
    assert np.array_equal(np.sort(np.floor(u * 40)), np.arange(40))     # one point per stratum


def test_grid(cfg):
    p = realize(cfg, 3, 0, grid={"ring.optics.q_loaded": [3600, 3800, 4000]}, variation="off", thermal_hook=None)
    np.testing.assert_array_equal(p.optics.q_loaded[:, 0], [3600, 3800, 4000])
    with pytest.raises(RealizeError, match="3 values"):
        realize(cfg, 3, 0, grid={"ring.optics.q_loaded": [1, 2]}, thermal_hook=None)


def test_latency_scale_quantized(cfg):
    p = realize(cfg, 64, 0, variation="off", thermal_hook=None)
    s = p.timing.latency_scale[:, 0]
    np.testing.assert_array_equal(p.timing.monitor_latency_ticks[:, 0], np.rint(2 * s).astype(np.int64))
    np.testing.assert_array_equal(p.temp_sensors.latency_ticks[:, 0], np.rint(1000 * s).astype(np.int64))
    assert p.timing.adc_period_ticks == 4 and p.timing.integration_window_ticks == 2
    assert p.timing.controller_period_ticks == 10


def test_variation_truncated_to_range():
    d = mini()
    d["ring"]["monitor"]["responsivity_a_per_w"]["range"] = [0.94, 0.96]
    d["variation"]["ring.monitor.responsivity_a_per_w"] = {"level": "ring", "ring_sigma": 0.05,
                                                            "provenance": "assumed", "generation": "none",
                                                            "source": "TEST"}
    p = realize(load(d), 200, 0, thermal_hook=None)
    r = p.monitor.responsivity_a_per_w
    assert r.min() >= 0.94 and r.max() <= 0.96 and r.std() > 0.004


def test_geometry_jacobian_correlation():
    """07 P1-16 test: the sampled correlation of (offset, K_d^2) matches the Jacobian prediction."""
    d = mini()
    d["variation"] = {"geometry": {"width": {"die_sigma_nm": 2.0, "ring_sigma_nm": 1.0},
                                   "gap_drop": {"ring_sigma_nm": 3.0},
                                   "jacobians": ["ring.geometry.jacobian"],
                                   "provenance": "assumed", "generation": "none", "source": "TEST"}}
    j1, j2, j3 = 1.5, 0.01, -0.02          # d lambda_res/dw (nm/nm), d ln K_d^2/dw, d ln K_d^2/d gap (per nm)
    d["ring"]["geometry"]["jacobian"]["value"] = {
        "d": ["width_nm", "gap_drop_nm"],
        "of": {"lambda_res_nm_per_nm": [j1, 0.0], "ln_kappa2_drop_per_nm": [j2, j3]}}
    cfg = load(d)
    p = realize(cfg, 3000, 0, uncertainty="nominal", thermal_hook=None)
    off = p.optics.resonance_offset_m.ravel() * 1e9
    ln_qd = np.log(p.optics.q_drop.ravel())
    var_w = 2.0 ** 2 + 1.0 ** 2
    pred = (j1 * j2 * var_w) / np.sqrt(j1 ** 2 * var_w * (j2 ** 2 * var_w + j3 ** 2 * 3.0 ** 2))
    got = np.corrcoef(off, ln_qd)[0, 1]
    assert got == pytest.approx(pred, abs=0.03)
    # offset moves by -j1 per nm of width (offset = line - resonance) and its spread is j1*sqrt(var_w)
    assert np.std(off) == pytest.approx(j1 * np.sqrt(var_w), rel=0.05)
    assert np.mean(off) == pytest.approx(7.6, abs=0.1)
    g = p.diagnostics["geometry_draws_nm"]["width"]
    np.testing.assert_allclose(off.reshape(3000, -1), 7.6 - j1 * g, rtol=0, atol=1e-9)


def test_geometry_jacobian_cannot_move_a_free_member():
    d = mini()
    d["ring"]["geometry"]["jacobian"]["value"]["of"]["ln_kappa2_through_per_nm"] = [0.01, 0.0]
    with pytest.raises(RealizeError, match="q_through"):
        realize(load(d), 4, 0, thermal_hook=None)


def test_timing_rules():
    d = mini()
    d["ring"]["monitor"]["integration_window_s"]["value"] = 5e-6       # longer than the 4 us ADC period
    with pytest.raises(RealizeError, match="integration_window"):
        realize(load(d), 1, 0, thermal_hook=None)


def test_selfheat_lag_bound():
    d = mini()
    th = d["ring"]["thermal"]
    th["xt_tau_min_s"]["value"] = 2e-6
    th["xt_tau_ref_s"]["value"] = 2e-6
    th["xt_nearest_frac"]["value"] = 0.1
    th["xt_tau_far_s"]["value"] = 2e-6
    with pytest.raises(RealizeError, match="07 P0-1"):
        realize(load(d), 1, 0, uncertainty="nominal", variation="off", thermal_hook=None)
    d["fidelity"]["self_heating"] = "off"
    realize(load(d), 1, 0, uncertainty="nominal", variation="off", thermal_hook=None)


def test_thermal_inputs_without_hook(cfg):
    p = realize(cfg, 4, 0, thermal_hook=None)
    tin = p.thermal_inputs
    assert p.thermal.system is None and tin.model == "interim"
    assert tin.interim["tau_fast_s"].shape == (4, 1) and tin.interim["xt_dist_ref_m"][0, 0] == pytest.approx(60e-6)
    assert set(tin.scales) == {"time_scale_fast", "time_scale_slow"}
    assert tin.scaling_hypothesis == ("resistance",) * 4
    np.testing.assert_array_equal(tin.z_heater_ring_k_per_w, p.thermal.z_heater_ring_k_per_w)


def test_thermal_hook_auto(cfg):
    thermal = pytest.importorskip("mrm.plant.thermal")
    if not hasattr(thermal, "assemble"):
        pytest.skip("mrm.plant.thermal.assemble not available")
    p = realize(cfg, 4, 0)
    assert p.thermal.system is not None and p.diagnostics["thermal_hook"] == "mrm.plant.thermal"
    assert p.diagnostics["selfheat_lag_gain"] < SELFHEAT_LAG_GAIN_MAX


def test_numba_disabled_parity(cfg):
    code = ("import numpy as np; from mrm.params import load, realize;"
            f"p = realize(load(r'{MINI}'), 6, 4, thermal_hook=None);"
            "print(repr(np.concatenate([p.optics.q_loaded.ravel(), p.heater.r_cold_ohm.ravel(),"
            " p.optics.resonance_offset_m.ravel()]).tolist()))")
    env = dict(os.environ, NUMBA_DISABLE_JIT="1")
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    p = realize(cfg, 6, 4, thermal_hook=None)
    ref = np.concatenate([p.optics.q_loaded.ravel(), p.heater.r_cold_ohm.ravel(), p.optics.resonance_offset_m.ravel()])
    np.testing.assert_array_equal(np.array(eval(out.stdout.strip())), ref)


def test_plant_does_not_import_config():
    """07 P0-4: an AST test that nothing under mrm/plant imports mrm.params.config."""
    import ast
    for path in (ROOT / "mrm" / "plant").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("mrm.params.config"), path
                if node.module == "mrm.params":
                    assert "config" not in [a.name for a in node.names], path
            if isinstance(node, ast.Import):
                assert all(not a.name.startswith("mrm.params.config") for a in node.names), path


def test_alt_selection_survives_sampling():
    cfg = load(MINI, alts={"ring.optics.q_loaded": "talk"})
    p = realize(cfg, 6, 0, thermal_hook=None)
    assert np.all(p.optics.q_loaded == 3500)


def test_backscatter_per_order():
    """07 P2-2: an independent complex beta per order with E|beta|^2 = (splitting/2)^2; rb = 2 pi beta/FSR."""
    cfg = load(MINI, overrides={"fidelity.backscatter": "on", "ring.optics.backscatter_splitting_ghz": 4.0})
    p = realize(cfg, 500, 0, thermal_hook=None)
    rb = p.optics.backscatter_rb
    assert rb.shape == (500, 4, 5) and rb.dtype == np.complex128
    beta = rb * p.optics.fsr_hz[:, :, None] / (2 * np.pi)
    assert np.mean(np.abs(beta) ** 2) == pytest.approx(4e18, rel=0.1)
    assert abs(np.mean(beta)) < 0.1 * 2e9
    off = realize(load(MINI), 2, 0, thermal_hook=None)
    assert not np.any(off.optics.backscatter_rb)


# ------------------------------------------------------------------------------------------------ review additions

def test_hypothesis_weights_with_numeric_choice_names():
    """Choices are strings internally; YAML weight keys written as numbers ({160: 0.7}) must still apply."""
    d = mini()
    d["ring"]["optics"]["q_loaded"]["alternatives"] = {"3500": {"value": 3500, "provenance": "printed"},
                                                       "4000": {"value": 4000, "provenance": "assumed"}}
    del d["uncertainty"]["q_loaded"]
    d["uncertainty"]["q_hyp"] = {"kind": "hypothesis", "target": "ring.optics.q_loaded",
                                 "choices": ["default", 3500, 4000], "nominal": "default",
                                 "weights": {"default": 0.0, 3500: 0.0, 4000: 1.0}, "provenance": "assumed",
                                 "generation": "none", "source": "TEST"}
    p = realize(load(d), 100, 0, variation="off", thermal_hook=None)
    assert np.all(p.world["q_hyp"] == 2) and np.all(p.optics.q_loaded == 4000)


def test_pin_nominal_of_absolute_entry_keeps_the_leaf():
    """An absolute interval without a nominal, pinned to 'nominal', keeps the leaf value (it used to give nan)."""
    p = realize(load(MINI, alts={"uncertainty.q_loaded": "nominal"}), 3, 0, variation="off", thermal_hook=None)
    assert np.all(p.optics.q_loaded == 3800)


def test_lognormal_dist_truncation_is_exact():
    d = mini()
    d["uncertainty"]["dark"] = {"kind": "dist", "target": "ring.monitor.dark_i0_na", "dist": "lognormal",
                                "mean": 5.0, "sigma": 0.8, "truncate": [1.0, 30.0], "provenance": "assumed",
                                "generation": "GF", "source": "R01"}
    x = realize(load(d), 4000, 0, variation="off", thermal_hook=None).world["dark"]
    assert x.min() >= 1.0 and x.max() <= 30.0
    assert np.median(x) == pytest.approx(5.0, rel=0.05)          # mean is the median of the lognormal
    d["uncertainty"]["dark"].update(dist="normal", mean=10.0, sigma=8.0, truncate=[1.0, 30.0])
    y = realize(load(d), 4000, 0, variation="off", thermal_hook=None).world["dark"]
    assert y.min() >= 1.0 and y.max() <= 30.0


def test_lag_bound_without_hook_and_without_far_keys():
    """The interim fallback of the 07 P0-1 lag bound works without the optional far-field keys (KeyError before)."""
    d = mini()
    for k in ("xt_far_um", "xt_tau_far_s"):
        del d["ring"]["thermal"][k]
    cfg = load(d)
    g_none = realize(cfg, 2, 0, thermal_hook=None).diagnostics["selfheat_lag_gain"]
    g_hook = realize(cfg, 2, 0).diagnostics["selfheat_lag_gain"]
    assert 0 < g_none < SELFHEAT_LAG_GAIN_MAX and 0 < g_hook < SELFHEAT_LAG_GAIN_MAX
    assert g_none == pytest.approx(g_hook, rel=1.0)                # same order: one law, two bookkeepings


def test_non_finite_values_are_refused(cfg):
    with pytest.raises(RealizeError, match="non-finite|close"):
        realize(cfg, 2, 0, grid={"ring.optics.q_loaded": [3800.0, float("nan")]}, thermal_hook=None)


def test_select_cuts_nested_thermal_objects(cfg):
    """select(b) is the B = 1 view everywhere, including the assembled ThermalSystem and ThermalInputs."""
    thermal = pytest.importorskip("mrm.plant.thermal")
    if not hasattr(thermal, "assemble"):
        pytest.skip("mrm.plant.thermal.assemble not available")
    big = realize(cfg, 5, 20)
    one = realize(cfg, 1, 22)
    sel = big.select(2)
    for name in ("taus_s", "res_k_per_w"):
        np.testing.assert_array_equal(getattr(sel.thermal.system, name), getattr(one.thermal.system, name))
    assert sel.thermal.system.batch == 1
    for tin in (sel.thermal.inputs, sel.thermal_inputs):
        np.testing.assert_array_equal(tin.z_heater_ring_k_per_w, one.thermal_inputs.z_heater_ring_k_per_w)
        assert all(np.asarray(v).shape[0] == 1 for v in tin.scales.values())
        assert all(v.shape == (1, 1) for v in tin.interim.values())
        assert tin.scaling_hypothesis == one.thermal_inputs.scaling_hypothesis
    assert sel.diagnostics["geometry_draws_nm"]["width"].shape == (1, cfg.n_rings)
    with pytest.raises(IndexError):
        big.select(5)


def test_thermal_hypotheses_are_switches_not_scales():
    """07 4.8 names the hypotheses thermal.family and thermal.stack; under uncertainty.thermal they must act through
    their per-world switches, not be passed to ThermalScales (a RealizeError before)."""
    d = mini()
    d["uncertainty"]["thermal"]["family"] = {
        "kind": "hypothesis", "nominal": "inring", "provenance": "assumed", "generation": "none", "source": "TEST",
        "choices": {"inring": {"ring.thermal.family": "inring_noundercut"},
                    "metal": {"ring.thermal.family": "metal_above"}}}
    p = realize(load(d), 16, 0)
    assert set(p.thermal_inputs.scales) == {"time_scale_fast", "time_scale_slow"}
    assert set(np.unique(p.thermal_inputs.family)) == {0, 2}
    np.testing.assert_array_equal(p.thermal_inputs.family[:, 0] == 2, p.world["thermal.family"] == 1)
