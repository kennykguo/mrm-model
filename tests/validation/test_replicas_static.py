"""Test the replicas: the model versions of the published measurement protocols (mrm/validation/replicas.py).

Read this first. A replica runs a published measurement (for example "read the loaded Q from the 3 dB width of a
low-power sweep") on the model, so that its result can be compared with the published number. These tests check the
replicas themselves, not the hardware: no target tolerance is asserted here (the validation report scores the
targets). What is checked:

- every name of targets_io.REPLICA_NAMES is implemented and every target's replica runs and returns its status;
- each protocol runs on ``g26`` and ``g25_test_vehicle`` and returns finite numbers in the target's units and ballpark;
- the static model closes against the plant's own relations: Q from the half-depth width against the realized Q_L
  (R4), the heater power per FSR against the R2 anchor, the cold offset against R3, the low-power bias ladder against
  the junction table, the driver-law ordering of the apparent sensitivity;
- the small-signal EO response against a direct integration of the TCMT ODE;
- the quasi-static steady state is a fixed point of the Python reference plant tick
  (tests/verification/test_plant_wiring.py::ReferencePlant), and the closed-loop replicas run through a small
  ``run_fn`` adapter built on that reference plant (the engine's runner replaces it later).
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mrm.plant import electrical
from mrm.validation import replicas
from mrm.validation import targets_io

PRESETS = ("g26", "g25_test_vehicle")
K0 = 273.15                   # 0 degC in K


@pytest.fixture(scope="module")
def targets():
    """{target id: Target} of every role."""
    return {t.id: t for role in targets_io.TARGET_ROLES for t in targets_io.load_targets(role)}


@pytest.fixture(scope="module")
def cfgs():
    """{preset: Config} of both presets."""
    return {p: replicas.load_config(p) for p in PRESETS}


@pytest.fixture(scope="module")
def pps(cfgs):
    """{preset: realized PlantParams} of both presets (the replicas' own nominal world)."""
    return {p: replicas.resolve_plant(c)[1] for p, c in cfgs.items()}


def _finite(x) -> bool:
    """True when every number in ``x`` (a number, an array or a dict of them) is finite."""
    if isinstance(x, dict):
        return all(_finite(v) for v in x.values())
    return bool(np.all(np.isfinite(np.asarray(x, dtype=float))))


# ------------------------------------------------------------------------------------------------------------- registry
def test_registry_implements_every_protocol():
    assert set(replicas.REPLICAS) == set(targets_io.REPLICA_NAMES)
    for name, fn in replicas.REPLICAS.items():
        assert callable(fn) and inspect.getcomments(fn), name


def test_every_target_names_a_registered_replica(targets):
    for t in targets.values():
        assert t.replica in replicas.REPLICAS, t.id
        assert replicas.preset_for(t) in PRESETS


# -------------------------------------------------------------------------------------------------- G25 bench protocols
@pytest.mark.parametrize("preset", PRESETS)
def test_q_from_half_depth_width_closes_r4(preset, pps, targets):
    """A low-power sweep read at half depth gives the realized loaded Q (R4) to the TCMT accuracy (~F^-2)."""
    pp = pps[preset]
    r = replicas.loaded_q_from_3db_width(pp, targets["CAL-1"])
    q = float(pp.optics.q_loaded[0, 0])
    assert r.ok and r.unit == "1"
    assert abs(r.value / q - 1.0) < 0.02
    assert "topology=all_pass" in r.variants and _finite(r.variants["topology=all_pass"])


@pytest.mark.parametrize("preset", PRESETS)
def test_notch_depth_and_tuning_ballpark(preset, pps, targets):
    pp = pps[preset]
    er = replicas.notch_er_vs_heater_v(pp, targets["CAL-2"])
    assert len(er.value) == len(targets["CAL-2"].value["x"]) and _finite(er.value)
    assert all(5.0 < v < 150.0 for v in er.value)          # near critical coupling the depth diverges
    v2 = replicas.notch_er_vs_heater_v(pp, targets["VAL-2a"])
    assert isinstance(v2.value, float) and 5.0 < v2.value < 150.0
    sh = replicas.resonance_vs_heater_v(pp, targets["CAL-4"])
    y = np.asarray(sh.value)
    assert np.all(np.diff(y) > 0) and 0.0 < y[0] < 0.3          # monotone; ~quadratic start
    tr = replicas.tuning_range_at_v(pp, targets["VAL-1"])
    assert 5.0 < tr.value < 30.0
    assert tr.value > y[-1]                                         # 1.5 V tunes further than 1.4 V


@pytest.mark.parametrize("preset", PRESETS)
def test_heater_power_per_fsr_closes_r2(preset, pps, targets):
    """The bench protocol (voltage sweep with the TCR until the next order arrives) reproduces the R2 anchor to the
    lambda^2 scaling of the FSR between the line and the cold resonance (well below 2 %)."""
    pp = pps[preset]
    p = replicas.heater_power_per_fsr(pp, targets["CAL-3a"], ring=0)
    v = replicas.heater_power_per_fsr(pp, targets["CAL-3b"], ring=0)
    assert p.unit == "mw" and v.unit == "v"
    assert abs(p.value / (float(pp.heater.p_fsr_w[0, 0]) * 1e3) - 1.0) < 0.015
    assert 0.9 < v.value < 1.5
    r = replicas.StaticRing(pp, driver_law="bench_voltage")
    assert float(r.power_from_u([v.value], r.t_ref)[0]) * 1e3 == pytest.approx(p.value, rel=1e-9)


@pytest.mark.parametrize("preset", PRESETS)
def test_cold_offset_closes_r3(preset, pps, targets):
    pp = pps[preset]
    r = replicas.cold_resonance_offset(pp, targets["CAL-7"])
    assert r.value == pytest.approx(float(pp.optics.resonance_offset_m[0, 0]) * 1e9, abs=2e-3)
    assert 0.0 < r.variants["at_preset_power_red_sweep"] < r.value       # self-heating drags the dip red


def test_cold_offset_variant_uses_the_same_bias(pps, targets):
    """CAL-7: the variant at the preset's line power is the same read-out as the primary, so it uses the same
    ``bias_v``. A 2 V reverse bias moves the primary by the junction shift (about -0.10 nm on g25_test_vehicle) and
    the variant by about the same amount (the self-heating drag changes a little with the bias). Before the fix the
    variant always used 0 V and did not move."""
    pp = pps["g25_test_vehicle"]
    r0 = replicas.cold_resonance_offset(pp, targets["CAL-7"])
    r2 = replicas.cold_resonance_offset(pp, targets["CAL-7"], bias_v=2.0)
    shift_primary_nm = r2.value - r0.value
    shift_variant_nm = r2.variants["at_preset_power_red_sweep"] - r0.variants["at_preset_power_red_sweep"]
    assert abs(shift_primary_nm) > 0.05
    assert shift_variant_nm == pytest.approx(shift_primary_nm, rel=0.3)


def test_bias_ladder_low_power_is_the_junction_shift(pps, targets):
    """At P_in -> 0 the swept dip moves exactly by the junction table's shift; at +4 dBm the self-heating relief
    (less absorption at higher reverse bias for this budget) is included, so the shift differs."""
    pp = pps["g25_test_vehicle"]
    r = replicas.resonance_shift_vs_bias(pp, targets["CAL-5"])
    lowp = np.asarray(r.variants["p_in_dbm=None"])
    junc = np.asarray(r.conditions["junction_shift_only_pm"])
    np.testing.assert_allclose(lowp, junc, atol=0.3)
    assert _finite(r.value) and np.all(np.asarray(r.value) > 0)
    assert r.conditions["p_in_dbm"] == 4.0


# ------------------------------------------------------------------------------------------------- apparent sensitivity
def test_apparent_shift_orders_the_driver_laws(cfgs):
    """Driver-law ordering: at a fixed code an ambient sweep reads below the ring's dlambda/dT for voltage drive,
    above it for current drive, and equal to it for regulated power (P_in -> 0, so only the TCR acts). H86_sensor
    reports the aggressor path through a thermometer that sees 0.8 of the ring's rise: 1/0.8 times the ambient
    reading."""
    r = replicas.apparent_shift_fixed_code(cfgs["g26"], hypotheses=["H86_sensor"],
                                    driver_laws=["voltage", "current", "power"], p_in_dbm=None, n_steps=5)
    rows = {(x["hypothesis"], x["heating"], x["driver_law"]): x for x in r.detail["rows"]}
    pw = rows[("plant", "ambient", "power")]
    assert pw["apparent_pm_per_k"] == pytest.approx(pw["ring_mean_s_pm_per_k"], rel=2e-3)
    v = rows[("plant", "ambient", "voltage")]["apparent_pm_per_k"]
    c = rows[("plant", "ambient", "current")]["apparent_pm_per_k"]
    assert v < pw["apparent_pm_per_k"] < c
    hs_amb = rows[("H86_sensor", "ambient", "voltage")]["apparent_pm_per_k"]
    hs_agg = rows[("H86_sensor", "aggressor", "voltage")]["apparent_pm_per_k"]
    frac = rows[("H86_sensor", "aggressor", "voltage")]["reference_rise_frac"]
    assert frac == pytest.approx(0.8)
    pp = replicas.resolve_plant(cfgs["g26"])[1]
    if replicas._aggressor_ratio(pp, replicas.StaticRing(pp)) == pytest.approx(1.0):
        # interim model: the aggressor heats the ring and its heater resistor alike, so only the thermometer differs
        assert hs_agg == pytest.approx(hs_amb / frac, rel=1e-6)
    assert 40.0 < r.value < 120.0 and r.unit == "pm_per_k"


# ------------------------------------------------------------------------------------------------------- locks and maps
@pytest.mark.parametrize("preset", PRESETS)
def test_lock_points(preset, pps):
    r = replicas.StaticRing(pps[preset])
    kw = dict(t_base=298.15, p_in_w=r.p_line_w, bias_v=1.05, modulate=True)
    pk = r.lock(mode="peak", **kw)
    hm = r.lock(mode="half_max", **kw)
    orc = r.lock(mode="oracle", **kw)
    rs = r.lock(mode="resonance", **kw)
    assert hm.p_drop_w == pytest.approx(0.5 * pk.drop_peak_w, rel=2e-3)
    assert hm.t_ring_k > pk.t_ring_k and hm.stable                     # hot side, stable against self-heating
    assert orc.reloma >= hm.reloma and orc.stable
    assert abs(rs.detuning_m) < 1e-15
    assert 0 < hm.u < 1 and hm.p_heater_w > 0


@pytest.mark.parametrize("preset", PRESETS)
def test_monitor_current_at_lock_ballpark(preset, pps, targets):
    r = replicas.monitor_current_at_lock(pps[preset], targets["CAL-8"])
    assert r.unit == "ua" and 1.0 < r.value < 100.0
    pk = r.variants["setpoint=peak"]
    assert r.value == pytest.approx(0.5 * pk, rel=0.01)                # half maximum (dark current ~nA)
    assert r.detail["half_max"]["stable"]


@pytest.mark.parametrize("preset", PRESETS)
def test_reloma_map_outputs(preset, pps, targets):
    pp = pps[preset]
    lev = replicas.relOMA_map(pp, targets["CAL-14"], with_variants=False)
    assert -30.0 < lev.value < 0.0 and lev.unit == "db"
    c = replicas.relOMA_map(pp, targets["VAL-4a"])
    assert c.unit == "v"
    assert math.isnan(c.value) or 1.0 <= c.value <= 4.0
    rel = c.variants["relative_to_low_bias_level"]
    assert math.isnan(rel) or 1.0 <= rel <= 4.0
    sp = replicas.relOMA_map(pp, targets["VAL-4c"], with_variants=False)
    assert 0.0 <= sp.value < 20.0
    b4 = replicas.relOMA_map(pp, targets["BND-4"], with_variants=False)
    assert 0.0 <= b4.value <= sp.value + 1e-9
    v5 = replicas.relOMA_map(pp, targets["VAL-5"], with_variants=False)
    assert set(v5.value) == {"below_peak_k", "above_peak_k"}
    assert _finite(v5.value) and min(v5.value["below_peak_k"] + v5.value["above_peak_k"]) > 0
    assert v5.detail["primary"]["peak_c"] == pytest.approx(69.6, abs=0.05)
    targets_io.evaluate(targets["VAL-5"], v5.value)                            # shaped as evaluate expects


@pytest.mark.parametrize("preset", PRESETS)
def test_stabilizer_codes_ballpark(preset, cfgs, targets):
    """One sampled die, the plant's own driver law: codes inside the DAC range and the three CAL-9 statistics well
    formed (one run gives all of them in detail); VAL-6's ambient excursion is positive (colder needs more code)."""
    cfg = cfgs[preset]
    a = replicas.stabilizer_code_vs_T(cfg, targets["CAL-9a"], n_dies=1, driver_mode="none", with_variants=False)
    lo, hi = a.value
    assert 0.0 <= lo <= hi <= 65535.0
    st = a.detail["primary"]["per_die"][0]
    assert set(st["lane_offsets"]) == {"at_code_maxima", "at_code_minima"} and _finite(st["lane_offsets"])
    assert 0.0 <= st["code_range"][0] <= st["code_range"][1] <= 65535.0
    assert 0.0 < a.detail["primary"]["calibration_residual_rms_k"][0] < 1.0
    targets_io.evaluate(targets["CAL-9a"], a.value)
    targets_io.evaluate(targets["CAL-9b"], replicas._median_stats([st], "lane_offsets"))
    targets_io.evaluate(targets["CAL-9c"], replicas._median_stats([st], "code_range"))
    v6 = replicas.stabilizer_code_vs_T(cfg, targets["VAL-6"], n_dies=1, driver_mode="none", with_variants=False)
    assert v6.value > 0


def test_bistability_grows_with_power(pps):
    pp = pps["g25_test_vehicle"]
    lo = replicas.bistability_sweep(pp, p_in_dbm=-10.0)
    mid = replicas.bistability_sweep(pp)
    hi = replicas.bistability_sweep(pp, p_in_dbm=8.0)
    assert not lo.value["bistable"]
    assert mid.value["bistable"] and hi.value["bistable"]
    assert hi.value["hysteresis_mw"] > mid.value["hysteresis_mw"] > 0.0
    assert hi.detail["bistability_ratio"] > mid.detail["bistability_ratio"] > lo.detail["bistability_ratio"]


# --------------------------------------------------------------------------------------------------------- EO bandwidth
def test_eo_response_matches_the_tcmt_ode(pps):
    """The linearized response equals a direct time-domain integration of the TCMT with the junction RC at small
    modulation (independent derivation: no shared formula)."""
    from scipy.integrate import solve_ivp
    r = replicas.StaticRing(pps["g25_test_vehicle"])
    f = np.array([1e6, 20e9, 45e9])
    an = replicas._eo_response(r, 1.0, f)
    kap, ke, D = an["kappa_rad_per_s"], an["ke_rad_per_s"], an["detuning_rad_per_s"]
    tau_rc = an["r_s_ohm"] * an["c_j_f"]
    eps = 1e-4 * kap                                                     # small modulation of the resonance (rad/s)
    for fi, hi in zip(f[1:], an["h"][1:]):
        W = 2 * np.pi * fi

        def rhs(t, y):
            a = y[0] + 1j * y[1]
            vj = y[2]
            da = (-1j * (D - eps * vj) - 0.5 * kap) * a - 1j * np.sqrt(ke)
            return [da.real, da.imag, (np.cos(W * t) - vj) / tau_rc]
        a0 = -1j * np.sqrt(ke) / (1j * D + 0.5 * kap)
        n_per = 40
        t_end = n_per * 2 * np.pi / W
        t_eval = np.linspace(t_end - 10 * 2 * np.pi / W, t_end, 2001)[:-1]
        sol = solve_ivp(rhs, (0.0, t_end), [a0.real, a0.imag, 1.0], t_eval=t_eval, rtol=1e-10, atol=1e-14,
                        method="DOP853")
        a = sol.y[0] + 1j * sol.y[1]
        p = np.abs(1.0 - 1j * np.sqrt(ke) * a) ** 2
        comp = 2.0 * np.mean(p * np.exp(-1j * W * sol.t))               # complex amplitude of the e^{jWt} part
        assert abs(comp) / eps == pytest.approx(abs(hi), rel=0.01)


@pytest.mark.parametrize("preset", PRESETS)
def test_eo_bandwidth_ballpark(preset, pps, targets):
    pp = pps[preset]
    a = replicas.eo_bandwidth_at_bias(pp, targets["CAL-10"])
    assert a.unit == "ghz" and 5.0 < a.value < 300.0 and a.conditions["method"] == "waveform"
    assert a.variants["r_drv_ohm=0"] > a.value                            # less series R, more bandwidth
    b = replicas.eo_bandwidth_at_bias(pp, targets["VAL-3b"])
    assert b.unit == "1" and 0.8 < b.value < 1.6


def test_eo_waveform_mode_matches_the_linearized_replica(pps):
    """Independent cross-check: mrm.waveform.eo_bandwidth (another owner's TCMT linearization) with the loss
    modulation off agrees with the replica's own index-only linearized TCMT x RC at the same source resistance; the
    loss modulation and the driver's 50 ohm both lower the bandwidth."""
    pp = pps["g25_test_vehicle"]
    r = replicas.StaticRing(pp)
    f = np.concatenate([[1e6], np.geomspace(1e8, 1e12, 4001)])
    for v in (1.0, 2.0):
        for rs in (0.0, 50.0):
            w = replicas._eo_waveform(pp, 0, 0, v, "max_slope", rs, loss_modulation=False)
            lin = replicas._eo_response(r, v, f, r_source_ohm=rs)
            assert w["bw_hz"] == pytest.approx(lin["bw_hz"], rel=0.005)
            assert w["c_j_f"] == pytest.approx(lin["c_j_f"], rel=1e-9)
        full = replicas._eo_waveform(pp, 0, 0, v, "max_slope", None)
        assert full["r_drv_ohm"] == pytest.approx(float(np.asarray(pp.datapath.driver_output_resistance_ohm)
                                                        .reshape(-1)[0]))
        assert full["bw_hz"] < replicas._eo_waveform(pp, 0, 0, v, "max_slope", None, loss_modulation=False)["bw_hz"]


# ---------------------------------------------------------------------------------------------------------- populations
@pytest.mark.parametrize("preset", PRESETS)
def test_population_replicas(preset, cfgs, targets):
    cfg = cfgs[preset]
    pr = replicas.pd_responsivity(cfg, targets["CAL-11"], batch=64)
    assert 0.85 <= pr.value[0] <= pr.value[1] <= 1.06
    a = replicas.laser_line_statistics(cfg, targets["CAL-12a"], batch=64)
    assert 0.0 <= a.value <= 20.0 + 1e-9
    b = replicas.laser_line_statistics(cfg, targets["CAL-12b"], batch=64)
    assert b.value["q25"] <= b.value["median"] <= b.value["q75"]
    c = replicas.laser_line_statistics(cfg, targets["CAL-12c"])
    assert c.value == pytest.approx(0.02, rel=0.03)
    d = replicas.laser_line_statistics(cfg, targets["CAL-12d"])
    assert d.value == pytest.approx(2.5, rel=1e-6)


# ------------------------------------------------------------------------------------------------- aggressor and bounds
@pytest.mark.parametrize("preset", PRESETS)
def test_aggressor_trajectory(preset, pps, targets):
    pp = pps[preset]
    a = replicas.aggressor_trajectory(pp, targets["CAL-13a"])
    t = np.asarray(a.value["t_s"])
    assert t[0] == pytest.approx(0.0, abs=1e-12) and t[-1] < 0.1307 and _finite(a.value["y"])
    assert a.detail["rms_vs_target_c"] < 1.0                              # a calibrated scenario (not a prediction)
    ev = targets_io.evaluate(targets["CAL-13a"], a.value)
    assert ev.max_abs_residual == pytest.approx(a.detail["rms_vs_target_c"], rel=1e-9)
    b = replicas.aggressor_trajectory(pp, targets["CAL-13b"])
    assert b.unit == "k_per_s" and 600.0 < b.value < 1000.0


def test_bounds(pps, targets):
    g26 = pps["g26"]
    b1 = replicas.adjacent_order_penalty(g26, targets["BND-1"], batch=1)
    assert 0.0 < b1.value < 100.0 and b1.detail["orders_in_span"] > 0
    g25 = replicas.adjacent_order_penalty(pps["g25_test_vehicle"], targets["BND-1"], batch=1)
    assert g25.value == 1e9                                               # one line: no other order in its span
    for p in PRESETS:
        b2 = replicas.order_hop_margin(pps[p], targets["BND-2"], n_dies=1)
        assert b2.unit == "nm" and -20.0 < b2.value < 10.0


def test_order_bookkeeping(pps):
    r = replicas.StaticRing(pps["g26"])
    fsr = replicas._fsr_m(r)
    assert r.order_offset_m(1) == pytest.approx(-fsr, rel=0.02)          # the next bluer order
    lam = r.lam_line
    t0 = r.t_res(lam, 0.0, 0)
    assert r.first_order(lam, 0.0, r.t_ref) == 0
    assert r.first_order(lam, 0.0, t0 + 1.0) == 1                        # order 0 would need cooling


# --------------------------------------------------------------------------------------------- third party, feasibility
def test_third_party_replicas_block_or_read_their_artefacts(targets):
    for tid in ("VAL-7a", "VAL-7b", "VAL-7c", "VAL-8a", "VAL-8e"):
        r = replicas.run_target(targets[tid])
        assert r.status in ("ok", "blocked"), (tid, r.notes)
        if r.status == "blocked":
            assert r.value is None and r.notes
        else:
            assert _finite(r.value)


def test_feasibility_is_blocked_until_controllers_and_engine_exist(targets):
    from mrm.control import registry
    names = targets["BND-3a"].replica_args["controllers"]
    r = replicas.run_target(targets["BND-3a"])
    if not set(names) <= set(registry.available()):
        assert r.status == "blocked" and "controllers" in r.notes[0]
    else:
        assert r.status in ("ok", "blocked")


@pytest.mark.parametrize("controllers", [[], None])
def test_feasibility_without_controllers_is_blocked(pps, targets, controllers):
    """BND-3a with no controllers listed has nothing to run: the replica is ``blocked`` and says why. Before the fix
    it raised TypeError (an ``error`` result) when a run function was given."""
    r = replicas.run_target(targets["BND-3a"], pps["g26"], controllers=controllers, run_fn=lambda req: {})
    assert r.status == "blocked" and "no controllers" in r.notes[0]


def test_predictions_file(tmp_path):
    ids = ["CAL-1", "CAL-7", "VAL-7a", "BND-3a"]
    recs = replicas.predict_all(ids=ids)
    p = replicas.write_predictions(tmp_path / "pred.json", records=recs)
    doc = json.loads(p.read_text())
    assert doc["schema"] == "mrm.validation.predictions/1"
    assert set(doc["frozen_sha256"]) == {"validation", "bounds"}
    got = {r["id"]: r for r in doc["records"]}
    assert set(got) == set(ids)
    assert got["CAL-1"]["replica_status"] == "ok" and got["CAL-1"]["evaluation"]["level"] in ("pass", "fail")
    assert got["CAL-7"]["preset"] == "g25_test_vehicle"
    assert got["BND-3a"]["replica_status"] in ("blocked", "ok")
    for r in doc["records"]:
        assert {"value", "unit", "tolerance", "predicted", "conditions"} <= set(r)


# ------------------------------------------------------------------------------------------------------ reference plant
def _reference_plant_class():
    """The Python reference plant class of tests/verification/test_plant_wiring.py (loaded from its file)."""
    path = Path(__file__).resolve().parents[1] / "verification" / "test_plant_wiring.py"
    spec = importlib.util.spec_from_file_location("_mrm_reference_wiring", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ReferencePlant


def _warm_plant(pp, cfg, scenario, code, warm):
    """A ReferencePlant at the quasi-static steady state ``warm`` (heater code held by the DAC from the start)."""
    ReferencePlant = _reference_plant_class()
    plant = ReferencePlant(pp, cfg, scenario=scenario)
    sysm = pp.thermal.system
    plant.heater = electrical.HeaterBank(plant.heater_p, initial_code=np.asarray(code, dtype=np.int64))
    u = np.zeros((pp.batch, sysm.n_src))
    u[:, sysm.heater_src] = warm["p_heater_w"]
    u[:, sysm.wg_src] = warm["p_abs_w"]
    if sysm.junction_src.size:
        u[:, sysm.junction_src] = warm.get("p_j_w", 0.0)
    if sysm.driver_src.size:
        u[:, sysm.driver_src] = warm.get("p_d_w", 0.0)
    plant.x[:] = sysm.steady_state(u)
    plant.t_obs[:] = sysm.outputs(plant.x, warm["t_base_k"])
    plant.t_ring[:] = plant.t_obs[:, sysm.ring_obs]
    plant.t_heater[:] = plant.t_obs[:, sysm.heater_obs]
    plant.t_branch[:] = plant.t_ring
    plant.p_prev[:] = warm["p_abs_w"]
    plant.u_heater[:] = warm["p_heater_w"]
    return plant


def _cw_scenario(t_base_k, bias_v):
    """A static CW scenario: base temperature ``t_base_k`` (K), driver off (no data swing), bias ``bias_v`` (V)."""
    from mrm.scenarios.generators import Scenario, const
    return Scenario(name="cw_static", defaults={"thermal.t_base_k": float(t_base_k)}, components=(
        const("data.swing_scale", 0.0, absolute=True), const("data.driver_enable", 0.0, absolute=True),
        const("tx.bias_v", float(bias_v), absolute=True)))


@pytest.fixture(scope="module")
def g25_reference(cfgs, pps):
    """(Config, PlantParams) of g25_test_vehicle."""
    return cfgs["g25_test_vehicle"], pps["g25_test_vehicle"]


def test_static_state_is_a_fixed_point_of_the_reference_plant(g25_reference):
    """The quasi-static solution (ROM DC gains, heater law with TCR, phase model, exact ring, self-heating) is a
    steady state of the reference plant tick: started there at a fixed heater code on the hot branch, CW, the
    plant stays within 0.05 K and 0.5 % of heater power over 300 us."""
    cfg, pp = g25_reference
    r = replicas.StaticRing(pp)
    t_base = 298.15
    # the bias DAC applies 1.0 V to within half an LSB (~21 uV, a ~1.5 fm resonance shift): solve at 1.0 V
    v_dac = 1.0
    lp = r.lock(mode="half_max", t_base=t_base, p_in_w=r.p_line_w, bias_v=v_dac, modulate=False)
    code = int(np.rint(r.u_to_code(lp.u)))
    u0 = float(r.code_to_u(code))
    t_static = float(replicas._fixed_drive_track(r, u0, [(t_base, 0.0, 0.0)], p_in_w=r.p_line_w, lam=r.lam_line,
                                          v=v_dac)[0])
    st = r.state_grid(r.lam_line, [t_static], t_base=t_base, p_in_w=r.p_line_w, v_lo=v_dac)
    warm = {"p_heater_w": float(st["p_h"][0, 0]), "p_abs_w": float(st["p_abs"][0, 0]), "t_base_k": t_base}
    plant = _warm_plant(pp, cfg, _cw_scenario(t_base, 1.0), [[code]], warm)
    for k in range(300):
        out = plant.tick(k)
    assert abs(float(out["bias"].v_avg[0, 0]) - v_dac) < 1e-3
    assert lp.stable and t_static > lp.t_res_k                             # the hot (stable) branch
    assert float(plant.t_ring[0, 0]) == pytest.approx(t_static, abs=0.05)
    assert float(out["heater"].p_w[0, 0]) == pytest.approx(warm["p_heater_w"], rel=5e-3)


class ReferenceLockRunner:
    """run_fn adapter for the closed-loop replicas (LockRunRequest -> traces) on the Python reference plant, with an
    integral controller on the true drop-monitor current (no ADC), updated every ``period`` ticks. Its gain is
    ``frac`` of the static inverse sensitivity d(code)/d(current) at the warm state.

    TEST ADAPTER ONLY: it reads the plant's true photocurrent (Truth, no ADC, no noise, no measurement latency) and
    its gain comes from the true static model, so it checks that the quasi-static replica is a fixed point of the
    dynamic plant; it is not a controller under evaluation (a controller never reads Truth). One ring (N = 1) only:
    the gain is ring 0's."""

    def __init__(self, period: int = 10, frac: float = 0.3):
        self.period, self.frac = period, frac

    def __call__(self, req):
        from mrm.plant.optics import RING_OUTPUT_ROW_TOTAL_CLOCKWISE_DROP_POWER, RING_LINE_OUTPUT_ROW_INPUT_POWER
        from mrm.scenarios.generators import Scenario, const

        pp, cfg = req.pp, req.cfg
        assert pp.n_rings == 1, "ReferenceLockRunner computes ring 0's gain only"
        warm = req.options["warm_state"]
        bias = float(req.options["bias_v"])
        sc = req.scenario + Scenario(name="bias", components=(const("tx.bias_v", bias, absolute=True),))
        code = np.asarray(req.initial_code, dtype=float).copy()
        w = {k: (np.asarray(v)[0, 0] if np.ndim(v) else v) for k, v in warm.items()}
        plant = _warm_plant(pp, cfg, sc, code.astype(np.int64), w)
        # static sensitivity of the monitor current to the code at the warm state (central difference)
        r = replicas.StaticRing(pp)
        tb = float(warm["t_base_k"])
        v_lo, v_hi = r.nrz_levels(bias)
        p_j, p_d = r.data_heat(v_lo, v_hi)

        def current(c):
            u = float(r.code_to_u(c))
            t = replicas._fixed_drive_track(r, u, [(tb, 0.0, 0.0)], p_in_w=r.p_line_w, lam=r.lam_line, v=bias)
            st = r.state_grid(r.lam_line, t, t_base=tb, p_in_w=r.p_line_w, v_lo=v_lo, v_hi=v_hi, p_j=p_j, p_d=p_d)
            return float(r.monitor_current_a(r.p_line_w * st["drop_avg"][0, 0], r.p_line_w)[0])

        c0 = float(code[0, 0])
        sens = (current(c0 + 20) - current(c0 - 20)) / 40.0
        gain = self.frac / abs(sens)
        hw = plant.mon.hw
        n = int(round(req.duration_s / pp.timing.dt_s))
        rec = {
            "t_s": np.zeros(n),
            "monitor_a": np.zeros((n, pp.batch, pp.n_rings)),
            "heater_code": np.zeros((n, pp.batch, pp.n_rings)),
            "t_ring_k": np.zeros((n, pp.batch, pp.n_rings)),
        }
        cmd = None
        for k in range(n):
            out = plant.tick(k, heater_cmd=cmd)
            cmd = None
            p_bus = out["out_rl"][:, RING_LINE_OUTPUT_ROW_INPUT_POWER].sum(axis=2)
            i_s, i_d = hw.photocurrent_a(
                out["out_r"][:, RING_OUTPUT_ROW_TOTAL_CLOCKWISE_DROP_POWER],
                p_bus,
                plant.t_obs[:, pp.thermal.system.pd_obs],
            )
            i_m = i_s + i_d
            if (k + 1) % self.period == 0:
                code = np.clip(code + gain * (i_m - req.setpoint_a), 0, 65535)
                cmd = np.rint(code).astype(np.int64)
            rec["t_s"][k] = (k + 1) * pp.timing.dt_s
            rec["monitor_a"][k] = i_m
            rec["heater_code"][k] = code
            rec["t_ring_k"][k] = plant.t_ring
        return rec


def test_closed_loop_replica_through_the_reference_adapter(g25_reference, targets):
    """monitor_current_at_lock(method='dynamic') runs the lock through a run_fn (here the reference-plant adapter
    with NRZ traffic, data heat, self-heating and the heater TCR) and agrees with the quasi-static replica."""
    cfg, pp = g25_reference
    dyn = replicas.monitor_current_at_lock(pp, targets["CAL-8"], method="dynamic", run_fn=ReferenceLockRunner(),
                                    duration_s=4e-4, setpoint="half_max")
    qs = replicas.monitor_current_at_lock(pp, targets["CAL-8"], setpoint="half_max")
    assert dyn.ok and dyn.conditions["method"] == "dynamic"
    assert dyn.value == pytest.approx(qs.value, rel=0.015)
    assert dyn.variants["quasi_static"] == pytest.approx(qs.value, rel=1e-12)


@pytest.mark.slow
def test_dynamic_code_excursion_matches_quasi_static(g25_reference):
    """An ambient step under the reference-plant lock moves the heater code by the quasi-static excursion (the 3 ms
    package tail of the interim model sets the run length)."""
    cfg, pp = g25_reference
    res = replicas.stabilizer_code_vs_T(pp, None, t_c=[25.0, 35.0], method="dynamic", run_fn=ReferenceLockRunner(),
                                 duration_s=1.2e-2)
    assert res.value == pytest.approx(res.variants["quasi_static"], rel=0.05)


def test_run_target_never_passes_silently(targets, monkeypatch):
    """A replica that raises gives status 'error' with the message, never a prediction."""
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setitem(replicas.REPLICAS, "loaded_q_from_3db_width", boom)
    r = replicas.run_target(targets["CAL-1"])
    assert r.status == "error" and r.value is None and "synthetic failure" in r.notes[0]


def test_default_run_fn_blocks_without_the_engine():
    try:
        fn = replicas.default_run_fn()
    except replicas.ReplicaBlocked as e:
        assert "runner" in str(e)
    else:
        assert callable(fn)


def test_results_are_json_serializable(pps, targets):
    r = replicas.loaded_q_from_3db_width(pps["g25_test_vehicle"], targets["CAL-1"])
    json.dumps(r.to_json(), allow_nan=False)


def test_feasibility_protocol_plumbing(pps, targets, monkeypatch):
    """With the controllers registered and a run function, BND-3a reports the signed error band of the best
    controller over the last aggressor period (here a synthetic run function: +-1 % and +-2 % ripples). The stated
    controller limits reach the run function as numbers, and the run starts at the scenario's base temperature."""
    from mrm.control import registry
    names = targets["BND-3a"].replica_args["controllers"]
    monkeypatch.setattr(registry, "available", lambda: sorted(set(names)))
    calls = []

    def fake_run(req):
        assert req.options["max_update_rate_hz"] == pytest.approx(1e6)
        assert req.options["min_total_latency_s"] == pytest.approx(2e-6)
        assert req.options["warm_state"]["t_base_k"] == pytest.approx(
            float(req.scenario.defaults.get("thermal.t_base_k", 298.15)))
        calls.append(req.controller)
        per = req.scenario.aggressors[0].target.period_s
        t = np.linspace(0.0, req.duration_s, 3001)
        amp = 0.01 if req.controller == names[1] else 0.02
        i = req.setpoint_a[0, 0] * (1.0 + amp * np.sin(2 * np.pi * t / per))
        return {"t_s": t, "monitor_a": i[:, None, None] * np.ones((1, req.pp.batch, req.pp.n_rings)),
                "heater_code": np.zeros((t.size, req.pp.batch, req.pp.n_rings))}
    r = replicas.aggressor_feasibility(pps["g26"], targets["BND-3a"], run_fn=fake_run)
    assert r.ok and calls == list(names)
    assert r.value == pytest.approx([-1.0, 1.0], abs=0.01)
    assert r.conditions["best"]["controller"] == names[1]
    assert targets_io.evaluate(targets["BND-3a"], r.value).passed


# --------------------------------------------------------------------------------------------- independent cross-checks
def test_bank_lock_matches_the_engine_locked_steady_state(pps, cfgs):
    """Independent cross-check of the bank heat balance: mrm.sim.runner.solve_locked_steady_state (the engine's own
    static solve, written separately) gives the same oracle lock temperatures, heater powers, codes and
    relOMA as _bank_codes / StaticRing.lock on g26 at two ambients and two biases. The engine's survey heats each ring
    with its own line only and gives every ring the chip-input power of its line (its bus walk multiplies only the
    ring's own line, after that ring), so the replica is compared with lines='own', bus_filter=False."""
    runner = pytest.importorskip("mrm.sim.runner")
    if not hasattr(runner, "solve_locked_steady_state"):
        pytest.skip("engine solve_locked_steady_state not available")
    pp = pps["g26"]
    amb, bv = [25.0, 105.0], [1.05, 3.75]
    qs = runner.solve_locked_steady_state(pp, cfg=cfgs["g26"], ambient_c=amb, bias_v=bv, iterations=6, n_scan=1201)
    r = replicas.StaticRing(pp)
    for ia, ta in enumerate(amb):
        for iv, vb in enumerate(bv):
            bc = replicas._bank_codes(pp, 0, t_base=ta + K0, bias=vb, mode="oracle", lines="own", bus_filter=False)
            assert not np.any(bc["infeasible"])
            np.testing.assert_allclose(bc["t_lock_k"], qs.t_ring_k[ia, iv, 0], atol=0.02)
            np.testing.assert_allclose(bc["p_w"][0], qs.p_heater_w[ia, iv, 0], atol=2e-5)
            np.testing.assert_allclose(bc["codes"][0], qs.heater_code[ia, iv, 0], rtol=0.01)
            lp = r.lock(mode="oracle", t_base=ta + K0, p_in_w=r.p_line_w, bias_v=vb, modulate=True)
            assert 10 * np.log10(lp.reloma) == pytest.approx(qs.reloma_db[ia, iv, 0, 0], abs=0.02)


def test_dynamic_lock_request_is_per_ring(pps):
    """The closed-loop request locks every ring of the bank at its OWN set-point, code and warm state (the bank heat
    balance with crosstalk), not ring 0's for all lanes."""
    pp = pps["g26"]
    seen = {}

    def fake_run(req):
        seen["req"] = req
        t = np.linspace(0.0, req.duration_s, 11)
        return {"t_s": t, "monitor_a": np.broadcast_to(req.setpoint_a, (t.size,) + req.setpoint_a.shape).copy(),
                "heater_code": np.broadcast_to(req.initial_code, (t.size,) + req.initial_code.shape).astype(float)}
    tr = replicas._dynamic_lock(None, pp, "half_max", bias=1.05, t_base=298.15, run_fn=fake_run, duration_s=1e-4)
    req = seen["req"]
    N = pp.n_rings
    assert req.setpoint_a.shape == (1, N) and req.initial_code.shape == (1, N)
    bc = replicas._bank_codes(pp, 0, t_base=298.15, bias=1.05, mode="half_max")
    np.testing.assert_array_equal(req.initial_code[0], np.rint(bc["codes"][0]).astype(np.int64))
    np.testing.assert_allclose(req.options["warm_state"]["p_heater_w"][0], bc["p_w"][0], rtol=1e-12)
    for n in range(N):
        rg = replicas.StaticRing(pp, n=n)
        lp = bc["locks"][n]
        assert lp.p_drop_other_w > 0 and lp.p_bus_w == pytest.approx(float(np.sum(bc["p_lines_at_ring_w"][n])))
        assert req.setpoint_a[0, n] == pytest.approx(float(rg.monitor_current_a(lp.p_drop_w, lp.p_bus_w,
                                                                                bc["t_pd_k"][0, n])[0]), rel=1e-9)
    # ring 0 is first on the bus: its lock is the stand-alone all-line lock (nothing filters its lines)
    r0 = replicas.StaticRing(pp)
    l0 = r0.lock(mode="half_max", t_base=298.15, p_in_w=r0.p_line_w, bias_v=1.05, modulate=True, lines="all")
    assert bc["locks"][0].p_drop_w == pytest.approx(l0.p_drop_w, rel=1e-12)
    # downstream rings see their lines filtered by the rings before them
    assert np.all(bc["p_lines_at_ring_w"][-1] < bc["p_lines_at_ring_w"][0])
    assert len(set(np.round(req.initial_code[0]).tolist())) > 1        # the lanes differ (crosstalk, lines)
    assert tr["setpoint_a"].shape == (1, N)


def test_val5_reports_the_hysteresis_of_the_loop_open_line(pps, targets):
    """At 0 dBm the g26 fixed-code line is bistable (the up-sweep jumps over the cold flank); the replica must say so,
    even when the 5 degC sampled offsets of both sweep directions coincide. At P_in -> 0 it is single-valued."""
    pp = pps["g26"]
    v5 = replicas.relOMA_map(pp, targets["VAL-5"])
    hy = v5.detail["primary"]["hysteresis_c"]
    assert hy is not None and hy[0] < 69.6 and v5.detail["primary"]["max_branch_gap_k"] > 1.0
    assert any("hysteretic" in n for n in v5.notes)
    assert not any("no hysteresis" in n for n in v5.notes)
    cold = replicas.relOMA_map(pp, targets["VAL-5"], p_in_dbm=None, with_variants=False)
    assert cold.conditions["p_in_w"] == 0.0
    assert cold.detail["primary"]["hysteresis_c"] is None
    # self-heating steepens the cold side: at P_in -> 0 the line is wider below the peak than at 0 dBm
    assert cold.value["below_peak_k"][0] > v5.value["below_peak_k"][0]


def test_power_convention_and_pd_temperature(pps, targets):
    """p_in_dbm=None means P_in -> 0 everywhere ("preset" is the line power); the monitor's dark current is taken at
    the PD temperature of the locked state, which the ring's own heat raises above the base; with every line of the
    bank on (the CAL-8 primary) the other lines' leakage adds to the drop current."""
    pp = pps["g26"]
    r = replicas.StaticRing(pp)
    m0 = replicas.monitor_current_at_lock(pp, targets["CAL-8"], p_in_dbm=None, setpoint="half_max")
    assert m0.conditions["p_in_w"] == 0.0 and m0.value < 0.1              # dark current only (nA)
    mp = replicas.monitor_current_at_lock(pp, targets["CAL-8"], setpoint="half_max", lines="own")
    assert mp.conditions["p_in_w"] == pytest.approx(r.p_line_w)
    lp = r.lock(mode="half_max", t_base=298.15, p_in_w=r.p_line_w, bias_v=1.05, modulate=True)
    assert lp.t_pd_k > 298.15 + 1.0
    i_pd = float(r.monitor_current_a(lp.p_drop_w, r.p_line_w, lp.t_pd_k)[0])
    assert mp.value == pytest.approx(i_pd * 1e6, rel=1e-9)
    assert i_pd > float(r.monitor_current_a(lp.p_drop_w, r.p_line_w, None)[0])     # warmer PD, more dark current
    ma = replicas.monitor_current_at_lock(pp, targets["CAL-8"], setpoint="half_max")
    assert ma.conditions["lines"] == "all" and ma.variants["lines=own"] == pytest.approx(mp.value, rel=1e-12)
    assert ma.value > mp.value + 0.3                                          # leakage of the neighbour lines
    hot = mp.detail["half_max"]["current_ua_at_base_105c"]
    assert hot > mp.value - 1.0                                               # set-point lock: about the same
    b0 = replicas.bistability_sweep(pp, p_in_dbm=None)
    assert b0.conditions["p_in_w"] == 0.0 and not b0.value["bistable"]
    rl = replicas.relOMA_map(pp, targets["CAL-14"], p_in_dbm="preset", with_variants=False)
    assert rl.conditions["p_in_w"] == pytest.approx(r.p_line_w)


def test_heater_compliance_is_applied(pps):
    """The driver's output compliance (its voltage limit) limits the static heater power as in the plant's
    heater_power_row, and the bank/map feasibility checks flag a demand above it."""
    r = replicas.StaticRing(pps["g26"], driver_law="current")
    t = r.t_ref
    p_free = float(r.power_from_u([1.0], t)[0])
    r.v_comp = 0.8 * math.sqrt(p_free * r._r_heater(t + r.z["hh"] * p_free))
    p_lim = float(r.power_from_u([1.0], t)[0])
    assert p_lim < p_free
    t_h = t + r.z["hh"] * p_lim
    assert math.sqrt(p_lim * r._r_heater(t_h)) == pytest.approx(r.v_comp, rel=1e-6)
    assert bool(r.compliance_ok(0.9 * p_lim, t_h)) and not bool(r.compliance_ok(1.2 * p_lim, t_h))
    assert replicas.StaticRing(pps["g26"], driver_law="bench_voltage").v_comp == math.inf


def test_val8e_reads_the_extraction_share_key(targets, monkeypatch):
    """thermal_val8_imec.json writes the two-pole fit as {tau_us, share}; the replica maps it onto the target's keys
    (and blocks, never NaN-passes, when the pair is missing)."""
    art = {"comparison": {"VAL-8e": {"predicted": {"tau_us": [3.0, 14.0], "share": [0.4, 0.6]}}}}
    monkeypatch.setattr(replicas, "_load_artefact", lambda name: art if name == "thermal_val8_imec.json" else None)
    r = replicas.thermal_step_published_stack(None, targets["VAL-8e"])
    assert r.ok and r.value == {"tau1_us": 3.0, "tau2_us": 14.0, "share1_frac": 0.4, "share2_frac": 0.6}
    targets_io.evaluate(targets["VAL-8e"], r.value)
    art["comparison"]["VAL-8e"]["predicted"] = {"tau_us": [3.0, 14.0]}
    assert replicas.thermal_step_published_stack(None, targets["VAL-8e"]).status == "blocked"


def test_nonfinite_predictions_are_marked(targets, monkeypatch):
    """A NaN prediction is written as null in the JSON file; the record says so and evaluate() fails it."""
    def nan_replica(plant, target=None, **kw):
        return replicas.ReplicaResult("loaded_q_from_3db_width", math.nan, "1")
    monkeypatch.setitem(replicas.REPLICAS, "loaded_q_from_3db_width", nan_replica)
    rec = replicas.predict_all(ids=["CAL-1"])[0]
    assert rec["replica_status"] == "ok" and rec["predicted"] is None and rec["predicted_nonfinite"] is True
    assert rec["evaluation"]["level"] == "fail"


def test_other_lines_match_the_control_models_static_point(pps):
    """Independent cross-check of the wrong-line terms and the bus filtering: mrm.control.models.static_point (the
    controllers' bank optics with every line, written separately) gives the same line power at each ring and the
    same absorbed power at the same ring temperatures as bus_line_powers + StaticRing(lines='all'), and about the
    same oracle heater codes as the bank solve; the own-line-only heat misses ~9 % of it in g26 (the other order
    2.6 FWHM from line n+4, BND-1). Line and absorbed powers agree to rounding (measured <= 1.3e-14): both walk the
    bus NRZ-averaged with the bus segment loss. Until the physics_optics fix static_point left the segment loss out,
    and the residual of up to 0.6 % (ring 7: 0.99916^7) was that loss, hidden by a 1 % tolerance. The codes compare
    two operating points (the bank's oracle lock against 107 pm on every ring), hence the looser bound."""
    models = pytest.importorskip("mrm.control.models")
    pp = pps["g26"]
    if not all(hasattr(models, f) for f in ("conditions_from_config", "static_point")):
        pytest.skip("control models API not available")
    cond = models.conditions_from_config(pp, 0)
    sp = models.static_point(pp, 0, np.full(pp.n_rings, 107e-12), cond=cond)
    pl = replicas.bus_line_powers(pp, 0, sp.t_ring_k, v_lo=cond.v_lo, v_hi=cond.v_hi)
    assert pl[0] == pytest.approx(replicas.StaticRing(pp).p_lines_w)                   # ring 0 is first on the bus
    gain = np.zeros(pp.n_rings)
    for n in range(pp.n_rings):
        r = replicas.StaticRing(pp, n=n)
        v_lo, v_hi = float(cond.v_lo[n]), float(cond.v_hi[n])
        t = np.array([float(sp.t_ring_k[n])])
        p_own = float(pl[n, r.line])
        assert p_own == pytest.approx(float(sp.p_in_w[n]), rel=1e-9)
        own = r.state_grid(r.lam_line, t, t_base=cond.t_base_k, p_in_w=p_own, v_lo=v_lo, v_hi=v_hi)
        allx = r.state_grid(r.lam_line, t, t_base=cond.t_base_k, p_in_w=p_own, v_lo=v_lo, v_hi=v_hi,
                            lines="all", p_lines_w=pl[n])
        assert float(allx["p_abs"][0, 0]) == pytest.approx(float(sp.p_abs_w[n]), rel=1e-9)
        gain[n] = float(allx["p_abs"][0, 0]) / float(own["p_abs"][0, 0])
        assert float(allx["drop_other_w"][0]) > 0.0
    # measured 1.092 1.091 1.088 1.069 1.033 1.038 1.036 1.029 for rings 0-7 with the landed
    # modes_rib160; the last rings on the bus see their foreign line after the ring that owns it has dropped part of
    # it, so the per-ring floor is 2 % and the ~9 % of the docstring holds for the first rings
    assert np.all(gain > 1.02), gain
    assert gain.max() > 1.08, gain
    bc = replicas._bank_codes(pp, 0, t_base=cond.t_base_k, bias=0.5 * float(cond.v_lo[0] + cond.v_hi[0]), mode="oracle")
    np.testing.assert_allclose(bc["codes"][0], sp.heater_code, rtol=0.005)
