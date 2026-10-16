"""The engine's sensing path against its own contract (physics_sensing review, 2026-10-16).

Each test pins one confirmed finding of that review:
- physics_sensing-1: the bit-statistics window takes the scenario's ones density, so under prbs7/9/15/23 it is the
  exact window of the last bits sent (07 P0-11; docs/handoff/plant_sensing.md 2.8), at every edge;
- physics_sensing-2: the ADC reference sits at ``ring.monitor.adc_ref_site`` (default the package base), not at each
  ring's PD observer 25 um from a ~133 degC ring;
- physics_sensing-3: a config with ``ring.monitor.reference_tap.present`` is refused, not run without the tap;
- physics_sensing-5: a per-world analog baseline stays per world when the batch equals the ring count;
- physics_sensing-6: rings that take turns on one integrating converter cannot overlap their windows, and share
  the converter's DNL thresholds and INL bow;
- physics_sensing-4 (meaning): background_fraction multiplies the total bus power at the ring.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from mrm import params
from mrm.control.api import ControllerSpec
from mrm.core import rng as _rng
from mrm.params.realize import RealizeError
from mrm.plant import sensing as se
from mrm.plant.datapath import _SLOT_BS_N1, prbs_table, prbs_window_ones
from mrm.plant.plant import Plant
from mrm.scenarios import library as lib
from mrm.scenarios import registry as scen_registry
from mrm.sim import kernel as K
from mrm.sim import runner
from mrm.sim.runner import _run_python


def _load(preset="g26", overrides=()):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load(preset, overrides=overrides if isinstance(overrides, dict) else list(overrides))


def _plant(cfg, B=2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = params.realize(cfg, B, seeds=7)
    sc = scen_registry.from_config(cfg)
    spec = ControllerSpec.from_config(cfg, n_batch=B, nominal=cfg, scenario_drives=sc.drives,
                                      scenario_offsets=sc.offsets)
    return Plant(pp, sc, spec, noise_seed=3, scenario_seed=5), spec


def _edges(plant, spec, n_ticks):
    """Run a do-nothing Python controller; at every edge record (tick, bs_n1, bit position) as the kernel left them
    after the release (the window just drawn, and the position it ends at)."""
    seen = []

    class Spy:
        name = "spy"
        latency_ticks = 1

        def reset(self, spec):
            pass

        def step(self, obs):
            seen.append((obs.tick, plant.S.ri[:, K.RI_BS_N1].copy(), plant.S.ri[:, K.RI_DP_BITPOS].copy()))
            return {}

    _run_python(plant, Spy(), spec, n_ticks, None)
    return seen


# ------------------------------------------------------------------------------------- physics_sensing-1
@pytest.mark.parametrize("mode", ["prbs7", "prbs9"])
def test_bitstat_window_is_the_exact_prbs_window_at_every_edge(mode):
    """Under a tabulated PRBS at the normal-traffic density 0.5 the window's ones count is the exact count of the 64
    bits that end at the lane's bit position (prbs_window_ones), at every edge including the first. Before the fix
    the kernel passed the realized per-tick density n1/nb (0.504 for prbs7, never exactly 0.5), so the window fell
    back to Binomial(64, n1/nb) and matched only ~8 % of the edges; the first edge read n1 = 0."""
    cfg = _load(overrides=[f"fidelity.pattern_noise={mode}"])
    plant, spec = _plant(cfg)
    seen = _edges(plant, spec, 300)
    assert len(seen) >= 30 and seen[0][0] == 0
    words, pre, period = prbs_table(int(mode[4:]))
    win = int(plant.P.dp_win_bits)
    for tick, n1, pos in seen:
        want = np.array([[prbs_window_ones(words, pre, period, (int(p) - win) % period, win) for p in row]
                         for row in pos])
        np.testing.assert_array_equal(n1, want, err_msg=f"edge {tick}")
    # the window is not degenerate: it moves with the sequence
    assert len({int(x) for _, n1, _ in seen for x in n1.ravel()}) > 5


def test_bitstat_window_draws_at_the_scenario_density_in_binomial_mode():
    """Binomial pattern noise: the window's ones are Binomial(64, data.ones_density) keyed by the edge, the exact
    marginal of i.i.d. bits; not Binomial(64, n1/nb) of the tick that preceded the edge."""
    cfg = _load(overrides=["fidelity.pattern_noise=binomial"])
    plant, spec = _plant(cfg)
    seen = _edges(plant, spec, 200)
    win = int(plant.P.dp_win_bits)
    keys = plant.P.dp_key_dp
    for tick, n1, _ in seen:
        want = np.array([[_rng.binomial(keys[b], n, tick, _SLOT_BS_N1, win, 0.5) for n in range(plant.N)]
                         for b in range(plant.B)])
        np.testing.assert_array_equal(n1, want, err_msg=f"edge {tick}")


# ------------------------------------------------------------------------------------- physics_sensing-2
def _quiet_run(site=None, ambient_c=None):
    """teaching_basic (every noise source off, ideal converters) at the settled quasi-static warm start, 200 us."""
    ovr = [] if site is None else [f"ring.monitor.adc_ref_site={site}"]
    cfg = _load("teaching_basic", ovr)
    sc = lib.static(ambient_c=ambient_c) if ambient_c is not None else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return runner.run(cfg, controller=None, scenario=sc, duration_s=2e-4, batch=1, seeds=0,
                          uncertainty="nominal", variation="off", heater_code="quasi_static", settle=True)


@pytest.mark.parametrize("ambient_c", [25.0, 60.0])
def test_adc_reference_sits_at_the_package_base_by_default(ambient_c):
    """g26 (side_by_side: the converter on its own die beside the PIC; here teaching_basic, g26 without noise) puts
    the ADC reference at the package base temperature: a noise-free code is the ideal transfer at T_base to rounding,
    and not the transfer at the ring's PD observer (13-21 K hotter at the lock). Before the fix every ring's reference
    sat at its own PD (-8 to -26 codes at the g26 lock, 20 ppm/K)."""
    r = _quiet_run(ambient_c=ambient_c)
    hw = se.MonitorHw.from_plant_params(r.plant.pp)
    assert hw.adc_ref_site == se.ADC_REF_PACKAGE_BASE
    tf = r.truth_final
    t_base = float(tf["t_base_k"][0, 0])
    assert t_base == pytest.approx(273.15 + ambient_c, abs=1e-9)
    np.testing.assert_array_equal(r.plant.S.rf[:, K.RF_T_ADC], np.full((1, r.plant.N), t_base))
    assert np.all(tf["t_pd_k"] > t_base + 10.0)                      # the PD observers are far hotter
    code = tf["monitor_code"]
    ok = (code > 10) & (code < hw.code_max - 10)
    assert ok.sum() >= 4
    at_base = code - hw.ideal_code(tf["i_monitor_a"], 0, t_base)
    at_pd = code - hw.ideal_code(tf["i_monitor_a"], 0, tf["t_pd_k"])
    assert np.max(np.abs(at_base[ok])) <= 0.6
    assert np.max(np.abs(at_pd[ok])) > 5.0


def test_adc_reference_at_the_pd_is_the_monitor_pd_site():
    """ring.monitor.adc_ref_site = monitor_pd keeps the former model (a converter next to each PD)."""
    r = _quiet_run(site="monitor_pd")
    hw = se.MonitorHw.from_plant_params(r.plant.pp)
    assert hw.adc_ref_site == se.ADC_REF_MONITOR_PD
    tf = r.truth_final
    np.testing.assert_array_equal(r.plant.S.rf[:, K.RF_T_ADC], r.plant.S.rf[:, K.RF_T_PD])
    code = tf["monitor_code"]
    ok = (code > 10) & (code < hw.code_max - 10)
    at_pd = code - hw.ideal_code(tf["i_monitor_a"], 0, tf["t_pd_k"])
    assert np.max(np.abs(at_pd[ok])) <= 0.6


def test_python_bank_follows_the_adc_reference_site():
    """MonitorBank.tick / level_codes apply the same site rule as the kernel (MonitorHw.adc_temperature)."""
    kw = dict(responsivity_a_per_w=1.0, adc_full_scale_a=40e-6, shot_noise=False, adc_ref_tempco_per_k=1e-3)
    t_pd, t_base = 330.0, 310.0
    for site, t_want in (("package_base", t_base), ("monitor_pd", t_pd)):
        hw = se.MonitorHw.build(1, 2, 1e-6, adc_ref_site=site, **kw)
        np.testing.assert_array_equal(hw.adc_temperature(t_pd, t_base), np.full((1, 2), t_want))
        bank = se.MonitorBank(hw, [0], m_max=1, pop_period_ticks=4)
        for k in range(8):
            bank.tick(k, 20e-6, 0.0, t_pd, t_base_k=t_base)
        got = bank.release(8).samples[0, :, 0]
        np.testing.assert_array_equal(got, np.round(hw.ideal_code(20e-6, 0, t_want)[0]).astype(int))
        mu1, _, _, _ = bank.level_codes(20e-6, 5e-6, t_pd_k=t_pd, t_base_k=t_base)
        np.testing.assert_allclose(mu1[0], hw.ideal_code(20e-6, 0, t_want)[0], rtol=1e-12)
    # package_base without a base temperature: the reference at its trim temperature (no tempco error)
    hw = se.MonitorHw.build(1, 2, 1e-6, **kw)
    np.testing.assert_array_equal(hw.adc_temperature(t_pd), np.full((1, 2), se.ADC_REF_T_K_DEFAULT))
    with pytest.raises(ValueError, match="monitor_pd"):
        se.MonitorHw.build(1, 8, 1e-6, adc_ref_site="monitor_pd", channels_per_adc=2, adc_rate_hz=125e3, **kw)
    with pytest.raises(RealizeError, match="adc_ref_site"):
        params.realize(_load(overrides=["ring.monitor.adc_ref_site=monitor_pd",
                                        "ring.monitor.adc_mux.channels_per_adc=2",
                                        "ring.monitor.integration_window_s=1e-6"]), 1, seeds=0)


# ------------------------------------------------------------------------------------- physics_sensing-3
def test_reference_tap_config_is_refused():
    """The engine has no reference-tap channels and the optics do not take the tap's share of the bus power, so a
    config that enables the tap must not run as if it had one (before the fix it ran silently without it)."""
    cfg = _load(overrides={"ring.monitor.reference_tap.present": True,
                           "ring.monitor.reference_tap.coupling_frac": 0.01})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = params.realize(cfg, 1, seeds=0)
    assert pp.monitor.ref_tap_present
    with pytest.raises(NotImplementedError, match="reference_tap"):
        se.MonitorHw.from_plant_params(pp)
    with pytest.raises(NotImplementedError, match="reference_tap"):
        _plant(cfg, B=1)


# ------------------------------------------------------------------------------------- physics_sensing-5
@pytest.mark.parametrize("B", [4, 8])
def test_per_world_monitor_baseline_stays_per_world(B):
    """MonitorParams.baseline_a is (B, 1): one analog baseline per world. With B == N == 8 the adapter used to
    flatten it to (B,), which build() reads as per ring, so world 0 got every world's baseline across its rings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = params.realize(_load(), B, seeds=0)
    assert pp.n_rings == 8
    bl = 1e-6 * np.arange(B, dtype=float).reshape(B, 1)
    pp = dataclasses.replace(pp, monitor=dataclasses.replace(pp.monitor, baseline_a=bl))
    hw = se.MonitorHw.from_plant_params(pp)
    np.testing.assert_array_equal(hw.pr[:, :, se.PR_IBASE], np.broadcast_to(bl, (B, 8)))


# ------------------------------------------------------------------------------------- physics_sensing-6
def _hw(N=8, **kw):
    base = dict(responsivity_a_per_w=1.0, adc_full_scale_a=40e-6, adc_bits=12, integration_window_s=1e-6,
                adc_rate_hz=250e3, adc_latency_s=2e-6, shot_noise=False)
    base.update(kw)
    return se.MonitorHw.build(1, N, 1e-6, **base)


def test_shared_converter_windows_cannot_overlap():
    """round_robin: C rings take turns on one integrating converter, so C windows must fit in one ADC period. The
    g26 timing (2 us window, 4 us period) with C = 4 was accepted before and had ring 0 and ring 1 integrating at
    once on converter 0."""
    with pytest.raises(ValueError, match="overlapping windows"):
        _hw(integration_window_s=2e-6, channels_per_adc=4)
    hw = _hw(integration_window_s=1e-6, channels_per_adc=4)             # 4 x 1 tick = the 4-tick period
    assert hw.phase.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    _hw(integration_window_s=2e-6, channels_per_adc=4, mux_order="bank")  # one front end per ring: allowed
    _hw(integration_window_s=2e-6, channels_per_adc=2)                   # 2 x 2 ticks = 4: allowed
    with pytest.raises(RealizeError, match="adc_mux"):
        params.realize(_load(overrides=["ring.monitor.adc_mux.channels_per_adc=4"]), 1, seeds=0)


def test_rings_on_one_converter_share_its_dnl_and_inl():
    """The DNL thresholds and the INL bow belong to the quantizer: rings 0-3 (converter 0) convert any value to the
    same code, rings 4-7 (converter 1) to their own. Before the fix every ring had its own thresholds and bow."""
    bow = se.draw_inl_bow_lsb([123], 8, 3.0, channels_per_adc=4)
    assert np.all(bow[0, :4] == bow[0, 0]) and np.all(bow[0, 4:] == bow[0, 4]) and bow[0, 0] != bow[0, 4]
    # keyed by the converter index: converter j draws what ring j draws without a mux (C = 1, one per ring)
    np.testing.assert_array_equal(se.draw_inl_bow_lsb([123], 8, 3.0)[0, :2], bow[0, ::4])
    with pytest.raises(ValueError, match="INL bow"):
        _hw(channels_per_adc=4, adc_inl_bow_lsb=np.arange(8.0))
    hw = _hw(channels_per_adc=4, adc_dnl_lsb=0.9, adc_inl_bow_lsb=bow)
    np.testing.assert_array_equal(hw.pr[0, :, se.PR_ADC_UNIT], [0, 0, 0, 0, 1, 1, 1, 1])
    lsb = hw.lsb_a[0]
    vals = (np.arange(60) * 0.37 + 500.3) * lsb                          # sweeps code boundaries
    codes = []
    for x in vals:
        bank = se.MonitorBank(hw, [0], device_seeds=[123], m_max=1, pop_period_ticks=4)
        for k in range(8):
            bank.tick(k, x, 0.0)
        codes.append(bank.release(8).samples[0, :, 0])
    codes = np.array(codes)
    for n in range(1, 4):
        np.testing.assert_array_equal(codes[:, n], codes[:, 0])
        np.testing.assert_array_equal(codes[:, 4 + n], codes[:, 4])
    assert np.any(codes[:, 0] != codes[:, 4])
    # the adapter keys them the same way
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = params.realize(_load(overrides=["ring.monitor.adc_mux.channels_per_adc=4",
                                             "ring.monitor.integration_window_s=1e-6"]), 2, seeds=3)
    hw = se.MonitorHw.from_plant_params(pp)
    np.testing.assert_array_equal(hw.pr[:, :, se.PR_ADC_UNIT], np.broadcast_to(np.arange(8) // 4, (2, 8)))
    b = hw.pr[:, :, se.PR_INL_BOW]
    assert np.all(b[:, :4] == b[:, :1]) and np.all(b[:, 4:] == b[:, 4:5])


# ------------------------------------------------------------------------------------- physics_sensing-4
def test_background_fraction_multiplies_the_total_bus_power():
    """i_mon gains gain_scale R f P_bus with P_bus the bus power entering ring n summed over every line (3-8 x the
    declared line's power in g26), which is what the schema doc now says."""
    cur = {}
    for f in (0.0, 1e-4):
        cfg = _load(overrides={"ring.monitor.background_fraction": f})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = runner.run(cfg, controller=None, duration_s=2e-5, batch=1, seeds=0, uncertainty="nominal",
                           variation="off", heater_code="quasi_static", settle=True)
        cur[f] = (r.truth_final["i_monitor_a"].copy(), r.plant.S.rf[:, K.RF_P_BG].copy(), r.truth_final["p_in_w"])
    pp = r.plant.pp
    gain = float(pp.monitor.gain_scale[0, 0]) * pp.monitor.responsivity_a_per_w
    d = cur[1e-4][0] - cur[0.0][0]
    np.testing.assert_allclose(d, gain * 1e-4 * cur[0.0][1], rtol=1e-9)
    assert np.all(cur[0.0][1] > 3.0 * cur[0.0][2])                   # the whole bus, not one line
