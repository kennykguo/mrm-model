"""TLC priors from the nominal G26 design (mrm/control/tlc/priors.py). Expected values come from the nominal model
and the preset (mrm.control.models, spec.nominal), not from the priors code itself."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import firwin, freqz

import mrm.params as P
from mrm.control import models as M
from mrm.control.api import ControllerSpec
from mrm.control.knowledge import KnowledgeError
from mrm.control.tlc.core import TLC
from mrm.control.tlc.priors import g26_priors
from mrm.control.tlc.tone_plan import collisions
from mrm.control.tlc.units import pm_per_ghz

TIER = "controller_interface.knowledge=[line_grid,ring_to_line,model_nominal]"


@pytest.fixture(scope="module")
def spec():
    cfg = P.load("g26", overrides=[TIER])
    return ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg)


@pytest.fixture(scope="module")
def priors(spec):
    return g26_priors(spec)


@pytest.fixture(scope="module")
def ppg(spec):
    return pm_per_ghz(np.asarray(spec.line_grid_hz)[np.asarray(spec.ring_to_line)])


def test_needs_model_nominal():
    cfg = P.load("g26")
    with pytest.raises(KnowledgeError):
        g26_priors(ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg))


def test_kappa_is_the_nominal_tuning_slope_at_the_operating_temperature(spec, priors, ppg):
    """s(T) = s0 (1 + beta (T - T0)) (thermo-optic order 2) at the nominal lock temperature, in GHz/K."""
    th = M.nominal_plant_params(spec.nominal.model_config()).thermal
    s0, beta, t0 = (float(np.ravel(v)[0]) for v in (th.dlambda_dt_s0_m_per_k, th.dlambda_dt_beta_per_k,
                                                     th.dlambda_dt_t0_k))
    t_lock = float(np.ravel(th.t_ring_lock_k)[0])
    want_pm_per_k = s0 * (1 + beta * (t_lock - t0)) * 1e12
    assert priors.pp.kappa * np.mean(ppg) == pytest.approx(want_pm_per_k, rel=0.02)


def test_linewidth_is_the_loaded_q_width_plus_nrz_broadening(spec, priors):
    cfg = spec.nominal.model_config()
    fwhm_ghz = np.asarray(M.nominal_plant_params(cfg).optics.fwhm_hz, float).ravel() / 1e9
    ratio = 2 * np.asarray(priors.pp.hwhm) / fwhm_ghz
    assert np.all((ratio >= 1.0) & (ratio <= 1.25)), ratio
    assert max(priors.derivation["lineshape_fit_rms_rel"]) < 0.02


def test_heater_paths_sum_to_the_nominal_heater_to_ring_gain(spec, priors):
    pp_nom = M.nominal_plant_params(spec.nominal.model_config())
    z = float(np.ravel(pp_nom.thermal.z_heater_ring_k_per_w)[0])
    assert priors.pp.r_hf + priors.pp.r_hs == pytest.approx(z, rel=1e-9)
    assert priors.pp.tau_hf < 5e-6 < priors.pp.tau_hs < 50e-6                 # G26 poles: 3.2 us and 13 us


def test_core_dc_level_is_the_nominal_drop_peak_level(priors):
    pp = priors.pp
    p_core = np.asarray(pp.laser_power) * np.asarray(pp.depth) / pp.pd_full_scale_mw
    np.testing.assert_allclose(p_core, priors.derivation["drop_peak_level_fs"], rtol=0.04)


def test_crosstalk_matches_the_preset_fractions(spec, priors):
    near = float(spec.nominal.si("ring.thermal.xt_nearest_frac"))
    nxt = float(spec.nominal.si("ring.thermal.xt_next_frac"))
    assert priors.pp.coupling[0] == pytest.approx(near, rel=0.05)
    assert priors.pp.coupling[1] == pytest.approx(nxt, rel=0.05)
    assert 1e-4 < priors.pp.tau_x < 1e-2


def test_controller_params_for_eight_rings_at_250ks(spec, priors):
    cp = priors.cp
    assert cp.n_rings == spec.n_rings == 8
    assert collisions(tuple(int(f / 250) for f in cp.tone_freqs)) == []
    assert cp.decim == 1 and priors.pp.pd_rate == spec.adc.rate_hz
    f = np.r_[cp.tone_freqs, 2 * np.asarray(cp.tone_freqs)]
    _, h = freqz(firwin(cp.fir_taps, cp.fir_cutoff, fs=priors.pp.pd_rate), worN=f, fs=priors.pp.pd_rate)
    assert np.all(np.abs(np.abs(h) - 1) < 1e-3)


def test_nrz_swing_is_tens_of_pm(priors, ppg):
    assert 20 < priors.pp.mod_shift * np.mean(ppg) < 50                     # G26 nominal: about 33 pm


def test_r13_er_floor_is_dropped_only_because_b6_finds_no_setpoint(priors):
    r13 = priors.derivation["r13_er_floor"]
    assert r13["source_db"] == 3.5
    assert max(r13["er_at_inflection_db"]) < 3.5 and r13["applied_db"] == 0.0 == priors.cp.er_min_db


def test_core_builds_and_steps_with_the_priors(priors):
    ctl = TLC(priors.pp, priors.cp)
    ctl.warm_start(np.full(8, 0.28), np.asarray(ctl.sp.targets))
    n_pd = int(round(priors.cp.ts * priors.pp.pd_rate))
    for k in range(5):
        seq = ctl.step(np.full((n_pd, 8), 0.7), k * priors.cp.ts)
    assert seq.shape == (10, 8) and np.all(np.isfinite(seq))


@pytest.mark.parametrize("corner", ["g26_fast_thermal", "g26_slow_thermal", "g26_textbook_thermal"])
def test_heater_paths_follow_the_corner_poles(corner):
    """The fast path takes the fastest sub-ms pole and the slow path the rest, whatever the time scales."""
    import yaml
    from pathlib import Path
    alts = yaml.safe_load((Path(P.__file__).parent / "presets" / f"{corner}.yaml").read_text()).get("alts", {})
    cfg = P.load("g26", overrides=[TIER], alts=alts)
    pr = g26_priors(ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg))
    pp_nom = M.nominal_plant_params(cfg)
    assert pr.pp.r_hf + pr.pp.r_hs == pytest.approx(float(np.ravel(pp_nom.thermal.z_heater_ring_k_per_w)[0]))
    assert 0 < pr.pp.tau_hf < pr.pp.tau_hs < 1e-3 and pr.pp.r_hf > 0 and pr.pp.r_hs > 0
