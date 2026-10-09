"""Scenario runs of the controllers_a baselines on the ReferencePlant (slow: tens of seconds per run). Metrics come from
Truth inside the test only (CTRL_A_METRIC lines; run with -s -m slow).

- A1 at its printed peak rate (2127 degC/s, 183 pm/ms of resonance target at the victim ring 0) with a 20 K swing so
  the cusp arrives within 15 ms, and the full printed A1 (88.5 K, 130.7 ms) on the 25 degC base (the aggressor heats
  the monitor PD so much that its dark current moves the lock: measured);
- A2 (808 degC/s run, symmetric-triangle reading: 1616 degC/s edges) with a 20 K swing;
- a -1 dB laser power step for the normalized locks (fraction-of-peak PI against the inflection dither lock);
- the PI in two sampled worlds (uncertainty and variation drawn).
"""
from __future__ import annotations

import numpy as np
import pytest

import test_ctrl_a_harness as H
from mrm.control.baselines.bias_dither import BiasDither
from mrm.control.baselines.dither_lockin import DitherLockin
from mrm.control.baselines.extremum_seeking import ExtremumSeekingSine
from mrm.control.baselines.feedforward import FeedForwardPI
from mrm.control.baselines.midranging import MidRanging
from mrm.control.baselines.pi_lock import ST_LOCK, PiLock, PiLockCompiled

pytestmark = pytest.mark.slow
KNOW = ["controller_interface.knowledge=[line_grid, ring_to_line, heater_law, heater_efficiency, dlambda_dt_nominal]"]


def report(tag, m: H.LockMetrics, **extra):
    s = m.summary()
    s.update(extra)
    print("CTRL_A_METRIC", tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


def victim(m: H.LockMetrics, ring: int = 0) -> H.LockMetrics:
    """The aggressor's victim ring only (the others see only the crosstalk)."""
    return H.LockMetrics(*(np.asarray(getattr(m, f))[..., ring:ring + 1] for f in m.__dataclass_fields__))


def locked_plant(*, bias_v=None, knowledge=False, settle_ticks=5500):
    """A plant locked by the PI (MAP + lock) and its factory table (peak, slope)."""
    cfg, pp, spec, plant = H.placed(bias_v=bias_v)
    if knowledge:
        spec = H.make_spec(H.load_cfg(KNOW + (list(H.BIAS_CONTROLLER) if bias_v else [])), 1)
    pi = PiLock()
    H.ClosedLoop(plant, spec, pi).run(settle_ticks)
    tel = pi.telemetry()
    assert np.all(tel["state"] == ST_LOCK)
    return cfg, pp, spec, plant, tel


CONTROLLERS = {
    "pi_lock": lambda tel, plant: PiLock(start="lock", peak_codes=tel["peak_codes"], slope_per_code=tel["slope_per_code"],
                                        init_codes=plant.lat_heater.copy()),
    "pi_lock_type2": lambda tel, plant: PiLock(start="lock", peak_codes=tel["peak_codes"], k2=0.005,
                                              slope_per_code=tel["slope_per_code"], init_codes=plant.lat_heater.copy()),
    "pi_lock_c_type2": lambda tel, plant: PiLockCompiled(start="lock", peak_codes=tel["peak_codes"][0], k2=0.005,
                                                        slope_per_code=tel["slope_per_code"][0],
                                                        init_codes=plant.lat_heater[0].copy()),
    "dither_lockin": lambda tel, plant: DitherLockin(init_codes=plant.lat_heater.copy()),
    "feedforward_pi": lambda tel, plant: FeedForwardPI(start="lock", peak_codes=tel["peak_codes"], k2=0.005,
                                                      slope_per_code=tel["slope_per_code"],
                                                      init_codes=plant.lat_heater.copy()),
    "es_sine": lambda tel, plant: ExtremumSeekingSine(init_codes=plant.lat_heater.copy()),
}


@pytest.mark.parametrize("which", ["a1_rate", "a2_rate"])
@pytest.mark.parametrize("name", sorted(CONTROLLERS))
def test_aggressor_at_the_printed_rates(name, which):
    knowledge = name == "feedforward_pi"
    cfg, pp, spec, plant, tel = locked_plant(knowledge=knowledge)
    ctrl = CONTROLLERS[name](tel, plant)
    loop = H.ClosedLoop(plant, spec, ctrl)
    pre = 2000 if name in ("dither_lockin", "es_sine") else 300     # hand-off transient of the slower locks
    loop.run(pre)
    t_min = (plant.k_next + 1000) * plant.dt
    if which == "a1_rate":
        sc, period = H.a1_rate_scenario(cfg, pp, t_min)
        n = int((period / 2 + 0.004) / plant.dt) + 1000
    else:
        sc, period = H.a2_rate_scenario(cfg, pp, t_min)
        n = int((period * 0.75) / plant.dt) + 1000
    plant.set_scenario(sc)
    rec = loop.run(n)
    m = victim(H.lock_metrics(rec, slice(0, 900), slice(1000, None)))
    ct = ctrl.telemetry()
    report(f"{name} {which} (victim ring 0)", m, falls=float(np.nansum(ct.get("fall_count", np.zeros(1)))),
           heater_min_code=float(rec["heater"][:, 0, 0].min()))
    if name.startswith("pi_lock") or name == "feedforward_pi":
        assert np.all(m.max_err_pm < 15.0), m.max_err_pm


def test_a1_full_on_the_25c_base_monitor_dark_current():
    """The printed trajectory on the printed 25 degC base. Measured: the heater does not saturate (it bottoms out
    near 22 k codes), but in the interim layout the aggressor heats the victim's monitor PD ~6x more than the ring
    (205 K vs 34 K per 100 mW at steady state: the PD site sits nearer the aggressor), so the PD dark current
    (Arrhenius, 0.37 eV) grows to microamps near the top. The fraction-of-peak set-point then corresponds to a
    larger detuning: the lock drifts red from ~25 ms and is lost near the cusp (~52 ms). After the loss the PI
    integrates the dark offset upward (one-sided actuator runaway, 04 C3) and the neighbours fall too (probably the
    array supply limit once ring 0 sits at full scale; not verified). Reported, not asserted beyond the first 20 ms."""
    cfg, pp, spec, plant = H.placed()
    pi = PiLock(k2=0.005)
    loop = H.ClosedLoop(plant, spec, pi)
    loop.run(5500)
    t_min = (plant.k_next + 1000) * plant.dt
    sc, period = H.a1_rate_scenario(cfg, pp, t_min, t_base_c=25.0, full=True)
    plant.set_scenario(sc)
    rec = loop.run(int((period / 2 + 0.005) / plant.dt) + 1000)
    h = rec["heater"][:, 0, 0]
    early = victim(H.lock_metrics(rec, slice(0, 900), slice(1000, 21000)))
    assert np.all(early.max_err_pm < 10.0), early.max_err_pm
    m = victim(H.lock_metrics(rec, slice(0, 900), slice(1000, None)))
    d = rec["delta_pm"][:, 0, 0].astype(float)
    lost = np.flatnonzero(np.abs(d - m.delta_ref_pm[0, 0]) > 50.0)
    report("pi_lock_type2 A1 full (88.5 K, 25 degC base, victim ring 0)", m, heater_min_code=float(h.min()),
           first_tick_err_above_50pm=float(lost[0] if lost.size else -1),
           neighbour_falls=float(pi.telemetry()["fall_count"][0, 1:].sum()))


@pytest.mark.parametrize("name", ["pi_lock", "dither_lockin"])
def test_power_step_normalized_locks(name):
    """-1 dB on every line: the fraction-of-peak set-point moves along the line (04 C3), the inflection (2f) lock
    should not (its zero does not depend on the power)."""
    cfg, pp, spec, plant, tel = locked_plant()
    ctrl = CONTROLLERS[name](tel, plant)
    loop = H.ClosedLoop(plant, spec, ctrl)
    loop.run(6000 if name == "dither_lockin" else 500)
    t_s = (plant.k_next + 500) * plant.dt
    plant.set_scenario(H.laser_power_step_scenario(cfg, t_s, -1.0))
    rec = loop.run(8000)
    d = rec["delta_pm"].astype(float)
    shift = np.median(d[-2000:], axis=0) - np.median(d[:480], axis=0)
    m = H.lock_metrics(rec, slice(0, 480), slice(500, None))
    report(f"{name} laser power step -1 dB", m, lock_shift_pm=float(np.mean(shift)))


def test_bias_controllers_on_a1_rate():
    for name in ("midranging", "bias_dither"):
        cfg, pp, spec, plant, tel = locked_plant(bias_v=1.6 if name == "midranging" else 1.8)
        if name == "midranging":
            ctrl = MidRanging(start="lock", peak_codes=tel["peak_codes"], slope_per_code=tel["slope_per_code"],
                              init_codes=plant.lat_heater.copy())
        else:
            H.use_fast_monitor(plant)
            ctrl = BiasDither()
        loop = H.ClosedLoop(plant, spec, ctrl)
        loop.run(3000)
        t_min = (plant.k_next + 1000) * plant.dt
        sc, period = H.a1_rate_scenario(cfg, pp, t_min)
        plant.set_scenario(sc)
        rec = loop.run(int((period / 2 + 0.004) / plant.dt) + 1000)
        m = victim(H.lock_metrics(rec, slice(0, 900), slice(1000, None)))
        report(f"{name} a1_rate (victim ring 0)", m)
        if name == "midranging":
            # with the saturation fallback (review): 9.7 pm max; it lost lock before (470 pm)
            assert np.all(m.max_err_pm < 20.0), m.max_err_pm


def test_pi_in_sampled_worlds():
    """Two sampled worlds (07 4.8 uncertainty entries and per-ring variation drawn): MAP and lock without Truth.

    Rings the test's own placement cannot reach (their cold resonance already sits red of the declared line, so a
    heater cannot reach that order: an acquisition / FSR-assignment matter, 07 P1-12) are excluded and counted. In
    world 0 (er_mechanism heater_t_loss, current drive, monitor gain x2) the monitor keeps rising past the resonance
    (the loss falls with the heater power), so the MAP of some rings runs to the fold or the ADC clip: slower."""
    cfg, pp, spec, plant = H.placed(batch=2, nominal=False, seeds=3, n_ticks=3000)
    placed_ok = np.abs(plant.delta_pm - 170.0) < 20.0
    pi = PiLock()
    loop = H.ClosedLoop(plant, spec, pi)
    rec = loop.run(18000)
    tel = pi.telemetry()
    assert np.all(tel["state"][placed_ok] == ST_LOCK), (tel["state"], placed_ok)
    m = H.lock_metrics(rec, slice(-1500, -1000), slice(-1000, None))
    assert np.all(m.max_err_pm[placed_ok] < 2.0)
    clipped = int(np.sum(placed_ok & (tel["peak_codes"] >= spec.adc.full_scale_codes)))
    sel = lambda a: np.where(placed_ok, a, np.nan)                      # noqa: E731
    m_ok = H.LockMetrics(*(sel(np.asarray(getattr(m, f), float)) for f in m.__dataclass_fields__))
    report("pi_lock steady, 2 sampled worlds", m_ok, time_to_lock_us=float(np.nanmax(sel(H.time_to_lock_us(rec)))),
           lock_delta_spread_pm=float(np.nanmax(m_ok.delta_ref_pm) - np.nanmin(m_ok.delta_ref_pm)),
           rings_unplaceable=int(np.sum(~placed_ok)), rings_peak_clipped=clipped)
