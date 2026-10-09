"""07 P0-8: streaming metric accumulators and record specs (mrm.sim.metrics; owner: engine).

- The accumulators updated every plant tick equal the metrics computed from the full trace (a 10 ms plant run and a
  synthetic run with every lock state transition), via the independent NumPy reference metrics_from_trace.
- A 5 us spike in a 1 s run decimated to 1 ms gives excursion_count == 1, and the min/max envelope keeps the spike
  that the 1 ms mean hides.
- Record reducers (mean | min | max | last) and multi-tick steps; histogram mass; BER windows.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from numba import njit

from mrm.sim.metrics import (AI_ABOVE0, MetricsSpec, MetricsState, RecordSet, RecordSpec, TR, envelope,
                             metrics_from_trace, metrics_tick_row, record_tick_row, summarize)


def _state(B, N, spec=None, dt=1e-6, hist=True):
    spec = spec or MetricsSpec(lock_threshold_pm=5.0, hysteresis_pm=1.0, ber_sample_period_s=1e-3)
    st = MetricsState.build(spec, B, N, dt, fwhm_max_m=330e-12, fsr_max_m=10.3e-9, code_lo=0, code_hi=65535,
                            histogram=hist)
    return spec, st


def _feed(st, det, k0=0, **kw):
    """Run metrics_tick_row over (T, B, N) traces."""
    T, B, N = det.shape
    z = np.zeros((T, B, N))
    order = kw.get("order", np.zeros((T, B, N), np.int64))
    wrong = kw.get("wrong", np.zeros((T, B, N), bool))
    rel = kw.get("reloma", z)
    orc = kw.get("oracle", z)
    oru = kw.get("oracle_unr", z)
    ph = kw.get("p_heater_w", z)
    pa = kw.get("p_abs_w", z)
    reg = kw.get("reg", np.zeros((T, B, N), np.int64))
    pop = kw.get("popped", np.zeros((T, B, N), bool))
    ber = kw.get("ber", z)
    bits = kw.get("bits", z)
    ref = kw.get("lock_ref", z)
    st.reset(k0, kw.get("reg0", np.zeros((B, N), np.int64)))
    for t in range(T):
        for b in range(B):
            metrics_tick_row(np.int64(k0 + t), np.int64(1), st.mpf, st.mpi, det[t, b], ref[t, b], order[t, b],
                             wrong[t, b], rel[t, b], orc[t, b], oru[t, b], ph[t, b], pa[t, b], reg[t, b], pop[t, b],
                             ber[t, b], bits[t, b], st.af[b], st.ai[b], st.hist[b], st.runhist[b])
    return dict(order=order, wrong=wrong, reloma=rel, oracle=orc, oracle_unr=oru, p_heater_w=ph, p_abs_w=pa,
                reg=reg, popped=pop, ber=ber, bits=bits)


@njit(cache=True)
def _drive(det, f, mpf, mpi, zi, zb, z, af, ai, hist, runhist, tv, rec_i, acc, cnt, idx, trace):
    """Feed a scalar detuning trace through the accumulators and the records, one tick at a time."""
    d = np.zeros(1)
    for k in range(det.shape[0]):
        d[0] = det[k]
        metrics_tick_row(k, 1, mpf, mpi, d, z, zi, zb, z, z, z, z, z, zi, zb, z, z, af, ai, hist, runhist)
        tv[f, 0] = det[k]
        record_tick_row(1, tv, rec_i, acc, cnt, idx, trace)


def _compare(acc, ref, keys):
    for k in keys:
        a, r = acc[k], ref[k]
        if isinstance(r, dict):
            for kk in r:
                np.testing.assert_allclose(a[kk], r[kk], rtol=1e-12, atol=0, err_msg=f"{k}[{kk}]")
        else:
            np.testing.assert_allclose(np.asarray(a, float), np.asarray(r, float), rtol=1e-10, atol=1e-300,
                                       equal_nan=True, err_msg=k)


KEYS = ("max_abs_detuning_pm", "mean_detuning_pm", "rms_detuning_pm", "max_abs_lock_error_pm", "rms_lock_error_pm",
        "time_above_s", "excursion_count",
        "time_unlocked_s", "excursion_max_s", "first_lock_s", "time_to_lock_s", "locked_final", "fsr_hop_events",
        "wrong_line_s", "wrong_line_events", "wrong_line_final", "mean_reloma", "oma_penalty_db",
        "oma_penalty_unrestricted_db", "mean_p_abs_w", "expected_errors", "heater_energy_j", "heater_sat_s",
        "heater_low_rail_s", "code_min_used", "code_max_used", "heater_range_used_codes", "peak_slew_codes_per_s")


def test_accumulators_equal_trace_metrics_synthetic():
    """Every lock-state path (acquisition, excursions with hysteresis, relock on another order, ending unlocked),
    wrong-line entries, rails, slews: the streaming accumulators equal the full-trace reference exactly."""
    rng = np.random.default_rng(3)
    T, B, N = 4000, 2, 3
    spec, st = _state(B, N)
    t = np.arange(T)[:, None, None]
    det = 1e-12 * (40.0 * np.exp(-t / 300.0) * np.cos(t / 25.0 + np.arange(N))
                   + 4.5 * np.sin(t / 97.0 + 1.3 * np.arange(B)[:, None]) + rng.normal(0.0, 1.2, (T, B, N)))
    det[2500:2600, 0, 1] = 30e-12                      # a long excursion
    det[3990:, 1, 2] = 50e-12                          # ends unlocked
    order = np.where(t > 3000, 1, 0) * np.ones((T, B, N), np.int64)
    wrong = (np.sin(t / 50.0) > 0.95) & (np.arange(N) == 0)
    wrong = np.broadcast_to(wrong, (T, B, N)).copy()
    reg = np.clip((30000 + 20000 * np.sin(t / 400.0) * np.ones((T, B, N))).astype(np.int64), 0, 65535)
    reg[100:150] = 65535
    reg[200:220] = 0
    popped = np.broadcast_to((t % 10) == 0, (T, B, N)).copy()
    rel = rng.uniform(0.05, 0.3, (T, B, N))
    orc = rel * 1.2
    oru = rel * 1.3
    ph = rng.uniform(0.0, 0.03, (T, B, N))
    pa = rng.uniform(0.0, 2e-3, (T, B, N))
    ber = 10.0 ** rng.uniform(-14, -8, (T, B, N))
    bits = np.full((T, B, N), 50.0)
    reg0 = np.full((B, N), 1000, np.int64)
    kw = _feed(st, det, k0=7, order=order, wrong=wrong, reloma=rel, oracle=orc, oracle_unr=oru, p_heater_w=ph,
               p_abs_w=pa, reg=reg, popped=popped, ber=ber, bits=bits, reg0=reg0)
    acc = summarize(st)
    ref = metrics_from_trace(spec, 1e-6, det_m=det, code_lo=0, code_hi=65535, reg0=reg0, k0=7, **kw)
    _compare(acc, ref, KEYS)
    assert np.all(acc["excursion_count"] >= 1)
    assert acc["locked_final"][1, 2] == False         # noqa: E712
    assert np.isnan(acc["time_to_lock_s"][1, 2])
    assert np.any(acc["fsr_hop_events"] > 0)
    # the histogram holds every tick once
    assert np.all(st.hist.sum(axis=2) == T)


def test_lock_reference_moves_the_lock_metrics():
    """Lock error about a reference (MetricsSpec.lock_reference 'target' or 'oracle'): a ring held 150 pm red of its
    line (a side-of-fringe or OMA lock) never "locks" about the line, but locks about its operating point; the
    streaming accumulators equal the full-trace reference with a time-varying reference too."""
    rng = np.random.default_rng(5)
    T, B, N = 3000, 1, 2
    t = np.arange(T)[:, None, None]
    ref = np.where(t < 1500, 150e-12, 140e-12) * np.ones((T, B, N))     # an operating point that moves once
    det = ref + 1e-12 * (30.0 * np.exp(-t / 200.0) + rng.normal(0.0, 1.0, (T, B, N)))
    det[2200:2210, 0, 1] += 12e-12                                      # one excursion of ring 1
    spec_line, st_line = _state(B, N)
    _feed(st_line, det)
    m_line = summarize(st_line)
    assert not m_line["ever_locked"].any()
    spec, st = _state(B, N, MetricsSpec(lock_threshold_pm=5.0, hysteresis_pm=1.0, ber_sample_period_s=1e-3,
                                        lock_reference="target", lock_target_pm=150.0))
    assert np.allclose(st.ref, 150e-12)
    kw = _feed(st, det, lock_ref=ref)
    acc = summarize(st)
    assert acc["ever_locked"].all() and acc["locked_final"].all()
    assert int(acc["excursion_count"][0, 1]) >= 1
    np.testing.assert_allclose(acc["max_abs_detuning_pm"], m_line["max_abs_detuning_pm"])
    ref_m = metrics_from_trace(spec, 1e-6, det_m=det, code_lo=0, code_hi=65535, reg0=np.zeros((B, N), np.int64),
                               lock_ref_m=ref, **kw)
    _compare(acc, ref_m, KEYS)
    with pytest.raises(ValueError):
        MetricsSpec(lock_reference="target")
    with pytest.raises(ValueError):
        MetricsSpec(lock_reference="oracle", oracle=False)


def test_spike_survives_decimation():
    """07 P0-8: a 5 us spike in a 1 s run decimated to 1 ms gives excursion_count == 1; the 1 ms mean hides it,
    the min/max envelope keeps it."""
    T = 1_000_000
    spec, st = _state(1, 1, MetricsSpec(lock_threshold_pm=5.0, hysteresis_pm=1.0, ber_sample_period_s=2.0),
                      hist=False)
    det = np.full(T, 1.0e-12)
    det[500_000:500_005] = 20e-12                       # 5 us at 20 pm
    z = np.zeros(1)
    zi = np.zeros(1, np.int64)
    zb = np.zeros(1, bool)
    st.reset(0, np.zeros((1, 1), np.int64))
    rs = RecordSet.build([RecordSpec("detuning_m", 1000, "mean")] + envelope("detuning_m", 1000), 1, 1, T)
    tv = np.zeros((len(TR), 1))
    _drive(det, np.int64(TR["detuning_m"]), st.mpf, st.mpi, zi, zb, z, st.af[0], st.ai[0], st.hist[0],
           st.runhist[0], tv, rs.rec_i, rs.acc[0], rs.cnt[0], rs.idx[0], rs.trace[0])
    m = summarize(st)
    assert int(m["excursion_count"][0, 0]) == 1
    assert m["excursion_max_s"][0, 0] == pytest.approx(5e-6, abs=1e-12)
    tr = rs.traces(1e-6)
    mean = tr["detuning_m@1000:mean"]["value"][:, 0, 0]
    mx = tr["detuning_m@1000:max"]["value"][:, 0, 0]
    assert mean.size == 1000
    assert np.all(mean < 5e-12)                         # the decimated mean never crosses the 5 pm threshold
    assert mx.max() == pytest.approx(20e-12)            # the envelope keeps the spike
    assert int(np.argmax(mx)) == 500


def test_record_reducers_and_multitick_steps():
    """mean / min / max / last over windows, and a multi-tick step of h ticks counts h times (07 P0-10)."""
    rs = RecordSet.build([("t_ring_k", 4, "mean"), ("t_ring_k", 4, "min"), ("t_ring_k", 4, "max"),
                          ("t_ring_k", 4, "last")], 1, 2, 16)
    tv = np.zeros((len(TR), 2))
    vals = np.arange(16, dtype=float)
    k = 0
    for h in (1, 1, 2, 4, 1, 1, 2, 4):            # 16 ticks in steps that never straddle a window
        tv[TR["t_ring_k"]] = vals[k]
        record_tick_row(np.int64(h), tv, rs.rec_i, rs.acc[0], rs.cnt[0], rs.idx[0], rs.trace[0])
        k += h
    tr = rs.traces(1e-6)
    mean = tr["t_ring_k@4:mean"]["value"][:, 0, 0]
    np.testing.assert_allclose(mean, [(0 + 1 + 2 * 2) / 4, 4.0, (8 + 9 + 2 * 10) / 4, 12.0])
    np.testing.assert_array_equal(tr["t_ring_k@4:min"]["value"][:, 0, 0], [0, 4, 8, 12])
    np.testing.assert_array_equal(tr["t_ring_k@4:max"]["value"][:, 0, 0], [2, 4, 10, 12])
    np.testing.assert_array_equal(tr["t_ring_k@4:last"]["value"][:, 0, 0], [2, 4, 10, 12])
    np.testing.assert_allclose(tr["t_ring_k@4:mean"]["t_s"], [4e-6, 8e-6, 12e-6, 16e-6])
    with pytest.raises(ValueError):
        RecordSpec("not_a_field")
    with pytest.raises(ValueError):
        RecordSet.build([("t_ring_k", 4, "mean"), ("t_ring_k", 4, "mean")], 1, 1, 8)


def test_ber_windows_and_time_above():
    spec = MetricsSpec(lock_threshold_pm=5.0, hysteresis_pm=1.0, ber_sample_period_s=10e-6, ber_limit=1e-9,
                       time_above_pm=(5.0, 20.0))
    _, st = _state(1, 1, spec)
    T = 100
    det = np.zeros((T, 1, 1))
    det[10:15] = 30e-12
    ber = np.full((T, 1, 1), 1e-12)
    ber[40:50] = 1e-6                                   # one bad 10-tick window
    _feed(st, det, ber=ber, bits=np.full((T, 1, 1), 50.0))
    m = summarize(st)
    assert int(m["ber_windows"][0, 0]) == 10
    assert int(m["ber_windows_over_limit"][0, 0]) == 1
    assert m["max_window_ber"][0, 0] == pytest.approx(1e-6)
    assert m["time_above_s"]["5pm"][0, 0] == pytest.approx(5e-6)
    assert m["time_above_s"]["20pm"][0, 0] == pytest.approx(5e-6)
    assert st.ai[0, 0, AI_ABOVE0 + 1] == 5


def test_metrics_spec_validation_and_config():
    with pytest.raises(ValueError):
        MetricsSpec(reloma_definition="oma_db")
    with pytest.raises(ValueError):
        MetricsSpec(allowed_assignments="sometimes")
    with pytest.raises(ValueError):
        MetricsSpec(lock_threshold_pm=1.0, hysteresis_pm=2.0)
    from mrm import params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26")
    ms = MetricsSpec.from_config(cfg)
    assert ms.lock_threshold_pm == 5.0 and ms.hysteresis_pm == 1.0 and ms.ber_sample_period_s == 2.0


@pytest.fixture(scope="module")
def plant_run():
    """A 10 ms g26 plant run (B = 2) with a per-tick record of every metric input."""
    from mrm import params
    from mrm.control.api import ControllerSpec
    from mrm.plant.plant import Plant
    from mrm.scenarios import registry as scen_registry
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26", overrides=["sim.reset_ramp_s=1e-3"])
        pp = params.realize(cfg, 2, seeds=11)
    sc = scen_registry.from_config(cfg)
    spec = ControllerSpec.from_config(cfg, n_batch=2, nominal=cfg, scenario_drives=sc.drives,
                                      scenario_offsets=sc.offsets)
    n = 10_000
    fields = ["detuning_m", "order", "reloma", "oracle_reloma", "p_heater_w", "p_abs_w", "heater_code", "ber"]
    plant = Plant(pp, sc, spec, record=[(f, 1, "last") for f in fields], record_ticks=n,
                  metrics=MetricsSpec.from_config(cfg, oracle_period_ticks=1))
    # a heater staircase so that the code range, slew, energy and lock states all move
    codes = np.zeros((2, 8), np.int64)
    for i in range(10):
        codes[:] = 2000 * i
        plant.advance({"tx.heater": codes}, n // 10)
    return plant, n


def test_accumulators_equal_trace_metrics_plant(plant_run):
    """07 P0-8 test: the accumulators of a 10 ms plant run equal the metrics computed from its full trace."""
    plant, n = plant_run
    tr = plant.traces()

    def g(f):
        return tr[f"{f}@1:last"]["value"]
    det = g("detuning_m")
    assert det.shape == (n, 2, 8)
    reg = g("heater_code").astype(np.int64)
    # latches: the staircase commands latch at 1000 i (+ controller latency 0)
    popped = np.zeros_like(reg, bool)
    popped[np.arange(0, n, n // 10)] = True
    acc = plant.metrics()
    st = plant.metrics_state
    bits_zero = np.zeros_like(det)
    ber = g("ber")
    ref = metrics_from_trace(st.spec, plant.dt, det_m=det, order=g("order").astype(np.int64),
                             wrong=np.zeros_like(det, bool), reloma=g("reloma"), oracle=g("oracle_reloma"),
                             oracle_unr=g("oracle_reloma"), p_heater_w=g("p_heater_w"), p_abs_w=g("p_abs_w"),
                             reg=reg, popped=popped, ber=ber, bits=bits_zero, code_lo=int(plant.P.he_lo),
                             code_hi=int(plant.P.he_hi), reg0=np.zeros((2, 8), np.int64), k0=0)
    _compare(acc, ref, ("max_abs_detuning_pm", "mean_detuning_pm", "rms_detuning_pm", "time_above_s",
                        "excursion_count", "time_unlocked_s", "excursion_max_s", "first_lock_s", "time_to_lock_s",
                        "locked_final", "fsr_hop_events", "mean_reloma", "oma_penalty_db", "mean_p_abs_w",
                        "heater_energy_j", "heater_sat_s", "heater_low_rail_s", "code_min_used", "code_max_used",
                        "peak_slew_codes_per_s"))
    assert np.all(acc["code_max_used"] == 18000)
    assert np.all(acc["peak_slew_codes_per_s"] == pytest.approx(2000 / (1000 * plant.dt)))
    assert np.all(acc["heater_energy_j"] > 0)
    assert np.all(np.isfinite(acc["oma_penalty_db"]))
