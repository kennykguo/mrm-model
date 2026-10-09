"""Verification of mrm.core.timing (07 P0-6): integer ticks, rejection of non-integer tick counts, edges, the
first-response tick of the event order, and the latency FIFOs (test_fifo_depth)."""
import math

import numba as nb
import numpy as np
import pytest

from mrm.core import timing as tm


# ---- seconds -> ticks -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value_s, dt_s, ticks", [
    (2e-6, 1e-6, 2), (0.0, 1e-6, 0), (1e-5, 1e-7, 100), (2e-6, 1e-7, 20), (2.0, 1e-6, 2_000_000),
    (30.4 * 3600.0, 1e-6, 109_440_000_000),      # the 30.4 h soak (07 P0-10) at 1 us
    (0.3e-6, 0.1e-6, 3),                        # 0.3e-6 / 0.1e-6 = 2.9999999999999996 in float64
])
def test_seconds_to_ticks_exact(value_s, dt_s, ticks):
    assert tm.seconds_to_ticks(value_s, dt_s, "k") == ticks


@pytest.mark.parametrize("value_s", [2.5e-6, 1.0000001e-6, 0.5e-6])
def test_non_integer_ticks_rejected_with_key(value_s):
    with pytest.raises(tm.TimingError, match=r"ring\.monitor\.adc_latency_s"):
        tm.seconds_to_ticks(value_s, 1e-6, "ring.monitor.adc_latency_s")


@pytest.mark.parametrize("bad", [-1e-6, float("nan"), float("inf"), "abc", None])
def test_bad_durations_rejected(bad):
    with pytest.raises(tm.TimingError, match="the_key"):
        tm.seconds_to_ticks(bad, 1e-6, "the_key")


def test_zero_not_allowed_when_requested():
    with pytest.raises(tm.TimingError, match="at least one tick"):
        tm.seconds_to_ticks(0.0, 1e-6, "ring.monitor.integration_window_s", allow_zero=False)


@pytest.mark.parametrize("dt", [0.0, -1e-6, float("nan")])
def test_bad_dt_rejected(dt):
    with pytest.raises(tm.TimingError, match="dt_plant_s"):
        tm.seconds_to_ticks(1e-6, dt, "k")


def test_rate_to_period_ticks():
    assert tm.rate_to_period_ticks(100e3, 1e-6, "controller_interface.update_rate_hz") == 10
    assert tm.rate_to_period_ticks(500e3, 1e-6, "ring.monitor.adc_rate_hz") == 2
    assert tm.rate_to_period_ticks(10e6, 0.1e-6, "x") == 1
    for rate in (3e6, 300e3, 2e6):     # 0.333, 3.333, 0.5 ticks at 1 us
        with pytest.raises(tm.TimingError, match="controller_interface.update_rate_hz"):
            tm.rate_to_period_ticks(rate, 1e-6, "controller_interface.update_rate_hz")
    for rate in (0.0, -1.0, float("inf")):
        with pytest.raises(tm.TimingError):
            tm.rate_to_period_ticks(rate, 1e-6, "k")


def test_check_tick_table_names_every_bad_key():
    entries = [("ring.monitor.adc_rate_hz", 300e3, "hz"), ("ring.heater.latency_s", 1.5e-6, "s"),
               ("controller_interface.update_rate_hz", 100e3, "hz"), ("ring.monitor.integration_window_s", 0.0, "s+")]
    with pytest.raises(tm.TimingError) as ei:
        tm.check_tick_table(entries, 1e-6)
    msg = str(ei.value)
    for key in ("ring.monitor.adc_rate_hz", "ring.heater.latency_s", "ring.monitor.integration_window_s"):
        assert key in msg
    assert "controller_interface.update_rate_hz" not in msg
    ok = tm.check_tick_table({"a": (100e3, "hz"), "b": (2e-6, "s")}, 1e-6)
    assert ok == {"a": 10, "b": 2}


def test_ticks_to_seconds_is_derived_not_accumulated():
    n = 10**10
    assert tm.ticks_to_seconds(n, 1e-6) == n * 1e-6
    assert np.allclose(tm.ticks_to_seconds(np.arange(5), 1e-6), np.arange(5) * 1e-6)


# ---- edges ----------------------------------------------------------------------------------------------------------

def test_edges():
    assert [k for k in range(25) if tm.is_edge(k, 10, 3)] == [3, 13, 23]
    assert tm.next_edge(4, 10, 3) == 13
    assert tm.next_edge(13, 10, 3) == 13
    assert tm.next_edge(0, 10, 3) == 3
    assert tm.edge_index(23, 10, 3) == 2
    assert tm.edge_index(2, 10, 3) == -1


def test_first_response_tick():
    # all latencies 0, every tick an ADC sample and a controller edge: acts on [k, k+1), seen at k + 1
    assert tm.first_response_tick(100, 0, 0, 1, 0, 0, 1, 0) == 101
    # k_act = 105, first sample tick on the 2-tick grid >= 106 is 106, released at 108, next 10-tick edge 110
    assert tm.first_response_tick(100, 2, 3, 2, 0, 2, 10, 0) == 110


@pytest.mark.parametrize("rate_hz", [10e3, 20e3, 50e3, 100e3, 200e3])
def test_latency_in_ticks_independent_of_controller_rate(rate_hz):
    """07 P0-6: latencies are owned in seconds by the I/O path; changing the controller rate does not change them."""
    dt = 1e-6
    lat = tm.check_tick_table({"ring.monitor.adc_latency_s": (2e-6, "s"), "ring.heater.latency_s": (3e-6, "s"),
                               "controller_interface.update_rate_hz": (rate_hz, "hz")}, dt)
    assert lat["ring.monitor.adc_latency_s"] == 2 and lat["ring.heater.latency_s"] == 3
    p = lat["controller_interface.update_rate_hz"]
    adc_p, ctrl_lat = 2, 1
    k = 10 * p                                    # a controller edge (and an ADC grid tick: p is even)
    # the plant-side delay (command edge -> first sample that sees it -> its release) does not depend on the rate
    k_act = k + ctrl_lat + lat["ring.heater.latency_s"]
    s = tm.next_edge(k_act + 1, adc_p, 0)
    assert s + lat["ring.monitor.adc_latency_s"] - k == 1 + 3 + 1 + 1 + 2      # latch, DAC, ZOH tick, grid, ADC
    # the controller observes it at its next edge: the only rate dependence is that quantization (< one period)
    first = tm.first_response_tick(k, ctrl_lat, 3, adc_p, 0, 2, p, 0)
    assert 0 <= first - (s + 2) < p and tm.is_edge(first, p, 0)
    # with every tick an edge, the observed delay is exactly the sum of the latencies + one tick of ZOH
    assert tm.first_response_tick(k, 0, 3, 1, 0, 2, 1, 0) - k == 3 + 1 + 2


# ---- FIFOs ----------------------------------------------------------------------------------------------------------

@nb.njit(cache=True)
def _run_delay(vals, rel, meta, latency, period, n_ticks, out):
    """Push the tick number every ``period`` ticks (release at tick + latency), pop at every tick (push first).
    out[t] = value popped at tick t or -1. Returns -2 on overflow, else the maximum occupancy."""
    occ = 0
    for t in range(n_ticks):
        if t % period == 0:
            if not tm.fifo_push(vals, rel, meta, 0, t, t + latency):
                return -2
        occ = max(occ, tm.fifo_count(meta, 0))
        ok, v, r = tm.fifo_pop_ready(vals, rel, meta, 0, t)
        out[t] = v if ok else -1
        if ok and r != t:
            return -3
    return occ


@pytest.mark.parametrize("latency", [0, 1, 2, 3, 5, 7, 10])
@pytest.mark.parametrize("period", [1, 2, 3, 4])
def test_fifo_depth(latency, period):
    cap = tm.fifo_capacity(latency, period)
    assert cap == latency // period + 1
    vals, rel, meta = tm.fifo_alloc(1, cap)
    out = np.zeros(200, dtype=np.int64)
    occ = _run_delay(vals, rel, meta, latency, period, 200, out)
    assert occ == cap                                    # the capacity is reached: it is tight
    pushed = np.arange(0, 200, period)
    popped_at = np.flatnonzero(out >= 0)
    assert np.array_equal(out[popped_at], popped_at - latency)          # exactly `latency` ticks of delay
    assert np.array_equal(popped_at, pushed[pushed + latency < 200] + latency)
    if cap > 1:
        vals, rel, meta = tm.fifo_alloc(1, cap - 1)
        assert _run_delay(vals, rel, meta, latency, period, 200, out) == -2    # one entry fewer overflows


def test_fifo_independent_queues_and_pop_all():
    vals, rel, meta = tm.fifo_alloc(3, 4, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            assert tm.fifo_push(vals, rel, meta, i, 10.0 * i + j, 5 + j + i)
    assert tm.fifo_push(vals, rel, meta, 0, 3.0, 9)              # 4th entry fits (capacity 4)
    assert not tm.fifo_push(vals, rel, meta, 0, 4.0, 10)         # 5th is refused
    ov = np.zeros(4)
    orl = np.zeros(4, dtype=np.int64)
    n = tm.fifo_pop_all_ready(vals, rel, meta, 1, 7, ov, orl)
    assert n == 2 and list(ov[:2]) == [10.0, 11.0] and list(orl[:2]) == [6, 7]
    ok, v, r = tm.fifo_peek(vals, rel, meta, 1)
    assert ok and v == 12.0 and r == 8
    assert tm.fifo_count(meta, 0) == 4 and tm.fifo_count(meta, 2) == 3
    tm.fifo_clear(meta)
    assert all(tm.fifo_count(meta, i) == 0 for i in range(3))
    ok, _, _ = tm.fifo_pop_ready(vals, rel, meta, 0, 10**9)
    assert not ok


def test_fifo_wraparound_keeps_order():
    vals, rel, meta = tm.fifo_alloc(1, 3)
    got = []
    for t in range(50):
        assert tm.fifo_push(vals, rel, meta, 0, t, t + 2)
        ok, v, r = tm.fifo_pop_ready(vals, rel, meta, 0, t)
        if ok:
            got.append(v)
    assert got == list(range(48))


def test_fifo_capacity_rejects_bad_input():
    with pytest.raises(tm.TimingError):
        tm.fifo_capacity(-1, 1)
    with pytest.raises(tm.TimingError):
        tm.fifo_capacity(3, 0)
    with pytest.raises(ValueError):
        tm.fifo_alloc(0, 3)


def test_event_order_enum():
    assert [int(e) for e in tm.EventPhase] == [1, 2, 3, 4, 5]
    assert "SensorBank" in tm.__doc__ and "latency_ticks" in tm.__doc__


# ---- review additions -----------------------------------------------------------------------------------------------

@nb.njit(cache=True)
def _max_occupancy(vals, rel, meta, latency, push_p, push_ph, pop_p, pop_ph, n_ticks, out_v, out_r):
    """Push every push_p ticks (phase push_ph), drain every ready entry every pop_p ticks (phase pop_ph), push before
    pop. Returns the largest occupancy just before a pop, -2 on overflow, -3 if an entry leaves before its release
    tick, -4 if a ready entry waits past the first pop tick at or after its release."""
    occ = 0
    for t in range(n_ticks):
        if t % push_p == push_ph:
            if not tm.fifo_push(vals, rel, meta, 0, t, t + latency):
                return -2
        if t % pop_p == pop_ph:
            occ = max(occ, tm.fifo_count(meta, 0))
            n = tm.fifo_pop_all_ready(vals, rel, meta, 0, t, out_v, out_r)
            for j in range(n):
                if out_r[j] > t:
                    return -3
                if t - out_r[j] >= pop_p:
                    return -4
    return occ


@pytest.mark.parametrize("pop_p", [1, 2, 3, 5, 10])
def test_fifo_capacity_with_pop_period_is_tight_bound(pop_p):
    """fifo_capacity(lat, push, pop) bounds the occupancy for every phase of the two grids and is reached for one
    (so a FIFO with one entry fewer overflows); a SensorBank that drains only at controller edges needs it."""
    out_v = np.zeros(64, dtype=np.int64)
    out_r = np.zeros(64, dtype=np.int64)
    for latency in range(0, 9):
        for push_p in range(1, 5):
            cap = tm.fifo_capacity(latency, push_p, pop_p)
            worst = 0
            for push_ph in range(push_p):
                for pop_ph in range(pop_p):
                    vals, rel, meta = tm.fifo_alloc(1, cap)
                    occ = _max_occupancy(vals, rel, meta, latency, push_p, push_ph, pop_p, pop_ph, 300, out_v, out_r)
                    assert occ >= 0, (latency, push_p, pop_p, push_ph, pop_ph, occ)
                    worst = max(worst, occ)
            assert worst == cap, (latency, push_p, pop_p)
    assert tm.fifo_capacity(5, 2) == tm.fifo_capacity(5, 2, 1) == 5 // 2 + 1
    with pytest.raises(tm.TimingError):
        tm.fifo_capacity(3, 1, 0)


def test_tick_counts_beyond_int64_rejected():
    with pytest.raises(tm.TimingError, match="the_key"):
        tm.seconds_to_ticks(1e30, 1e-6, "the_key")
    assert tm.seconds_to_ticks(4.0e12, 1e-6, "k") == 4 * 10**18        # inside the 2**62 range


def test_edge_jitter_is_bounded_reproducible_and_uniform():
    """07 P2-7 tick jitter: d_j in 0..J, a pure function of (key, j); about uniform; 0 when J = 0."""
    from mrm.core import rng
    key = rng.stream_key(9, int(rng.Stream.SCHEDULE))
    d = np.array([tm.edge_jitter(key, j, 3) for j in range(4000)])
    assert d.min() == 0 and d.max() == 3
    assert np.array_equal(d, [tm.edge_jitter(key, j, 3) for j in range(4000)])
    counts = np.bincount(d, minlength=4)
    assert np.all(np.abs(counts - 1000) < 5 * np.sqrt(1000 * 0.75))
    assert all(tm.edge_jitter(key, j, 0) == 0 for j in range(10))
    other = rng.stream_key(10, int(rng.Stream.SCHEDULE))
    assert not np.array_equal(d[:200], [tm.edge_jitter(other, j, 3) for j in range(200)])
