"""The monitor sample buffer gathers the samples of several controller edges into one core block, in order, using
``tx.monitor.count`` to skip the unused slots of ``tx.monitor.samples``."""
from __future__ import annotations

import numpy as np
import pytest

from mrm.control.tlc.sample_buffer import SampleBuffer


def _edge(values_per_ring, m=3):
    """samples (1, N, m) and count (1, N) for one edge from per-ring lists of new samples."""
    n = len(values_per_ring)
    samples = np.full((1, n, m), -1.0)
    count = np.zeros((1, n), dtype=np.int64)
    for i, vals in enumerate(values_per_ring):
        samples[0, i, :len(vals)] = vals
        count[0, i] = len(vals)
    return samples, count


def test_drain_returns_the_counted_samples_in_arrival_order():
    buf = SampleBuffer(n_batch=1, n_rings=2, capacity=8)
    buf.push(*_edge([[1, 2, 3], [10, 20]]))
    buf.push(*_edge([[4, 5], [30, 40, 50]]))
    data, fill = buf.drain()
    assert fill.tolist() == [[5, 5]]
    assert data[0, 0, :5].tolist() == [1, 2, 3, 4, 5]
    assert data[0, 1, :5].tolist() == [10, 20, 30, 40, 50]


def test_drain_empties_the_buffer():
    buf = SampleBuffer(n_batch=1, n_rings=1, capacity=4)
    buf.push(*_edge([[1, 2]]))
    buf.drain()
    data, fill = buf.drain()
    assert fill.tolist() == [[0]]


def test_drained_arrays_are_copies():
    buf = SampleBuffer(n_batch=1, n_rings=1, capacity=4)
    buf.push(*_edge([[7]]))
    data, _ = buf.drain()
    buf.push(*_edge([[9]]))
    assert data[0, 0, 0] == 7


def test_overflow_raises_instead_of_dropping_samples():
    buf = SampleBuffer(n_batch=1, n_rings=1, capacity=4)
    buf.push(*_edge([[1, 2, 3]]))
    with pytest.raises(OverflowError):
        buf.push(*_edge([[4, 5]]))
