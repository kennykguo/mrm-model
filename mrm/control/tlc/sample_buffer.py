"""Monitor samples gathered over several controller edges into one core block.

mrm hands the controller 2-3 monitor samples per ring at each 10 us edge (``tx.monitor.samples`` (B, N, M), with
``tx.monitor.count`` (B, N) of them new). The core consumes one 100 us block at a time, so the adapter pushes
every edge and drains once per core step.
"""
from __future__ import annotations

import numpy as np


class SampleBuffer:
    """Fixed-capacity (B, N, capacity) buffer; raises rather than drop a sample."""

    def __init__(self, n_batch: int, n_rings: int, capacity: int):
        self._data = np.zeros((n_batch, n_rings, capacity))
        self._fill = np.zeros((n_batch, n_rings), dtype=np.int64)

    @property
    def capacity(self) -> int:
        return self._data.shape[-1]

    def push(self, samples: np.ndarray, count: np.ndarray) -> None:
        """Append the first ``count[b, n]`` of ``samples[b, n, :]`` to each ring's block."""
        samples = np.asarray(samples, dtype=np.float64)
        count = np.asarray(count, dtype=np.int64)
        if np.any(self._fill + count > self.capacity):
            raise OverflowError(f"SampleBuffer: more than {self.capacity} samples in one core block; raise "
                                f"buffer_capacity or run the core more often")
        b_idx, n_idx = np.indices(count.shape)
        for m in range(samples.shape[-1]):
            new = m < count
            self._data[b_idx[new], n_idx[new], self._fill[new]] = samples[..., m][new]
            self._fill += new

    def drain(self) -> tuple[np.ndarray, np.ndarray]:
        """(data copy (B, N, capacity), fill (B, N)) of the block so far; the buffer starts empty again."""
        out = self._data.copy(), self._fill.copy()
        self._fill[:] = 0
        return out
