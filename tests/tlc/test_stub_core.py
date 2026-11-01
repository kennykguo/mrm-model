"""The Phase 1 stub core: holds the warm command and adds the source's per-ring dither tones at the 100 kS/s output
rate. Expected values are the closed-form sinusoids."""
from __future__ import annotations

import numpy as np
import pytest

from mrm.control.tlc.stub_core import SOURCE_TONES_HZ, StubCore

TS = 1e-4


def test_source_tone_plan_is_the_ece496_one():
    assert SOURCE_TONES_HZ == (2.0e3, 3.0e3, 3.5e3, 4.5e3)


def test_output_is_hold_plus_tone_at_the_next_blocks_sample_times():
    core = StubCore(4, ts=TS, n_out=10, dither_amp_u=0.01)
    u0 = np.array([0.2, 0.3, 0.4, 0.5])
    core.warm_start(u0)
    t0 = 0.0123
    seq = core.step(np.zeros((25, 4)), t0)
    t = t0 + TS + np.arange(10) * TS / 10                     # the plan covers the next block
    want = u0[None, :] + 0.01 * np.sin(2 * np.pi * np.array(SOURCE_TONES_HZ)[None, :] * t[:, None])
    assert seq.shape == (10, 4)
    np.testing.assert_allclose(seq, want, rtol=0, atol=1e-15)


def test_output_is_clipped_to_the_power_fraction_range():
    core = StubCore(1, ts=TS, n_out=10, dither_amp_u=0.5)
    core.warm_start(np.array([0.9]))
    seq = core.step(np.zeros((0, 1)), 0.0)
    assert seq.min() >= 0.0 and seq.max() <= 1.0


def test_more_rings_than_tones_needs_zero_dither():
    StubCore(8, ts=TS, n_out=10, dither_amp_u=0.0)
    with pytest.raises(ValueError, match="tone"):
        StubCore(8, ts=TS, n_out=10, dither_amp_u=0.01)


def test_last_has_the_source_telemetry_fields():
    core = StubCore(4, ts=TS, n_out=10)
    core.warm_start(np.full(4, 0.3))
    core.step(np.zeros((25, 4)), 0.0)
    for k in ("delta_hat", "rate_hat", "sigma", "modes", "u", "amp", "squelch", "fault", "eff"):
        assert k in core.last
