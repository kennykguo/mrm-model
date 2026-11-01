"""Deviation 2.3-1: the ring count is a parameter of the vendored core (ControllerParams.n_rings) instead of the
source's module-level N = 4. The 4-ring default stays bit-identical to the source (test_core_equivalence.py);
here the core is built and stepped with 8 rings, and mismatched per-ring parameters are refused.

The 8-ring parameters below are placeholders for shape only (copies of the 4-ring values, tones on the 250 Hz
grid); they are NOT a valid 8-ring tone plan or G26 prior (Phase 3)."""
from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from mrm.control.tlc.core import TLC, Mode
from mrm.control.tlc.core.params import ControllerParams, PlantParams


def _params(n: int):
    pp0, cp0 = PlantParams(), ControllerParams()
    rep = lambda t: tuple(np.resize(np.asarray(t, dtype=float), n))
    pp = dataclasses.replace(pp0, hwhm=rep(pp0.hwhm), depth=rep(pp0.depth), laser_power=rep(pp0.laser_power),
                             unheated_offset=rep(pp0.unheated_offset), heater_eff_scale=rep(pp0.heater_eff_scale))
    cp = dataclasses.replace(cp0, n_rings=n, tone_freqs=tuple(2.0e3 + 250.0 * 3 * np.arange(n)))
    return pp, cp


def _warm(n: int) -> TLC:
    pp, cp = _params(n)
    ctl = TLC(pp, cp)
    ctl.warm_start(np.full(n, 0.33), np.asarray(ctl.sp.targets))
    return ctl


def _pd_block(ctl: TLC, n: int) -> np.ndarray:
    return np.full((int(round(ctl.cp.ts * ctl.pp.pd_rate)), n), 0.4)


@pytest.mark.parametrize("n", [4, 8])
def test_core_steps_with_n_rings(n):
    ctl = _warm(n)
    for k in range(20):
        seq = ctl.step(_pd_block(ctl, n), k * ctl.cp.ts)
    assert seq.shape[1] == n
    assert np.all(np.isfinite(seq))
    assert ctl.last["delta_hat"].shape == (n,)
    assert ctl.modes.shape == (n,) and set(np.unique(ctl.modes)) <= set(int(m) for m in Mode)


def test_cold_start_covers_every_ring():
    pp, cp = _params(8)
    ctl = TLC(pp, cp)
    ctl.cold_start()
    assert np.all(ctl.modes == Mode.ACQUIRE)


def test_mismatched_per_ring_parameters_are_refused():
    pp, cp = _params(8)
    with pytest.raises(ValueError, match="hwhm"):
        TLC(dataclasses.replace(pp, hwhm=PlantParams().hwhm), cp)
    with pytest.raises(ValueError, match="tone_freqs"):
        TLC(pp, dataclasses.replace(cp, tone_freqs=ControllerParams().tone_freqs))


@pytest.mark.perf
def test_report_core_step_time_for_4_and_8_rings(capsys):
    """Wall time per 100 us core step (sizes the Monte Carlo campaign, spec 6 Q10). Reported, not asserted."""
    out = {}
    for n in (4, 8):
        ctl = _warm(n)
        pd = _pd_block(ctl, n)
        for k in range(20):
            ctl.step(pd, k * ctl.cp.ts)
        t0 = time.perf_counter()
        steps = 200
        for k in range(20, 20 + steps):
            ctl.step(pd, k * ctl.cp.ts)
        out[n] = (time.perf_counter() - t0) / steps * 1e3
    with capsys.disabled():
        print(f"\nTLC core step: N=4 {out[4]:.2f} ms, N=8 {out[8]:.2f} ms per 100 us step")
