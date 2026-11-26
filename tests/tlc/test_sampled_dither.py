"""Sampled dither shapes on G26 (source ae9556b B1): the tone plan for out_rate / N tones, the priors' monitor noise,
and the faithful core's ``dither_shape`` / ``tone_plan`` knobs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import mrm.params as P
from mrm.control.api import ControllerSpec
from mrm.control.tlc.core import params as vparams
from mrm.control.tlc.faithful_core import FaithfulCore
from mrm.control.tlc.priors import g26_priors
from mrm.control.tlc.tone_plan import SAMPLED_N, sampled_collisions, sampled_plan

TIER = "controller_interface.knowledge=[line_grid,ring_to_line,model_nominal]"


@pytest.fixture(scope="module")
def spec():
    cfg = P.load("g26", overrides=[TIER])
    return ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg)


def test_rule_with_every_ring_in_reach_accepts_the_source_square_tones():
    assert sampled_collisions(SAMPLED_N, reach=3) == []
    assert tuple(100e3 / n for n in SAMPLED_N) == vparams.SQUARE_TONES


def test_rule_flags_a_harmonic_on_a_neighbour_tone():
    bad = sampled_collisions((30, 10 * 3), reach=1)          # identical tones on neighbours
    assert bad and all("ring" in b for b in bad)
    assert any("3f" in b for b in sampled_collisions((36, 12), reach=1))   # 3 x 2778 Hz = 8333 Hz = f1


def test_eight_ring_plan_tiles_the_source_tones_every_four_rings():
    f = sampled_plan(8)
    assert f[:4] == vparams.SQUARE_TONES and f[4:] == f[:4]
    assert max(f) < 7.2e3
    assert sampled_collisions(tuple(SAMPLED_N[i % 4] for i in range(8)), reach=4) != []   # reuse needs reach <= 3


def test_every_sampled_tone_is_a_whole_even_number_of_output_samples():
    for f in sampled_plan(8):
        n = 100e3 / f
        assert n == pytest.approx(round(n)) and round(n) % 2 == 0


def test_priors_squelch_floor_is_the_g26_monitor_sigma(spec):
    pp = g26_priors(spec).pp
    assert not pp.pd_shot and not pp.pd_johnson
    assert pp.noise_floor_fs() == pytest.approx(pp.pd_noise_fs, rel=1e-12)
    assert len(pp.pd_offset_fs) == spec.n_rings and not any(pp.pd_offset_fs)


@pytest.mark.parametrize("shape", ["triangle", "three", "square"])
def test_faithful_core_uses_the_sampled_plan_for_sampled_shapes(spec, shape):
    core = FaithfulCore(spec, ts=1e-4, n_out=10, dither_shape=shape)
    assert core.ctl.cp.dither_shape == shape and core.ctl.dither.square
    assert core.ctl.cp.tone_freqs == sampled_plan(spec.n_rings)
    assert core.ctl.dither.n_per.tolist() == [SAMPLED_N[i % 4] for i in range(spec.n_rings)]


def test_faithful_core_sine_keeps_the_grid_plan_unless_asked(spec):
    grid = FaithfulCore(spec, ts=1e-4, n_out=10)
    assert grid.ctl.cp.dither_shape == "sine" and grid.ctl.cp.tone_freqs == g26_priors(spec).cp.tone_freqs
    sampled = FaithfulCore(spec, ts=1e-4, n_out=10, tone_plan="sampled")
    assert sampled.ctl.cp.dither_shape == "sine" and sampled.ctl.cp.tone_freqs == sampled_plan(spec.n_rings)


@pytest.mark.parametrize("kw", [{"dither_shape": "sawtooth"}, {"tone_plan": "grid"}])
def test_faithful_core_rejects_unknown_knobs(spec, kw):
    with pytest.raises(ValueError):
        FaithfulCore(spec, ts=1e-4, n_out=10, **kw)


def test_triangle_plan_is_slew_limited_with_triangle_harmonics(spec):
    """B1's triangle pattern: the edges respect the per-sample step limit; the fundamental of a unit-peak triangle is
    8 / pi^2 (times the hold's sinc) and the 3rd harmonic is about 1/9 of it."""
    core = FaithfulCore(spec, ts=1e-4, n_out=10, dither_shape="triangle")
    d = core.ctl.dither
    for i in range(spec.n_rings):
        pat = d.pat[i]
        assert np.max(np.abs(np.diff(np.r_[pat, pat[:1]]))) <= d.edge_step / d._built[i] + 1e-12
        assert d.fund[i] == pytest.approx(8 / np.pi ** 2, rel=0.05)
        assert abs(d.b[1, i]) / abs(d.b[0, i]) == pytest.approx(1 / 9, rel=0.15)


def test_report_removes_the_sampled_tones_and_third_harmonics():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import tlc_report as R
    lines = R._tlc_tones("tlc", 8, "triangle")
    assert lines == sorted({k * f for f in set(sampled_plan(8)) for k in (1, 3)})
    assert R._tlc_tones("pi_lock_c", 8, "triangle") is None
