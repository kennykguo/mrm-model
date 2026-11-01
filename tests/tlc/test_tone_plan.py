"""Dither tone plans (mrm/control/tlc/tone_plan.py). The collision rule is checked by hand on small cases; the
search must rediscover the source's 4-ring plan, and the stored 8-ring plan must be collision-free."""
from __future__ import annotations

import pytest

from mrm.control.tlc.stub_core import SOURCE_TONES_HZ
from mrm.control.tlc.tone_plan import EIGHT_RING_BINS, collisions, search, to_hz


def test_source_plan_is_collision_free():
    assert collisions((8, 12, 14, 18)) == []
    assert to_hz((8, 12, 14, 18)) == SOURCE_TONES_HZ


def test_harmonic_on_another_tone_is_a_collision():
    bad = collisions((10, 20))                    # 2 f0 = f1
    assert any("2f0 = 20" in b and "f1 = 20" in b for b in bad)


def test_intermodulation_product_near_a_tone_is_a_collision():
    bad = collisions((10, 17, 27))                # f0 + f1 = 27 = f2
    assert any("f2 = 27" in b and "f0+f1" in b for b in bad)


def test_ripple_and_dc_rules():
    assert any("ripple" in b for b in collisions((20,)))     # 2 f = 40 bins = 10 kHz
    assert any("DC" in b for b in collisions((5,)))


def test_withdrawn_rev1_grid_collides():
    assert collisions(tuple(range(6, 21, 2)))     # 1.5-5.0 kHz in 0.5 kHz steps


def test_search_rediscovers_the_source_plan():
    assert search(4) == (8, 12, 14, 18)


def test_stored_eight_ring_plan_is_collision_free():
    assert len(EIGHT_RING_BINS) == 8
    assert collisions(EIGHT_RING_BINS) == []


@pytest.mark.slow
def test_stored_eight_ring_plan_is_the_search_result():
    assert search(8) == EIGHT_RING_BINS
