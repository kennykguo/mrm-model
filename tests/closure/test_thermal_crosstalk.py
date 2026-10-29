"""Compare the thermal crosstalk of the g26 interim model with the in-ring FEM thermal ROM.

Read this first. Crosstalk is the heat that one ring's heater puts into a neighbour ring. The g26 interim model uses
imec's published numbers (CO24-JOM, CO22-TCPMT: natural convection with an EIC on top; 3 % to the nearest ring,
1.5 % floor to every other ring, first-order poles of 1.5-9 ms). The in-ring FEM ROM, solved with a TEC-held package
(R_pkg 2 K/W), gives a 1.0 % nearest-neighbour coupling that reaches 63 % in 0.12 ms at the 50 um pitch, and a
0.09 % floor at 1 mm.

The G26 package is unpublished, so both are hypotheses (uncertainty.thermal.stack). The default keeps imec's numbers.
The weight-0 choice side_by_side_tec_held (corner g26_tec_package) reproduces the FEM within the tolerances below.
The open conflict is VALIDATION.md#closure-crosstalk-dynamics (the strict xfail test); the other tests guard the
corner and the documented couplings.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from mrm.params import load, realize
from mrm.plant import thermal
from mrm.sim.experiment import NAMED_CORNERS

ROM = Path(thermal.__file__).resolve().parents[1] / "data" / "extracted" / "thermal_rom_inring_noundercut_160.npz"
TIMES_S = np.logspace(-7, 0, 20000)        # time grid for the step responses, s (100 ns to 1 s)


def _fem(src: str, obs: str, d_um: float):
    """(DC K/W, t63 s, Z_heater->ring K/W) of a cross pair of the in-ring FEM ROM at centre distance d_um."""
    rom = thermal.load_rom(ROM, warn=False)
    si, oi = rom.sources.index(src), rom.observers.index(obs)
    r = np.array([np.interp(d_um, rom.cross_r_um, rom.cross_res[si, oi, :, k]) for k in range(rom.taus_s.size)])
    # normalized step response y(t) = sum_k r_k (1 - exp(-t / tau_k)) / sum_k r_k (Foster poles of the ROM)
    y = (r[None, :] * -np.expm1(-TIMES_S[:, None] / rom.taus_s[None, :])).sum(1) / r.sum()
    z0 = float(rom.self_res[rom.sources.index("heater"), rom.observers.index("ring_mode_weighted")].sum())
    return float(r.sum()), float(TIMES_S[np.argmax(y >= 1.0 - np.exp(-1.0))]), z0


def _nominal(name: str):
    """Nominal realized PlantParams (one world) of preset ``name``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return realize(load(name), 1, 0, uncertainty="nominal", variation="off")


def _lag(system, obs: int, src: int):
    """(DC gain K/W, slowest pole s) of the thermal lags from source ``src`` to observer ``obs`` (world 0)."""
    m = np.flatnonzero((system.lag_obs == obs) & (system.lag_src == src))
    return float(system.res_k_per_w[0, m].sum()), float(system.taus_s[0, m].max())


def _z0(system, i: int = 0) -> float:
    """DC gain from ring i's own heater to ring i (Z_heater->ring), K/W."""
    return _lag(system, system.ring_obs[i], system.heater_src[i])[0]


@pytest.mark.xfail(strict=True, reason="VALIDATION.md#closure-crosstalk-dynamics: the g26 default (imec, natural "
                                       "convection) crosstalk pole at the pitch is ~12x slower than the in-ring FEM "
                                       "ROM's t63 (TEC-held package); the G26 package is unpublished")
def test_g26_default_crosstalk_matches_the_fem_rom():
    system = _nominal("g26").thermal.system
    _, tau = _lag(system, system.ring_obs[0], system.heater_src[1])
    _, t63, _ = _fem("heater", "ring_mode_weighted", 50.0)
    assert tau == pytest.approx(t63, rel=1.0)


def test_tec_held_package_reaches_the_fem_crosstalk():
    """The corner g26_tec_package (uncertainty.thermal.stack = side_by_side_tec_held) puts the nearest-neighbour
    crosstalk, its pole, the second neighbour and a die sensor 1 mm away within the FEM ROM's numbers (DC to 10 %,
    t63 to 1.3x), and the robustness set (the named corners) contains it."""
    assert "g26_tec_package" in NAMED_CORNERS
    system = _nominal("g26_tec_package").thermal.system
    z0 = _z0(system)
    dc, tau = _lag(system, system.ring_obs[0], system.heater_src[1])
    fdc, ft63, fz0 = _fem("heater", "ring_mode_weighted", 50.0)
    assert dc / z0 == pytest.approx(fdc / fz0, rel=0.1)
    assert ft63 / 1.3 <= tau <= ft63 * 1.3
    dc2, tau2 = _lag(system, system.ring_obs[0], system.heater_src[2])
    fdc2, ft2, _ = _fem("heater", "ring_mode_weighted", 100.0)
    assert dc2 / z0 == pytest.approx(fdc2 / fz0, rel=0.1)
    assert ft2 / 3.0 <= tau2 <= ft2 * 3.0                 # the FEM response is far from first order at 100 um
    die = system.observer_index("temp_sensor", name="die0")
    ddc, _ = _lag(system, die, system.heater_src[0])
    fdd, _, _ = _fem("heater", "temp_sensor", 1000.0)
    assert ddc / z0 == pytest.approx(fdd / fz0, rel=0.1)
    # sampled g26 worlds never draw it (weight 0, a named corner only; appended last, so no draw changed)
    pp = realize(load("g26"), 256, 11)
    choices = pp.world_choices["thermal.stack"]
    assert choices[-1] == "side_by_side_tec_held"
    assert choices.index("side_by_side_tec_held") not in set(np.unique(pp.world["thermal.stack"]))


def test_far_site_floor_is_the_package_term():
    """The site law's floor (imec 1.5 %, measured to 300 um with natural convection) applies
    at any distance, so the g26 die sensor 1 mm from the bank sees each heater with 1.5 % of Z_heater->ring (85 K/W,
    one 9 ms pole), 33x the FEM ROM's TEC-held package. It is now its own key, ring.thermal.site_floor_frac, equal to
    xt_floor_frac in every hypothesis except side_by_side_tec_held (0.09 %, the FEM at 1 mm), so the default plant is
    unchanged."""
    system = _nominal("g26").thermal.system
    z0 = _z0(system)
    die = system.observer_index("temp_sensor", name="die0")
    dc, tau = _lag(system, die, system.heater_src[0])
    assert dc / z0 == pytest.approx(0.015, rel=1e-9) and tau == pytest.approx(8.99e-3)
    for name, frac in (("g26_tec_package", 0.0009),):
        s2 = _nominal(name).thermal.system
        assert _lag(s2, s2.observer_index("temp_sensor", name="die0"), s2.heater_src[0])[0] / _z0(s2) == \
            pytest.approx(frac, rel=1e-9)
    # stacked EIC and the textbook family keep the site floor equal to the ring floor, as before the new key
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for alts in ({"uncertainty.thermal.stack": "stacked_eic"}, {"uncertainty.thermal.family": "textbook"}):
            pp = realize(load("g26", alts=alts), 1, 0, uncertainty="nominal", variation="off")
            it = pp.thermal.inputs.interim
            np.testing.assert_array_equal(it["site_floor_frac"], it["xt_floor_frac"])
