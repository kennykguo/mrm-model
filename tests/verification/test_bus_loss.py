"""Checks that the bus waveguide loss between rings reaches every code path that walks the bus.

Read this first
---------------
Light travels along the bus from ring to ring. Each segment between two rings transmits t2 of the power
(mrm.plant.phase equation 6), with t2 = 10^(-loss_db_per_mm * spacing_mm / 10) from the preset leaves
optical_path.bus_loss_db_per_mm and optical_path.ring_spacing_on_bus_um. The first ring has no segment in front of it:
the chip input loss covers that path.

These paths must all apply t2, and they must agree:

- the engine, on both optics paths (kernel_ref copies the plant's rows, so it is covered too);
- runner.solve_locked_steady_state;
- mrm.control.models (the controllers' nominal model);
- mrm.params.derive and the replicas' bus walk.

An earlier version ran every segment at transmission 1 in the engine and in the controllers' models, while derive and
the replicas applied the loss. These tests catch that mismatch.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from mrm import params
from mrm.params.derive import derive
from mrm.plant import optics, phase
from mrm.sim import runner


def _cfg(*ov):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26", overrides=list(ov))


def _nominal_pp(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.realize(cfg, 1, seeds=0, uncertainty="nominal", variation="off")


@pytest.fixture(scope="module")
def g26():
    return _cfg()


@pytest.fixture(scope="module")
def seg(g26):
    """Segment power transmission of g26 from the preset leaves (0.073 dB/mm over 50 um)."""
    loss_db_per_mm = float(g26.get("optical_path.bus_loss_db_per_mm"))
    spacing_mm = float(g26.get("optical_path.ring_spacing_on_bus_um")) * 1e-3
    return 10.0 ** (-loss_db_per_mm * spacing_mm / 10.0)


def test_bus_t2_from_plant_params(g26, seg):
    pp = _nominal_pp(g26)
    t2 = phase.bus_t2_from_plant_params(pp)
    order = np.asarray(pp.layout.bus_order)
    assert t2.shape == (1, pp.n_rings)
    assert t2[0, order[0]] == 1.0                     # the chip input loss covers the path to the first ring
    np.testing.assert_allclose(t2[0, order[1:]], seg, rtol=1e-12)
    assert seg == pytest.approx(0.99916, abs=1e-5)
    # zero or absent spacing: every segment transmits 1
    for ov in ("optical_path.ring_spacing_on_bus_um=0.0", "optical_path.ring_spacing_on_bus_um=null"):
        assert np.all(phase.bus_t2_from_plant_params(_nominal_pp(_cfg(ov))) == 1.0)


@pytest.mark.parametrize("self_heating", ["on", "off"])
def test_engine_applies_the_bus_segment_loss(g26, seg, self_heating):
    """Every ring's bus input is the previous ring's through output times the segment transmission, on every line.
    The bus output is the last ring's through output (there is no output segment). Without the loss every ratio
    would be 1.0."""
    cfg = g26 if self_heating == "on" else _cfg("fidelity.self_heating=off")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = runner.run(cfg, None, "static", duration_s=20e-6, uncertainty="nominal", variation="off")
    pl = r.plant
    v = optics.selfheat_views(pl.S.selfheat_float[0], pl.P.selfheat_int[0])
    np.testing.assert_array_equal(
        v[optics.SELF_HEATING_VIEW_INDEX_BUS_SEGMENT_POWER_TRANSMISSION],
        phase.bus_t2_from_plant_params(pl.pp)[0],
    )
    assert float(v[optics.SELF_HEATING_VIEW_INDEX_BUS_OUTPUT_SEGMENT_POWER_TRANSMISSION][0]) == 1.0
    out_rl, out_l = (
        v[optics.SELF_HEATING_VIEW_INDEX_RING_LINE_OUTPUTS],
        v[optics.SELF_HEATING_VIEW_INDEX_LASER_LINE_OUTPUTS],
    )
    order = np.asarray(pl.pp.layout.bus_order)
    for q in range(1, order.size):
        k, kp = order[q], order[q - 1]
        np.testing.assert_allclose(
            out_rl[optics.RING_LINE_OUTPUT_ROW_INPUT_POWER, k]
            / out_rl[optics.RING_LINE_OUTPUT_ROW_THROUGH_POWER, kp],
            seg,
            rtol=1e-12,
        )
    np.testing.assert_allclose(
        out_l[optics.LASER_LINE_OUTPUT_ROW_BUS_OUTPUT_POWER]
        / out_rl[optics.RING_LINE_OUTPUT_ROW_THROUGH_POWER, order[-1]],
        1.0,
        rtol=1e-12,
    )
    # Truth p_in_w (the ring's declared line at its input) carries the same walk
    tr = r.truth_final
    r2l = np.asarray(pl.pp.layout.ring_to_line)
    np.testing.assert_allclose(
        np.asarray(tr.p_in_w)[0],
        out_rl[optics.RING_LINE_OUTPUT_ROW_INPUT_POWER, np.arange(r2l.size), r2l],
        rtol=1e-12,
    )


def test_derive_p_in_ring_uses_the_plant_segment_law(g26):
    """derive's p_in_ring_dbm (documentation: the chip power at each ring without the upstream rings' filtering) is
    the chip-input power times the plant's segment transmissions along the bus."""
    d = derive(g26)
    pp = d.params
    t2 = phase.bus_t2_from_plant_params(pp)[0]
    order = np.asarray(pp.layout.bus_order)
    walk = np.ones(pp.n_rings)
    acc = 1.0
    for k in order:
        acc *= t2[k]
        walk[k] = acc
    r2l = np.asarray(pp.layout.ring_to_line)
    p_chip = np.asarray(pp.laser.power_w)[0, r2l] * float(pp.laser.input_transmission[0, 0])
    np.testing.assert_allclose(1e-3 * 10.0 ** (np.asarray(d.diagnostics["p_in_ring_dbm"]) / 10.0), p_chip * walk,
                               rtol=1e-12)


def test_solve_locked_steady_state_applies_the_bus_segment_loss(g26, seg):
    """solve_locked_steady_state's own-line walk loses t2 per segment: with the loss switched off the absorbed power
    of the ring at bus position q rises by 1/t2^q (up to 0.59 % at q = 7; the lock detuning and temperatures barely
    move)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kw = dict(ambient_c=[25.0], bias_v=[1.05], uncertainty="nominal", variation="off")
        qs = runner.solve_locked_steady_state(g26, **kw)
        qs0 = runner.solve_locked_steady_state(_cfg("optical_path.ring_spacing_on_bus_um=0.0"), **kw)
    order = np.asarray(_nominal_pp(g26).layout.bus_order)
    pos = np.argsort(order)
    ratio = qs.p_abs_w[0, 0, 0] / qs0.p_abs_w[0, 0, 0]
    np.testing.assert_allclose(ratio, seg ** pos, rtol=5e-4)
    assert ratio[order[-1]] < 0.995


def test_static_point_sees_the_bus_segment_loss(g26, seg):
    """mrm.control.models.static_point (the controllers' nominal model) walks the bus with the loss: the own-line power
    at each ring equals the upstream rings' through transmission times t2 per segment."""
    from mrm.control import models
    pp = _nominal_pp(g26)
    cond = models.conditions_from_config(pp, 0)
    sp = models.static_point(pp, 0, np.full(pp.n_rings, 107e-12), cond=cond)
    sp0 = models.static_point(_nominal_pp(_cfg("optical_path.ring_spacing_on_bus_um=0.0")), 0,
                              np.full(pp.n_rings, 107e-12), cond=cond)
    order = np.asarray(pp.layout.bus_order)
    pos = np.argsort(order)
    np.testing.assert_allclose(sp.p_in_w / sp0.p_in_w, seg ** pos, rtol=1e-6)
    first = int(order[0])                              # the first ring sees the chip-input power of its line
    assert math.isclose(float(sp.p_in_w[first]), float(cond.p_chip_w[int(pp.layout.ring_to_line[first])]),
                        rel_tol=1e-12)
