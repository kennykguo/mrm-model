"""The g26 thermal notes and the thermal contract quote the artefacts and the code correctly (physics_thermal review,
2026-10-16). A provenance note that misstates a solver artefact misleads whoever judges a parameter against it
(ARCHITECTURE 7.1-7.2): these tests recompute every number the notes quote."""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from mrm.params import load, realize
from mrm.plant import thermal as th

ROOT = Path(__file__).resolve().parents[2]
ROM = ROOT / "mrm" / "data" / "extracted" / "thermal_rom_inring_noundercut_160.npz"
T = np.logspace(-9, -3, 60001)


def _step(r, taus):
    y = (np.asarray(r)[None, :] * -np.expm1(-T[:, None] / np.asarray(taus)[None, :])).sum(1)
    return y / np.sum(r)


def _t63(y):
    return float(T[np.argmax(y >= 1.0 - np.exp(-1.0))])


def _note(cfg, path):
    return cfg.leaves[path].note


def _nums(pattern, text):
    m = re.search(pattern, text)
    assert m, f"note does not state {pattern!r}: {text}"
    return [float(g) for g in m.groups()]


@pytest.fixture(scope="module")
def g26():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = load("g26")
        return cfg, realize(cfg, 1, 0, uncertainty="nominal", variation="off")


@pytest.fixture(scope="module")
def rom():
    return th.load_rom(ROM, warn=False)


def test_selfheat_note_quotes_the_rom_step_fractions(g26, rom):
    """physics_thermal-6: the note said 'the in-ring FEM ROM puts 65 % of Z_opt below 1 us' (an older fit; the
    artefact puts 38.5 % of its DC in poles below 1 us, and pole shares are not unique between fits). It now quotes
    the step response: 27 % at 100 ns, 54 % at 1 us, 63 % at 1.76 us, and the interim path's 48 % at 1 us."""
    cfg, pp = g26
    note = _note(cfg, "ring.thermal.selfheat_tau_s")
    assert "65 %" not in note
    r = rom.self_res[rom.sources.index("waveguide_absorption"), rom.observers.index("ring_mode_weighted")]
    y = _step(r, rom.taus_s)
    f100, f1, t63 = _nums(r"reaches (\d+) % of Z_opt at 100 ns, (\d+) % at 1 us and 63 % at ([\d.]+) us", note)
    assert f100 == pytest.approx(100 * np.interp(1e-7, T, y), abs=0.6)
    assert f1 == pytest.approx(100 * np.interp(1e-6, T, y), abs=0.6)
    assert t63 == pytest.approx(_t63(y) * 1e6, abs=0.006)
    below = _nums(r"([\d.]+) % of the DC sits in poles below 1 us", note)[0]
    assert below == pytest.approx(100 * r[rom.taus_s < 1e-6].sum() / r.sum(), abs=0.06)
    sy = pp.thermal.system
    yi = th.step_response(sy, np.array([1e-6]), obs=[sy.ring_obs[0]], src=[sy.wg_src[0]])[0, 0, 0, 0]
    dci = th.dc_gain(sy)[0][sy.ring_obs[0], sy.wg_src[0]]
    assert _nums(r"the interim path reaches (\d+) % at 1 us", note)[0] == pytest.approx(100 * yi / dci, abs=0.6)


def test_heater_node_tau_is_tagged_as_the_excess_pole(g26, rom):
    """physics_thermal-4: heater_node_tau_s was tagged solver(assumed-geometry) as 'the t63 of heater ->
    heater_resistor in the in-ring FEM ROM (1.47 us)', but the interim model uses it only as the pole of the excess
    (h - 1) Z0, 13.6 % of Z_HH: the interim Z_HH has t63 6.7 us, 4.5x the FEM's, set by h and S_H. The tag is now
    assumed and the note quotes the FEM excess t63, the FEM Z_HH t63 and the interim Z_HH t63, recomputed here."""
    cfg, pp = g26
    leaf = cfg.leaves["ring.thermal.heater_node_tau_s"]
    assert not leaf.provenance.startswith("solver")
    note = leaf.note
    hh = rom.self_res[rom.sources.index("heater"), rom.observers.index("heater_resistor")]
    rh = rom.self_res[rom.sources.index("heater"), rom.observers.index("ring_mode_weighted")]
    t_ex = _t63(_step(hh - rh, rom.taus_s)) * 1e6
    t_hh = _t63(_step(hh, rom.taus_s)) * 1e6
    assert _nums(r"excess response \(Z_HH - Z_RH, t63 ([\d.]+) us\)", note)[0] == pytest.approx(t_ex, abs=0.006)
    assert _nums(r"heater_resistor response \(t63 ([\d.]+) us\)", note)[0] == pytest.approx(t_hh, abs=0.006)
    sy = pp.thermal.system
    t = np.logspace(-8, -3, 50001)
    y = th.step_response(sy, t, obs=[sy.heater_obs[0]], src=[sy.heater_src[0]])[0, 0, 0]
    y = y / th.dc_gain(sy)[0][sy.heater_obs[0], sy.heater_src[0]]
    t63_i = float(t[np.argmax(y >= 1.0 - np.exp(-1.0))]) * 1e6
    assert _nums(r"interim Z_HH has t63 ([\d.]+) us", note)[0] == pytest.approx(t63_i, abs=0.06)
    h = float(cfg.leaves["ring.thermal.heater_self_ratio"].value)
    assert _nums(r"carries \(h - 1\)/h = ([\d.]+) % of Z_HH", note)[0] == pytest.approx(100 * (h - 1) / h, abs=0.06)


def test_architecture_5_3_states_the_thermal_model_rule():
    """physics_thermal-8: ARCHITECTURE 5.3 said 'Until the FEM artefacts exist, thermal.py builds the interim 3-pole
    model'. All six artefacts exist and g26 keeps the interim model by choice (07 build order step 3: an artefact
    replaces assumed values only through the calibration split). The contract must state the rule the presets follow."""
    text = (ROOT / "docs" / "ARCHITECTURE.md").read_text()
    sec = text[text.index("### 5.3"):text.index("### 5.4")]
    assert "Until the FEM artefacts exist" not in sec
    assert all((ROOT / "mrm" / "data" / "extracted" / f"thermal_rom_inring_noundercut_{f}.npz").exists()
               for f in (160, 200))
    assert load("g26").get("fidelity.thermal_model") == "interim"
    for phrase in ("fidelity.thermal_model", "calibration split", "interim"):
        assert phrase in sec, phrase
