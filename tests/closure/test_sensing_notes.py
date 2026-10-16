"""The g26 sensing keys say what the code does (physics_sensing review, 2026-10-16).

- physics_sensing-4: ring.monitor.background_fraction multiplies the TOTAL bus power at the ring (all lines), not one
  line's power; its g26 range must keep every nominal drop peak plus the background below the ADC full scale.
- physics_sensing-7: the g26 junction_model comment named the junction LUT as missing while
  mrm/data/extracted/junction_rib160.json exists.
"""
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np

from mrm.params import load, realize
from mrm.params import schema as sch

ROOT = Path(__file__).resolve().parents[2]
G26 = ROOT / "mrm" / "params" / "presets" / "g26.yaml"


def _nominal():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = load("g26")
        return cfg, realize(cfg, 1, seeds=0, uncertainty="nominal", variation="off")


def test_background_fraction_doc_names_the_total_bus_power():
    """The schema doc (and the generated key appendix) said 'fraction of the line power'; the kernel applies the
    fraction to the sum of every line's bus power at the ring (3-8 x one line in g26)."""
    doc = sch.lookup("ring.monitor.background_fraction").doc
    assert "line power" not in doc
    assert "total bus power" in doc and "all lines" in doc
    cfg, _ = _nominal()
    assert "TOTAL bus power" in cfg.leaves["ring.monitor.background_fraction"].note


def test_background_fraction_range_keeps_the_monitor_in_range():
    """At the top of the g26 range, the nominal drop peak of every ring plus the background (bounded with the whole
    chip input on the bus at every ring, no bus loss: conservative) stays below the ADC full scale, so a stray-light
    study inside the range does not saturate the monitor (the former top 0.01 put 69-191 uA on a 40 uA scale)."""
    cfg, pp = _nominal()
    top = float(cfg.leaves["ring.monitor.background_fraction"].range[1])
    o, m = pp.optics, pp.monitor
    fs = float(m.adc_full_scale_a[0, 0])
    gs = float(m.gain_scale[0, 0])
    p_bus = float(np.sum(pp.laser.power_w[0] * pp.laser.input_transmission[0, 0]))
    for i in range(pp.n_rings):
        k1, k2 = float(o.kappa2_through[0, i]), float(o.kappa2_drop[0, i])
        a = float(o.round_trip_amplitude[0, i])
        t1, t2 = math.sqrt(1.0 - k1), math.sqrt(1.0 - k2)
        drop = k1 * k2 * a / (1.0 - a * t1 * t2) ** 2
        line = int(pp.layout.ring_to_line[i])
        p_in = float(pp.laser.power_w[0, line] * pp.laser.input_transmission[0, 0])
        r = float(m.responsivity_a_per_w[0, i])
        i_peak = gs * r * drop * p_in
        assert i_peak + gs * r * top * p_bus < fs, (i, 1e6 * i_peak, 1e6 * gs * r * top * p_bus)


def test_g26_junction_model_comment_matches_the_artefacts():
    """The comment on fidelity.junction_model must not call the junction LUT missing when it is on disk; it states
    why g26 keeps the analytic law instead."""
    lines = G26.read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if re.match(r"\s+junction_model:\s", ln))
    block = [lines[i]]
    for ln in lines[i + 1:]:                       # continuation comment lines
        if ln.strip().startswith("#"):
            block.append(ln)
        else:
            break
    text = " ".join(block)
    assert "not extracted" not in text
    named = re.findall(r"junction_[a-z0-9]+\.json", text)
    assert named, text
    for f in named:
        assert (ROOT / "mrm" / "data" / "extracted" / f).exists(), f
    assert "film_thickness" in text and "analytic" in text
