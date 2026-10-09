"""Smoke test: Simphony 0.7.x (now a thin layer over SAX 0.14) - import, built-in ideal models, a ring
circuit through its sax re-export. Run:
  uv run --isolated --no-project --python 3.12 --with simphony==0.7.3 python docs/research/05_tools_smoke/t07_simphony.py
"""
import json, time
t0 = time.time()
import simphony, sax
import numpy as np
from simphony.libraries import ideal
print("simphony", simphony.__version__, "sax", sax.__version__, "ideal models:", [m for m in dir(ideal) if not m.startswith("_")][:12])
wl = np.linspace(1.30, 1.32, 2001)
ckt, _ = sax.circuit(
    netlist={"instances": {"dc": "coupler", "wg": "waveguide"},
             "connections": {"dc,o2": "wg,o0", "wg,o1": "dc,o3"},
             "ports": {"in": "dc,o0", "out": "dc,o1"}},
    models={"coupler": ideal.coupler, "waveguide": ideal.waveguide})
S = ckt(wl=wl, dc={"coupling": 0.1}, wg={"length": 47.1, "neff": 2.49, "ng": 4.34, "loss": 100.0})
T = np.abs(np.asarray(S["in", "out"])) ** 2
print(json.dumps(dict(min_T=float(T.min()), max_T=float(T.max()), seconds=round(time.time() - t0, 1))))
