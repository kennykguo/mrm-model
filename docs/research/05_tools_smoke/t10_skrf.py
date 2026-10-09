"""Smoke test: scikit-rf - cascade of 2-port Networks (an electrical driver/bond-wire model would live here).
Run:  uv run --isolated --no-project --python 3.12 --with scikit-rf==2.1.0 python docs/research/05_tools_smoke/t10_skrf.py
"""
import json
import numpy as np, skrf as rf
f = rf.Frequency(1, 60, 601, "GHz")
line = rf.media.DefinedGammaZ0(frequency=f, z0_port=50, z0=50).line(2, "mm")
c = rf.media.DefinedGammaZ0(frequency=f, z0_port=50).shunt_capacitor(50e-15)
n = line ** c ** line
print(json.dumps(dict(skrf=rf.__version__, s21_dB_at_50GHz=float(n.s_db[500, 1, 0]), passive=bool(np.all(np.abs(n.s) <= 1 + 1e-12)))))
