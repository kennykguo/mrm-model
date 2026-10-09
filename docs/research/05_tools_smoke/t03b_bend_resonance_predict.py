"""Predict the m = 89 resonance of the isolated R = 7.5 um ring of t03 from bent-mode solvers, with the
same CONSTANT indices as the Meep run (n_Si = 3.5005287, n_SiO2 = 1.4468043, no material dispersion):
lambda_m solves 2 pi R n_eff,bend(lambda) / lambda = m. femwell (conformal eps (1 + x/R)^2) and Tidy3D
(exact radial transform with Jacobians on E and H), Tidy3D for both signs of bend_radius.
Run:
  .venv/bin/python tools/memguard.py --limit-gb 3 -- .venv/bin/python docs/research/05_tools_smoke/t03b_bend_resonance_predict.py
"""
import json, os, warnings
from collections import OrderedDict
import numpy as np

warnings.simplefilter("ignore")
NSI, NOX, W, H, R, M = 3.5005287, 1.4468043, 0.40, 0.20, 7.5, 89
LAMS = (1.312, 1.316)
out = {}

import shapely
from skfem import Basis, ElementTriP0
from skfem.io.meshio import from_meshio
from femwell.mesh import mesh_from_OrderedDict
from femwell.maxwell.waveguide import compute_modes

mesh = from_meshio(mesh_from_OrderedDict(
    OrderedDict(core=shapely.box(-W / 2, -H / 2, W / 2, H / 2), clad=shapely.box(-1.5, -1.0, 2.5, 1.0)),
    {"core": {"resolution": 0.01, "distance": 0.5}}, default_resolution_max=0.2))
b0 = Basis(mesh, ElementTriP0())
eps = b0.zeros() + NOX**2
eps[b0.get_dofs(elements="core")] = NSI**2


def fem(lam, radius):
    ms = compute_modes(b0, eps, wavelength=lam, num_modes=2, order=2, radius=radius)
    return max((m for m in ms if m.te_fraction > 0.5), key=lambda m: m.n_eff.real).n_eff.real


import tidy3d as td
from tidy3d.plugins.mode import ModeSolver
td.config.logging.level = "ERROR"


def t3d(lam, radius, dl=0.01, box=(3.61, 2.01)):
    core = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(td.inf, W, H)), medium=td.Medium(permittivity=NSI**2))
    sim = td.Simulation(size=(2.0, *box), grid_spec=td.GridSpec.uniform(dl=dl), structures=[core],
                        medium=td.Medium(permittivity=NOX**2), run_time=1e-12,
                        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()))
    kw = dict(num_modes=1, num_pml=(12, 12))
    if radius is not None:
        kw.update(bend_radius=radius, bend_axis=1)
    ms = ModeSolver(simulation=sim, plane=td.Box(center=(0, 0, 0), size=(0, *box)), mode_spec=td.ModeSpec(**kw),
                    freqs=[td.C_0 / lam])
    return float(ms.solve().n_eff.values[0, 0].real)


def lam_m(fn):
    vals = {lam: fn(lam) for lam in LAMS}
    g = {lam: 2 * np.pi * R * vals[lam] / lam - M for lam in LAMS}   # root of g(lambda)
    l0, l1 = LAMS
    lam = l0 - g[l0] * (l1 - l0) / (g[l1] - g[l0])
    return dict(n_eff=vals, lambda_m89_um=float(lam), n_eff_at_lambda_m=float(M * lam / (2 * np.pi * R)))


out["femwell_straight"] = lam_m(lambda l: fem(l, np.inf))
out["femwell_bend"] = lam_m(lambda l: fem(l, R))
out["tidy3d_straight"] = lam_m(lambda l: t3d(l, None))
out["tidy3d_bend_plus"] = lam_m(lambda l: t3d(l, R))
out["tidy3d_bend_minus"] = lam_m(lambda l: t3d(l, -R))
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "out", "t03b_bend_resonance_predict.json"), "w"), indent=1, default=str)
