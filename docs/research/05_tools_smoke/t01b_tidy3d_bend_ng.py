"""Follow-up to t01: like-for-like Tidy3D checks.
(a) straight vs bent TE0 on the SAME 3.61 x 2.01 um plane with the same PML, so the bend shift is not
    mixed with a box change; (b) n_g with material dispersion (central difference of n_eff over
    lambda +- 5 nm with the Sellmeier indices re-evaluated), to compare with femwell's 4.336.
Run:
  .venv/bin/python tools/memguard.py --limit-gb 3 -- .venv/bin/python docs/research/05_tools_smoke/t01b_tidy3d_bend_ng.py
"""
import json, os, time, warnings
import numpy as np
import tidy3d as td
from tidy3d.plugins.mode import ModeSolver

td.config.logging.level = "ERROR"
W, H, LAM = 0.40, 0.20, 1.31


def n_si(lu):
    l2 = lu * lu
    return np.sqrt(11.6858 + 0.939816 / l2 + 0.00810461 * 1.1071**2 / (l2 - 1.1071**2))


def n_sio2(lu):
    l2 = lu * lu
    return np.sqrt(1 + 0.6961663 * l2 / (l2 - 0.0684043**2) + 0.4079426 * l2 / (l2 - 0.1162414**2)
                   + 0.8974794 * l2 / (l2 - 9.896161**2))


def neff(lam, box=(3.61, 2.01), bend=None, pml=(12, 12), dl=0.01):
    core = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(td.inf, W, H)),
                        medium=td.Medium(permittivity=n_si(lam) ** 2))
    sim = td.Simulation(size=(2.0, box[0], box[1]), grid_spec=td.GridSpec.uniform(dl=dl),
                        structures=[core], medium=td.Medium(permittivity=n_sio2(lam) ** 2),
                        run_time=1e-12, boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()))
    kw = dict(num_modes=1, num_pml=pml)
    if bend is not None:
        kw.update(bend_radius=bend, bend_axis=1)
    ms = ModeSolver(simulation=sim, plane=td.Box(center=(0, 0, 0), size=(0, box[0], box[1])),
                    mode_spec=td.ModeSpec(**kw), freqs=[td.C_0 / lam])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = ms.solve()
    return float(d.n_eff.values[0, 0].real)


t0 = time.time()
out = {"same_box_straight": neff(LAM)}
for R in (7.5, 5.0, -7.5):
    out[f"same_box_bend_R{R}"] = neff(LAM, bend=R)
dl = 0.005
npl, nmi = neff(LAM + dl, box=(2.61, 2.01), pml=(0, 0)), neff(LAM - dl, box=(2.61, 2.01), pml=(0, 0))
n0 = neff(LAM, box=(2.61, 2.01), pml=(0, 0))
out["n_g_material_dispersive"] = n0 - LAM * (npl - nmi) / (2 * dl)
out["seconds"] = round(time.time() - t0, 1)
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "out", "t01b_tidy3d_bend_ng.json"), "w"), indent=1)
