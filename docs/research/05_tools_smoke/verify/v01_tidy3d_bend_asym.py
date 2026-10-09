"""Independent check: does Tidy3D's +R / -R bend asymmetry scale with the grid step (half-cell Jacobian offset)?"""
import json, time, warnings
import numpy as np
import tidy3d as td
from tidy3d.plugins.mode import ModeSolver
td.config.logging.level = "ERROR"
W, H, LAM = 0.40, 0.20, 1.31
NSI, NOX = 3.5005287119841753, 1.4468043175527276
def neff(dl, box, bend=None):
    core = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(td.inf, W, H)), medium=td.Medium(permittivity=NSI**2))
    sim = td.Simulation(size=(2.0, box[0], box[1]), grid_spec=td.GridSpec.uniform(dl=dl), structures=[core],
                        medium=td.Medium(permittivity=NOX**2), run_time=1e-12,
                        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()))
    kw = dict(num_modes=1)
    if bend is not None:
        kw.update(bend_radius=bend, bend_axis=1)
    ms = ModeSolver(simulation=sim, plane=td.Box(center=(0, 0, 0), size=(0, box[0], box[1])),
                    mode_spec=td.ModeSpec(**kw), freqs=[td.C_0 / LAM])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = ms.solve()
    return float(d.n_eff.values[0, 0].real)
out = {}
for dl, box in [(0.02, (1.62, 1.22)), (0.01, (1.61, 1.21)), (0.005, (1.605, 1.205))]:
    t = time.time()
    s = neff(dl, box); p = neff(dl, box, 7.5); m = neff(dl, box, -7.5); p50 = neff(dl, box, 50.0); m50 = neff(dl, box, -50.0)
    out[str(dl)] = dict(straight=s, plus7p5=p, minus7p5=m, asym7p5=m - p, mean_shift7p5=(p + m) / 2 - s,
                        plus50=p50, minus50=m50, asym50=m50 - p50, sec=round(time.time() - t, 1))
    print(dl, json.dumps(out[str(dl)]), flush=True)
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)  # run: NONET MG --heavy .venv/bin/python -u <this file>
