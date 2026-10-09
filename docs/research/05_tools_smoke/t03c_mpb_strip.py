"""Smoke test: MPB (plane-wave eigensolver, in the conda-forge Meep env) on the 400 x 200 nm Si strip at
1310 nm with the REF Sellmeier indices held constant (n_Si = 3.5005287, n_SiO2 = 1.4468043): find_k gives
n_eff at fixed frequency, and the group velocity gives the waveguide-only n_g (no material dispersion),
to compare with Tidy3D's built-in n_group (4.1425) and femwell. Resolutions 32, 64, 128 px/um.
Run:
  /Users/kenny/photonics-simulations/.meep/bin/python docs/research/05_tools_smoke/t03c_mpb_strip.py
"""
import json, os, time
import meep as mp
from meep import mpb

NSI, NOX, W, H, LAM = 3.5005287, 1.4468043, 0.40, 0.20, 1.31
out = {}
for res in (32, 64, 128):
    t0 = time.time()
    ms = mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(0, 3.0, 2.0)),
                        geometry=[mp.Block(size=mp.Vector3(mp.inf, W, H), material=mp.Medium(index=NSI))],
                        default_material=mp.Medium(index=NOX), resolution=res, num_bands=1)
    mpb.verbosity(0)
    f = 1 / LAM
    k = ms.find_k(mp.NO_PARITY, f, 1, 1, mp.Vector3(1, 0, 0), 1e-6, f * 2.5, f * 1.5, f * 3.5)
    out[res] = dict(n_eff=k[0] / f, seconds=round(time.time() - t0, 1))
    try:
        ms.k_points = [mp.Vector3(k[0], 0, 0)]
        ms.run()
        v = ms.compute_group_velocity_component(mp.Vector3(1, 0, 0))[0]
        out[res]["n_g_wg"] = 1 / v
    except Exception as e:  # noqa
        out[res]["n_g_error"] = repr(e)
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "out", "t03c_mpb_strip.json"), "w"), indent=1)
