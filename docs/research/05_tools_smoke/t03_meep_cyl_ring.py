"""Smoke test: Meep FDTD in CYLINDRICAL coordinates (r, z, angular order m) for an isolated ring.

A ring with a rectangular 400 x 200 nm Si cross-section (constant n = 3.5005, the REF Sellmeier value at
1310 nm) centred on radius R = 7.5 um in SiO2 (n = 1.4468). For one angular order m the 2-D (r, z)
simulation is the exact 3-D vectorial ring (no bus): Harminv gives the resonance wavelength and the
radiation Q, and n_eff,bend = m * lambda_res / (2 pi R) is the bent-mode index referenced to the
waveguide centre. This arbitrates between the bend formulations of femwell and Tidy3D.
Run (argument = resolution in px/um):
  /Users/kenny/photonics-simulations/.meep/bin/python docs/research/05_tools_smoke/t03_meep_cyl_ring.py 50
"""
import json, os, sys, time
import numpy as np
import meep as mp

res = int(sys.argv[1]) if len(sys.argv) > 1 else 50
m = int(sys.argv[2]) if len(sys.argv) > 2 else 89
R, W, H = 7.5, 0.40, 0.20
NSI, NOX = 3.5005287, 1.4468043
dpml, pad = 1.0, 1.2
sr = R + W / 2 + pad + dpml
sz = H + 2 * (pad + dpml)
mp.verbosity(0)
fcen, df = 1 / 1.31, 0.08
geom = [mp.Block(center=mp.Vector3(R, 0, 0), size=mp.Vector3(W, mp.inf, H), material=mp.Medium(index=NSI))]
src = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Er, center=mp.Vector3(R + 0.05, 0, 0.03))]
sim = mp.Simulation(cell_size=mp.Vector3(sr, 0, sz), geometry=geom, sources=src, resolution=res,
                    boundary_layers=[mp.PML(dpml, direction=mp.R, side=mp.High), mp.PML(dpml, direction=mp.Z)],
                    dimensions=mp.CYLINDRICAL, m=m, default_material=mp.Medium(index=NOX))
h = mp.Harminv(mp.Er, mp.Vector3(R - 0.07, 0, -0.04), fcen, df)
t0 = time.time()
sim.run(mp.after_sources(h), until_after_sources=int(os.environ.get("T_AFTER", "400")))
modes = [dict(freq=md.freq, lam_um=1 / md.freq, Q=md.Q, amp=abs(md.amp)) for md in h.modes]
best = max(modes, key=lambda d: d["amp"]) if modes else None
out = dict(resolution=res, m=m, R_um=R, modes=modes, seconds=round(time.time() - t0, 1))
if best:
    out["lambda_res_um"] = best["lam_um"]
    out["Q_rad"] = best["Q"]
    out["n_eff_bend"] = m * best["lam_um"] / (2 * np.pi * R)
print(json.dumps(out, indent=1))
os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "out", f"t03_meep_cyl_res{res}_m{m}.json"), "w"), indent=1)
