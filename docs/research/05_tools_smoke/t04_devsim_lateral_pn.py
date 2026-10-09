"""Smoke test: DEVSIM drift-diffusion of a 1-D lateral PN junction across a ring waveguide (the cut
through the rib core along the width). ASSUMED illustration values: N_A = N_D = 5e17 cm^-3 abrupt
junction at x = 0, contacts at +-0.5 um (in a real rib the contacts sit on P++/N++ slab regions further
out; a 1-D cut ignores the slab and the 2-D field spreading). Biases 0 to -2 V (reverse).

Reports, per bias: the depletion width from the simulated hole/electron profiles (where the majority
density falls below half its doping) against the abrupt-junction depletion approximation, and the
change of integrated electrons and holes inside the 400 nm core |x| < 0.2 um (cm^-2), which is what a
plasma-dispersion model (Soref-Bennett or Nedeljkovic) turns into delta n_eff.
Run:
  .venv/bin/python tools/memguard.py --limit-gb 3 -- .venv/bin/python docs/research/05_tools_smoke/t04_devsim_lateral_pn.py
"""
import json, os, time
import numpy as np
import devsim as ds
from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
from devsim.python_packages.simple_physics import (
    GetContactBiasName, SetSiliconParameters, CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact, CreateSiliconDriftDiffusion, CreateSiliconDriftDiffusionAtContact)

t0 = time.time()
dev, reg = "pn", "si"
NA = ND = 5e17
L = 0.5e-4  # cm
ds.create_1d_mesh(mesh="m")
ds.add_1d_mesh_line(mesh="m", pos=-L, ps=2e-7, tag="pc")
ds.add_1d_mesh_line(mesh="m", pos=0.0, ps=2e-8, tag="j")
ds.add_1d_mesh_line(mesh="m", pos=L, ps=2e-7, tag="nc")
ds.add_1d_contact(mesh="m", name="p", tag="pc", material="metal")
ds.add_1d_contact(mesh="m", name="n", tag="nc", material="metal")
ds.add_1d_region(mesh="m", material="Si", region=reg, tag1="pc", tag2="nc")
ds.finalize_mesh(mesh="m")
ds.create_device(mesh="m", device=dev)
SetSiliconParameters(dev, reg, 300)
CreateNodeModel(dev, reg, "Acceptors", f"{NA}*step(-x)")
CreateNodeModel(dev, reg, "Donors", f"{ND}*step(x)")
CreateNodeModel(dev, reg, "NetDoping", "Donors-Acceptors")
CreateSolution(dev, reg, "Potential")
CreateSiliconPotentialOnly(dev, reg)
for c in ("p", "n"):
    ds.set_parameter(device=dev, name=GetContactBiasName(c), value=0.0)
    CreateSiliconPotentialOnlyContact(dev, reg, c)
ds.solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=40)
CreateSolution(dev, reg, "Electrons")
CreateSolution(dev, reg, "Holes")
ds.set_node_values(device=dev, region=reg, name="Electrons", init_from="IntrinsicElectrons")
ds.set_node_values(device=dev, region=reg, name="Holes", init_from="IntrinsicHoles")
CreateSiliconDriftDiffusion(dev, reg)
for c in ("p", "n"):
    CreateSiliconDriftDiffusionAtContact(dev, reg, c)
ds.solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=40)

x = np.array(ds.get_node_model_values(device=dev, region=reg, name="x"))
q, eps_si, kT = 1.602176634e-19, 11.7 * 8.8541878128e-14, 0.025852
ni = 1.0e10
vbi = kT * np.log(NA * ND / ni**2)
core = np.abs(x) <= 0.2e-4
rows, ref = [], None
for v in np.round(np.arange(0.0, -2.01, -0.25), 2):
    ds.set_parameter(device=dev, name=GetContactBiasName("p"), value=float(v))
    ds.solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=40)
    n = np.array(ds.get_node_model_values(device=dev, region=reg, name="Electrons"))
    p = np.array(ds.get_node_model_values(device=dev, region=reg, name="Holes"))
    xp = x[(x < 0) & (p >= NA / 2)].max()          # p-side depletion edge
    xn = x[(x > 0) & (n >= ND / 2)].min()          # n-side depletion edge
    w_sim = (xn - xp) * 1e4
    w_dep = np.sqrt(2 * eps_si * (vbi - v) / q * (1 / NA + 1 / ND)) * 1e4
    Ne, Ph = np.trapezoid(n[core], x[core]), np.trapezoid(p[core], x[core])
    if ref is None:
        ref = (Ne, Ph)
    rows.append(dict(V=float(v), W_sim_um=float(w_sim), W_depletion_approx_um=float(w_dep),
                     dN_core_cm2=float(Ne - ref[0]), dP_core_cm2=float(Ph - ref[1])))
out = dict(devsim=ds.__version__ if hasattr(ds, "__version__") else "?", nodes=int(x.size),
           Vbi_V=float(vbi), rows=rows, seconds=round(time.time() - t0, 2))
print(json.dumps(out, indent=1))
os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "out", "t04_devsim_lateral_pn.json"), "w"), indent=1)
