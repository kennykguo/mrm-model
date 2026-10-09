"""Derived estimate: supermode splitting of two 400x200 nm Si strips in SiO2 at O-band,
to get coupling-coefficient dispersion d ln(kappa^2)/d lambda. Local tidy3d mode solver, no subpixel."""
import numpy as np, tidy3d as td, warnings
from tidy3d.plugins.mode import ModeSolver
warnings.filterwarnings("ignore")
td.config.logging.level = "ERROR"
si = td.material_library['cSi']['Li1993_293K']
sio2 = td.material_library['SiO2']['Horiba']
W, H = 0.400, 0.200
def solve(lams, gap=None, nmodes=2):
    structs = []
    if gap is None:
        structs.append(td.Structure(geometry=td.Box(center=(0,0,0), size=(td.inf, W, H)), medium=si))
        ly = 2.4
    else:
        off = (W+gap)/2
        for s in (-1,1):
            structs.append(td.Structure(geometry=td.Box(center=(0,s*off,0), size=(td.inf, W, H)), medium=si))
        ly = 2.4 + W + gap
    freqs = [td.C_0/l for l in lams]
    sim = td.Simulation(size=(1, ly, 2.0), center=(0,0,0), medium=sio2, structures=structs,
        grid_spec=td.GridSpec.uniform(dl=0.0125), run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        sources=[], monitors=[])
    plane = td.Box(center=(0,0,0), size=(0, ly, 2.0))
    ms = ModeSolver(simulation=sim, plane=plane, freqs=freqs,
        mode_spec=td.ModeSpec(num_modes=nmodes, target_neff=3.0, group_index_step=True if gap is None else False))
    d = ms.solve()
    return d
lams = [1.290, 1.310, 1.330]
out = {}
d = solve(lams)
neff = d.n_eff.values[:,0]; ng = d.n_group.values[:,0] if d.n_group is not None else None
out['single'] = {'lams': lams, 'neff': neff.tolist(), 'ng': None if ng is None else ng.tolist()}
print('single neff', neff, 'ng', ng)
for gap in (0.15, 0.20, 0.25):
    d = solve(lams, gap=gap, nmodes=4)
    n = d.n_eff.values  # (freq, mode)
    # pick the two TE-like supermodes: use te_fraction if available
    tef = d.pol_fraction.te.values if hasattr(d, 'pol_fraction') else None
    rows = []
    for i,l in enumerate(lams):
        idx = [m for m in range(n.shape[1]) if tef is None or tef[i,m] > 0.5]
        ne, no = sorted([n[i,m] for m in idx], reverse=True)[:2]
        C = np.pi*(ne-no)/l  # 1/um
        rows.append((l, ne, no, C))
    out[str(gap)] = rows
    print('tef', None if tef is None else tef.round(2).tolist()); print('gap', gap, [(r[0], round(r[1],5), round(r[2],5), round(r[3],5)) for r in rows])
