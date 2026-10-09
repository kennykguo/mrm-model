"""Derived: d n_eff / d n_Si and d n_eff / d n_SiO2 for a 400x200 nm Si strip in SiO2 at 1310 nm
(local tidy3d mode solver, grid aligned to interfaces, no subpixel)."""
import numpy as np, tidy3d as td, warnings
from tidy3d.plugins.mode import ModeSolver
warnings.filterwarnings("ignore"); td.config.logging.level = "ERROR"
lam = 1.31
nsi0 = float(np.real(td.material_library['cSi']['Li1993_293K'].nk_model(td.C_0/lam)[0]))
nox0 = float(np.real(td.material_library['SiO2']['Horiba'].nk_model(td.C_0/lam)[0]))
print('n_Si(1310)=%.4f n_SiO2(1310)=%.4f' % (nsi0, nox0))
def neff(nsi, nox, W=0.4, H=0.2):
    s = td.Structure(geometry=td.Box(center=(0,0,0), size=(td.inf, W, H)), medium=td.Medium(permittivity=nsi**2))
    sim = td.Simulation(size=(1, 2.4, 2.0), medium=td.Medium(permittivity=nox**2), structures=[s],
        grid_spec=td.GridSpec.uniform(dl=0.0125), run_time=1e-12, sources=[], monitors=[],
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()))
    ms = ModeSolver(simulation=sim, plane=td.Box(size=(0, 2.4, 2.0)), freqs=[td.C_0/lam],
        mode_spec=td.ModeSpec(num_modes=1, target_neff=3.0))
    return float(ms.solve().n_eff.values[0,0])
n0 = neff(nsi0, nox0)
d = 0.01
dsi = (neff(nsi0+d, nox0) - neff(nsi0-d, nox0))/(2*d)
dox = (neff(nsi0, nox0+d) - neff(nsi0, nox0-d))/(2*d)
print('neff=%.4f  dneff/dnSi=%.4f  dneff/dnSiO2=%.4f' % (n0, dsi, dox))
ng = 4.33
for dnsidT in (1.86e-4, 2.0e-4, 2.2e-4):
    dndT = dsi*dnsidT + dox*0.95e-5
    print('dnSi/dT=%.2e -> dneff/dT=%.3e -> dlam/dT=%.1f pm/K (ng=%.2f)' % (dnsidT, dndT, 1310*dndT/ng*1e3, ng))
