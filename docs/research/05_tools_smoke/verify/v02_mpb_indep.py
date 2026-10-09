"""Independent MPB check of the 400 x 200 nm Si strip in SiO2 at 1.31 um (written by the checker, not reused)."""
import json, sys
import numpy as np
import meep as mp
from meep import mpb
def nsi(l): l2=l*l; return np.sqrt(11.6858+0.939816/l2+0.00810461*1.1071**2/(l2-1.1071**2))
def nox(l): l2=l*l; return np.sqrt(1+0.6961663*l2/(l2-0.0684043**2)+0.4079426*l2/(l2-0.1162414**2)+0.8974794*l2/(l2-9.896161**2))
mp.verbosity(0)
def solve(lam, ncore, nclad, res, sx=3.0, sy=2.5):
    geom=[mp.Block(size=mp.Vector3(0.40,0.20,mp.inf), material=mp.Medium(index=ncore))]
    ms=mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(sx,sy,0)), geometry=geom,
                      default_material=mp.Medium(index=nclad), resolution=res, num_bands=1)
    f=1/lam
    k=ms.find_k(mp.NO_PARITY, f, 1, 1, mp.Vector3(0,0,1), 1e-9, f*2.4, f*1.5, f*3.5)
    vg=ms.compute_group_velocities()  # not used; group index from find_k output
    return k[0]/f
out={}
res=int(sys.argv[1]) if len(sys.argv)>1 else 96
L=1.31
n0=solve(L,nsi(L),nox(L),res)
nT=solve(L,nsi(L)+1.86e-4*10,nox(L)+1e-5*10,res)
d=0.005
# waveguide-only n_g (constant indices) and with Sellmeier dispersion
np_c=solve(L+d,nsi(L),nox(L),res); nm_c=solve(L-d,nsi(L),nox(L),res)
np_s=solve(L+d,nsi(L+d),nox(L+d),res); nm_s=solve(L-d,nsi(L-d),nox(L-d),res)
out=dict(res=res, n_eff=n0, dneff_dT=(nT-n0)/10, ng_wg=n0-L*(np_c-nm_c)/(2*d), ng_disp=n0-L*(np_s-nm_s)/(2*d))
out["dlam_dT_pm"]=1310*out["dneff_dT"]/out["ng_disp"]*1e3
print(json.dumps(out))
