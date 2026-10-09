"""Smoke test: femwell (FEM, scikit-fem + gmsh) on the same 400 x 200 nm Si strip at 1310 nm as t01.

1. straight TE0: n_eff, n_g (central difference +-5 nm with Sellmeier materials), power fraction in core,
   dn_eff/dn_Si sensitivity (calculate_confinement_factor) and dn_eff/dT by a +10 K re-solve;
2. bent TE0 at R = 7.5 um and 5 um (femwell's radius= keyword: conformal eps*(1+x/R)^2 mapping);
3. ring-bus coupling: even/odd supermodes of two parallel 400 x 200 nm guides at four gaps and three
   wavelengths -> kappa(g, lambda) = pi (n_even - n_odd) / lambda, then a point-coupler integral
   kappa_tot = int kappa(g0 + z^2/(2R)) dz for R = 7.5 um, g0 = 0.20 um (both ASSUMED, illustration only).
Run:
  .venv/bin/python tools/memguard.py --limit-gb 3 -- .venv/bin/python docs/research/05_tools_smoke/t02_femwell_modes.py
"""
import json, os, time, warnings
from collections import OrderedDict
import numpy as np
import shapely
from skfem import Basis, ElementTriP0
from skfem.io.meshio import from_meshio
from femwell.mesh import mesh_from_OrderedDict
from femwell.maxwell.waveguide import compute_modes

warnings.simplefilter("ignore")


def n_si(lu):
    l2 = lu * lu
    return np.sqrt(11.6858 + 0.939816 / l2 + 0.00810461 * 1.1071**2 / (l2 - 1.1071**2))


def n_sio2(lu):
    l2 = lu * lu
    return np.sqrt(1 + 0.6961663 * l2 / (l2 - 0.0684043**2) + 0.4079426 * l2 / (l2 - 0.1162414**2)
                   + 0.8974794 * l2 / (l2 - 9.896161**2))


W, H, LAM = 0.40, 0.20, 1.31
t0 = time.time()


def mesh(polys, res=0.01, box=(-1.5, -1.0, 1.5, 1.0), rmax=0.2):
    polys = OrderedDict(list(polys.items()) + [("clad", shapely.box(*box))])
    resd = {k: {"resolution": res, "distance": 0.5} for k in polys if k != "clad"}
    m = from_meshio(mesh_from_OrderedDict(polys, resd, default_resolution_max=rmax))
    return m


def eps_on(m, lam, dT=0.0, cores=("core",)):
    b0 = Basis(m, ElementTriP0())
    eps = b0.zeros() + (n_sio2(lam) + 1e-5 * dT) ** 2
    for c in cores:
        eps[b0.get_dofs(elements=c)] = (n_si(lam) + 1.86e-4 * dT) ** 2
    return b0, eps


def te0(modes):
    te = [m for m in modes if m.te_fraction > 0.5]
    return max(te, key=lambda m: m.n_eff.real)


out = {}
m1 = mesh({"core": shapely.box(-W / 2, -H / 2, W / 2, H / 2)})
out["straight_mesh_triangles"] = int(m1.nelements)


def neff_straight(lam, dT=0.0, radius=np.inf, mm=m1):
    b0, eps = eps_on(mm, lam, dT)
    modes = compute_modes(b0, eps, wavelength=lam, num_modes=2, order=2, radius=radius)
    return te0(modes)


t1 = time.time()
mo = neff_straight(LAM)
n0 = float(mo.n_eff.real)
out["straight"] = dict(n_eff=n0, te_fraction=float(mo.te_fraction),
                       power_fraction_core=float(np.real(mo.calculate_power(elements="core"))),
                       dneff_dnSi=float(np.real(mo.calculate_confinement_factor("core"))),
                       seconds_per_solve=round(time.time() - t1, 1))
dl = 0.005
npl, nmi = float(neff_straight(LAM + dl).n_eff.real), float(neff_straight(LAM - dl).n_eff.real)
out["straight"]["n_g_material_dispersive"] = n0 - LAM * (npl - nmi) / (2 * dl)
out["straight"]["dneff_dT_per_K"] = (float(neff_straight(LAM, dT=10).n_eff.real) - n0) / 10
out["straight"]["dlambda_dT_pm_per_K"] = LAM * 1e6 * out["straight"]["dneff_dT_per_K"] / out["straight"]["n_g_material_dispersive"]
# mesh convergence: 5 nm elements in the core
m_fine = mesh({"core": shapely.box(-W / 2, -H / 2, W / 2, H / 2)}, res=0.005)
out["straight"]["n_eff_core_mesh_5nm"] = float(neff_straight(LAM, mm=m_fine).n_eff.real)

# bends (wider box on the outer side; the conformal map pushes the mode to +x)
mb = mesh({"core": shapely.box(-W / 2, -H / 2, W / 2, H / 2)}, box=(-1.5, -1.0, 2.5, 1.0))
out["bend_box_straight_n_eff"] = float(neff_straight(LAM, mm=mb).n_eff.real)
for R in (7.5, 5.0):
    out[f"bend_R{R}_n_eff"] = float(neff_straight(LAM, radius=R, mm=mb).n_eff.real)

# coupler supermodes: kappa(gap, lambda)
gaps = [0.15, 0.20, 0.25, 0.30]
lams = [1.29, 1.31, 1.33]
kap = {}
for g in gaps:
    xa = g / 2 + W / 2
    mc = mesh({"core_a": shapely.box(-xa - W / 2, -H / 2, -xa + W / 2, H / 2),
               "core_b": shapely.box(xa - W / 2, -H / 2, xa + W / 2, H / 2)}, res=0.015,
              box=(-2.0, -1.0, 2.0, 1.0))
    for lam in lams:
        b0, eps = eps_on(mc, lam, cores=("core_a", "core_b"))
        modes = compute_modes(b0, eps, wavelength=lam, num_modes=4, order=2)
        te = sorted([m.n_eff.real for m in modes if m.te_fraction > 0.5], reverse=True)[:2]
        kap[(g, lam)] = float(np.pi * (te[0] - te[1]) / lam)   # rad/um
out["kappa_rad_per_um"] = {f"g{g}_l{lam}": kap[(g, lam)] for (g, lam) in kap}

R, g0 = 7.5, 0.20
z = np.linspace(-4, 4, 4001)
res = {}
for lam in lams:
    gk = np.array(gaps)
    kk = np.array([kap[(g, lam)] for g in gaps])
    p = np.polyfit(gk, np.log(kk), 1)              # ln kappa = p0 g + p1
    gz = g0 + R - np.sqrt(R**2 - z**2)               # gap between straight bus and circular ring
    ktot = np.trapezoid(np.exp(np.polyval(p, gz)), z)
    res[lam] = dict(gamma_per_um=-p[0], kappa_tot_rad=float(ktot), K2_power=float(np.sin(ktot) ** 2))
out["point_coupler_R7.5_g0.20"] = {str(k): v for k, v in res.items()}
out["dK2_dlambda_per_nm"] = (res[1.33]["K2_power"] - res[1.29]["K2_power"]) / 40.0
out["total_seconds"] = round(time.time() - t0, 1)
print(json.dumps(out, indent=1))
os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "out", "t02_femwell_modes.json"), "w"), indent=1)
