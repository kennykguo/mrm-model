"""Thermo-optic weights of an SOI waveguide mode from a full-vector FEM mode solver (femwell).

Computes, for the fundamental TE-like mode at 1310 nm:
  n_eff, n_g (with material dispersion), d n_eff / d n_Si, d n_eff / d n_ox (finite differences),
  the power fraction in Si (Poynting), and the sensitivity d lambda / d T for several dn_Si/dT values.
Checks the exact identity  sum_i n_g,i * d n_eff / d n_i = n_g.

Run (light job, about 1-2 GB peak):
  .venv/bin/python tools/memguard.py --limit-gb 3 -- .venv/bin/python docs/research/scripts/06_thermo_optic_weights.py strip500
Arguments: strip500 | strip400 | rib | bent | all.  Written for docs/research/06_textbook_audit.md.
"""
import sys
from collections import OrderedDict
import numpy as np
import shapely
from shapely.geometry import box
from skfem import Basis, ElementTriP0
from skfem.io.meshio import from_meshio
from femwell.mesh import mesh_from_OrderedDict
from femwell.maxwell.waveguide import compute_modes


def n_si(lam_um):
    # H. H. Li (1980) room-temperature fit, the same form the textbook uses (nSilicon in src/physics/index.ts)
    l2 = lam_um ** 2
    eps = 11.6858 + 0.939816 / l2 + (0.00810461 * 1.1071 ** 2) / (l2 - 1.1071 ** 2)
    return np.sqrt(eps)


def n_ox(lam_um):
    l2 = lam_um ** 2
    eps = 1 + 0.6961663 * l2 / (l2 - 0.0684043 ** 2) + 0.4079426 * l2 / (l2 - 0.1162414 ** 2) + 0.8974794 * l2 / (l2 - 9.896161 ** 2)
    return np.sqrt(eps)


def build(w, h, slab=0.0, slab_w=4.0):
    core = box(-w / 2, 0, w / 2, h)
    polys = OrderedDict(core=core)
    if slab > 0:
        polys["slab"] = box(-slab_w / 2, 0, slab_w / 2, slab)
    polys["clad"] = box(-3.0, -2.0, 3.0, 2.5)
    res = {"core": {"resolution": 0.02, "distance": 0.6}}
    if slab > 0:
        res["slab"] = {"resolution": 0.03, "distance": 0.5}
    mesh = from_meshio(mesh_from_OrderedDict(polys, res, default_resolution_max=0.3))
    return mesh


def solve(mesh, lam, nsi, nox, radius=np.inf):
    basis0 = Basis(mesh, ElementTriP0())
    eps = basis0.zeros(dtype=complex)
    for sub, n in (("core", nsi), ("slab", nsi), ("clad", nox)):
        if sub in mesh.subdomains:
            eps[basis0.get_dofs(elements=sub)] = n ** 2
    modes = compute_modes(basis0, eps, wavelength=lam, num_modes=2, order=2, radius=radius)
    m = max(modes, key=lambda x: x.te_fraction * (x.n_eff.real))
    return m


def analyse(name, w, h, slab=0.0, radius=np.inf, lam=1.31):
    mesh = build(w, h, slab)
    nsi, nox = n_si(lam), n_ox(lam)
    m0 = solve(mesh, lam, nsi, nox, radius)
    ne0 = m0.n_eff.real
    d = 2e-3
    dSi = (solve(mesh, lam, nsi + d, nox, radius).n_eff.real - solve(mesh, lam, nsi - d, nox, radius).n_eff.real) / (2 * d)
    dOx = (solve(mesh, lam, nsi, nox + d, radius).n_eff.real - solve(mesh, lam, nsi, nox - d, radius).n_eff.real) / (2 * d)
    dl = 0.005
    nep = solve(mesh, lam + dl, n_si(lam + dl), n_ox(lam + dl), radius).n_eff.real
    nem = solve(mesh, lam - dl, n_si(lam - dl), n_ox(lam - dl), radius).n_eff.real
    ng = ne0 - lam * (nep - nem) / (2 * dl)
    # material group indices
    ngsi = nsi - lam * (n_si(lam + dl) - n_si(lam - dl)) / (2 * dl)
    ngox = nox - lam * (n_ox(lam + dl) - n_ox(lam - dl)) / (2 * dl)
    ident = ngsi * dSi + ngox * dOx
    # power fraction in silicon
    elems_si = [s for s in ("core", "slab") if s in mesh.subdomains]
    p_tot = m0.calculate_power().real
    p_si = sum(m0.calculate_power(elements=s).real for s in elems_si)
    gamma_power = p_si / p_tot
    w_si = ngsi * dSi / ng
    w_ox = ngox * dOx / ng
    print(f"\n== {name}: w={w} h={h} slab={slab} radius={radius} lambda={lam} um")
    print(f"n_Si={nsi:.4f} n_g,Si={ngsi:.4f} n_ox={nox:.4f} n_g,ox={ngox:.4f}")
    print(f"n_eff={ne0:.4f}  n_g={ng:.4f}  TE fraction={m0.te_fraction:.3f}")
    print(f"dneff/dnSi={dSi:.4f}  dneff/dnOx={dOx:.4f}  power fraction in Si={gamma_power:.4f}")
    print(f"identity: n_gSi*dSi + n_gOx*dOx = {ident:.4f}  vs n_g = {ng:.4f}  (ratio {ident/ng:.4f})")
    print(f"energy-like weights w_Si={w_si:.4f} w_ox={w_ox:.4f} sum={w_si + w_ox:.4f}")
    for dndt_si in (1.86e-4, 1.94e-4, 2.15e-4):
        dldt = lam * 1e3 / ng * (dSi * dndt_si + dOx * 0.95e-5) * 1e3  # pm/K
        cap = lam * 1e3 * dndt_si / ngsi * 1e3
        naive = lam * 1e3 / ng * (gamma_power * dndt_si + (1 - gamma_power) * 0.95e-5) * 1e3
        print(f"  dnSi/dT={dndt_si:.2e}: dlambda/dT={dldt:.1f} pm/K | power-Gamma formula gives {naive:.1f} | cap lambda*dn/dT/n_gSi={cap:.1f}")
    sys.stdout.flush()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "strip500"):
        analyse("strip 500x220", 0.5, 0.22)
    if which in ("all", "strip400"):
        analyse("strip 400x200 (Lightmatter SMTAI)", 0.4, 0.2)
    if which in ("all", "rib"):
        analyse("rib 500x220 / 90 nm slab", 0.5, 0.22, slab=0.09)
    if which in ("all", "bent"):
        # Conformal-map bend. (A bent RIB with a 4 um slab at R = 5 um leaks into the slab; not reported.)
        analyse("strip 500x220, R=5 um", 0.5, 0.22, radius=5.0)
        analyse("strip 400x200, R=5 um", 0.4, 0.2, radius=5.0)
