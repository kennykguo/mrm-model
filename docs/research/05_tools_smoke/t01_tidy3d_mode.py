"""Smoke test: Tidy3D LOCAL mode solver (no account, no API key, no network).

Lightmatter strip cross-section 400 x 200 nm (printed, SMTA Oct 2025, via REF.waveguideLmNm), Si core in
SiO2, lambda = 1310 nm. Materials: the REF Sellmeier fits (nSilicon, nSilica in src/physics/index.ts) at
1.31 um. Thermo-optic: REF dnSi_dT = 1.86e-4 /K, dnSiO2_dT = 1e-5 /K (both textbook values).

Reports: TE0 n_eff, n_g (Tidy3D built-in group_index_step), dn_eff/dT by a +10 K re-solve, and the bent
TE0 at R = 7.5 um (Lightmatter slides print a ~15 um ring diameter) with PML to get radiation loss.
Run (light job):
  .venv/bin/python tools/memguard.py --limit-gb 3 -- .venv/bin/python docs/research/05_tools_smoke/t01_tidy3d_mode.py
"""
import json, os, time, warnings
import numpy as np


def n_si(lu):  # REF nSilicon
    l2 = lu * lu
    return np.sqrt(11.6858 + 0.939816 / l2 + 0.00810461 * 1.1071**2 / (l2 - 1.1071**2))


def n_sio2(lu):  # REF nSilica
    l2 = lu * lu
    return np.sqrt(1 + 0.6961663 * l2 / (l2 - 0.0684043**2) + 0.4079426 * l2 / (l2 - 0.1162414**2)
                   + 0.8974794 * l2 / (l2 - 9.896161**2))


LAM, W, H = 1.31, 0.40, 0.20
NSI, NOX = n_si(LAM), n_sio2(LAM)

t0 = time.time()
import tidy3d as td
from tidy3d.plugins.mode import ModeSolver
td.config.logging.level = "ERROR"
print("tidy3d", td.__version__, "import", round(time.time() - t0, 1), "s")


def solve(nsi, nox, dl=0.01, box=(2.61, 2.01), bend=None, num_pml=(0, 0), num_modes=2, ng=True):
    core = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(td.inf, W, H)),
                        medium=td.Medium(permittivity=nsi**2))
    sim = td.Simulation(size=(2.0, box[0], box[1]), grid_spec=td.GridSpec.uniform(dl=dl),
                        structures=[core], medium=td.Medium(permittivity=nox**2), run_time=1e-12,
                        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()))
    kw = dict(num_modes=num_modes, group_index_step=ng, num_pml=num_pml)
    if bend is not None:
        kw.update(bend_radius=bend, bend_axis=1)
    ms = ModeSolver(simulation=sim, plane=td.Box(center=(0, 0, 0), size=(0, box[0], box[1])),
                    mode_spec=td.ModeSpec(**kw), freqs=[td.C_0 / LAM])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ms.solve()


out = {"n_si": float(NSI), "n_sio2": float(NOX)}
t1 = time.time()
d = solve(NSI, NOX)
out["straight"] = dict(n_eff=float(d.n_eff.values[0, 0].real), n_g=float(d.n_group.values[0, 0]),
                       te_fraction=float(d.pol_fraction.te.values[0, 0]),
                       mode1_n_eff=float(d.n_eff.values[0, 1].real),
                       mode1_te_fraction=float(d.pol_fraction.te.values[0, 1]),
                       seconds=round(time.time() - t1, 1))
# grid convergence on a smaller plane (a uniform 5 nm grid on the full 2.6 x 2.0 um plane needs > 3 GB):
# same 1.6 x 1.2 um plane at 10 nm and at 5 nm, so the box effect and the grid effect are separated.
t1 = time.time()
d10s = solve(NSI, NOX, dl=0.01, box=(1.61, 1.21), num_modes=1, ng=False)
d5s = solve(NSI, NOX, dl=0.005, box=(1.605, 1.205), num_modes=1, ng=False)
out["small_box_dl10nm_n_eff"] = float(d10s.n_eff.values[0, 0].real)
out["small_box_dl5nm_n_eff"] = float(d5s.n_eff.values[0, 0].real)
out["grid_check_seconds"] = round(time.time() - t1, 1)
# thermo-optic: +10 K
dT = 10.0
dh = solve(NSI + 1.86e-4 * dT, NOX + 1e-5 * dT, num_modes=1, ng=False)
dndT = (float(dh.n_eff.values[0, 0].real) - out["straight"]["n_eff"]) / dT
out["dneff_dT_per_K"] = dndT
out["dlambda_dT_pm_per_K_fixed_FSR_formula"] = LAM * 1e6 * dndT / out["straight"]["n_g"]
# bend R = 7.5 um and R = 5 um, PML on the plane edges for radiation loss
for R in (7.5, 5.0):
    t1 = time.time()
    db = solve(NSI, NOX, bend=R, num_pml=(12, 12), box=(3.61, 2.01), num_modes=1, ng=False)
    neff = complex(db.n_eff.values[0, 0])
    k = float(db.k_eff.values[0, 0])
    alpha_db_cm = 10 * np.log10(np.e) * 2 * (2 * np.pi / (LAM * 1e-4)) * k
    out[f"bend_R{R}"] = dict(n_eff=neff.real, k_eff=k, rad_loss_dB_per_cm=float(alpha_db_cm),
                             seconds=round(time.time() - t1, 1))
out["total_seconds"] = round(time.time() - t0, 1)
print(json.dumps(out, indent=1))
os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "out", "t01_tidy3d_mode.json"), "w"), indent=1)
