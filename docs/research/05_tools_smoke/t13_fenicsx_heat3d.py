"""Smoke test: FEniCSx (dolfinx 0.11, PETSc) 3-D transient heat in a layered block - the scale-up path
if the 3-D thermal model outgrows scikit-fem. ASSUMED stack as in t05 (Si substrate to -22 um with a
fixed-temperature bottom, 2 um BOX, 0.4 x 0.2 um Si heater/core line at z in [0, 0.2], oxide to 3.2 um),
units um, us, mW, K. Steady state (heater 1 mW over the 12 um line) + 20 backward-Euler steps of 1 us.
Run (scratch env created with: micromamba create -p <env> -c conda-forge python=3.12 fenics-dolfinx=0.11.0):
  <env>/bin/python docs/research/05_tools_smoke/t13_fenicsx_heat3d.py
"""
import json, time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem

t0 = time.time()
msh = mesh.create_box(MPI.COMM_WORLD, [np.array([-12.0, -6.0, -22.0]), np.array([12.0, 6.0, 3.2])],
                      [96, 24, 64], mesh.CellType.tetrahedron)
Q0 = fem.functionspace(msh, ("DG", 0))
V = fem.functionspace(msh, ("Lagrange", 1))


def piece(sub, core, ox):
    f = fem.Function(Q0)
    x = Q0.tabulate_dof_coordinates()
    is_core = (np.abs(x[:, 0]) < 0.2 + 0.125) & (x[:, 2] > 0) & (x[:, 2] < 0.2 + 0.4)   # coarse mesh: ~1 cell
    f.x.array[:] = np.where(x[:, 2] < -2.0, sub, np.where(is_core, core, ox))
    return f, is_core


k, is_core = piece(148e-3, 100e-3, 1.4e-3)
rc, _ = piece(1.66e-3, 1.66e-3, 1.62e-3)
ind = fem.Function(Q0)
ind.x.array[:] = is_core.astype(float)
vol = msh.comm.allreduce(fem.assemble_scalar(fem.form(ind * ufl.dx)), op=MPI.SUM)
q = fem.Constant(msh, PETSc.ScalarType(1.0 / vol))           # 1 mW spread over the heater cells
bottom = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[2], -22.0))
bc = fem.dirichletbc(PETSc.ScalarType(0.0), bottom, V)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
steady = LinearProblem(ufl.inner(k * ufl.grad(u), ufl.grad(v)) * ufl.dx, q * ind * v * ufl.dx, bcs=[bc],
                       petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
                       petsc_options_prefix="steady_")
Ts = steady.solve()
Tavg = msh.comm.allreduce(fem.assemble_scalar(fem.form(Ts * ind * ufl.dx)), op=MPI.SUM) / vol
dt = 1.0
Told = fem.Function(V)
a = (rc * u * v + dt * ufl.inner(k * ufl.grad(u), ufl.grad(v))) * ufl.dx
L = (rc * Told * v + dt * q * ind * v) * ufl.dx
trans = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10},
                      petsc_options_prefix="trans_")
hist = []
for n in range(20):
    Tn = trans.solve()
    Told.x.array[:] = Tn.x.array
    hist.append(msh.comm.allreduce(fem.assemble_scalar(fem.form(Tn * ind * ufl.dx)), op=MPI.SUM) / vol)
out = dict(dolfinx=dolfinx.__version__, cells=int(msh.topology.index_map(3).size_global),
           dofs=int(V.dofmap.index_map.size_global), Rth_K_per_mW=float(Tavg),
           T_after_us=[round(h, 4) for h in hist[::5]], frac_of_steady_at_20us=float(hist[-1] / Tavg),
           seconds=round(time.time() - t0, 1))
print(json.dumps(out, indent=1))
