"""Smoke test: transient heat conduction of a ring heater in a layered chip stack (axisymmetric r-z, and
a small 3-D block), assembled with scikit-fem, then thermal model-order reduction to a small state space
with (a) one-sided Krylov / moment matching at s = 0 (Galerkin, PRIMA-like, plain scipy), (b) pyMOR
balanced truncation (low-rank ADI) and (c) pyMOR IRKA (H2-optimal interpolation); python-control (with
slycot) checks the reduced model and does a dense balred.

Every geometry and material value here is ASSUMED for the smoke test (generic SOI: 2 um BOX as in the
Tidy3D example [T], 3 um top oxide, Si substrate truncated at 20 um with a fixed-temperature bottom,
ring R = 7.5 um with a 400 x 200 nm core that also carries the heat source, k_Si,film = 100,
k_Si,sub = 148, k_SiO2 = 1.4 W/(m K), rho*c = 1.66e6 (Si) and 1.62e6 (SiO2) J/(m^3 K)).
Units: um, us, mW, K  ->  k [mW/(um K)] = k_SI * 1e-3,  rho*c [mW us/(um^3 K)] = (rho c)_SI * 1e-9.
Run:
  uv run --isolated --no-project --python 3.12 --with pymor==2026.1.1 --with scikit-fem==12.0.2 \
     --with control==0.10.2 --with slycot==0.7.0 python docs/research/05_tools_smoke/t05_heat_mor.py
"""
import json, os, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from skfem import MeshTri, MeshTet, Basis, ElementTriP1, ElementTetP1, BilinearForm, LinearForm, condense
from skfem.helpers import dot, grad

t0 = time.time()
out = {}
R, W, H = 7.5, 0.40, 0.20
Z_SUB, Z_BOX, Z_TOP, R_MAX = -22.0, -2.0, 3.2, 30.0


def graded(a, b, h0, h1, n_min=2):
    """1-D node list from a to b, element size growing geometrically from h0 (at a) to <= h1."""
    xs, h = [a], h0
    while xs[-1] < b - 1e-12:
        xs.append(min(xs[-1] + h, b))
        h = min(h * 1.3, h1)
    return np.array(xs)


def sym(center, half, h0, h1, lo, hi):
    """Nodes refined around [center-half, center+half] and graded outward to [lo, hi]."""
    core = np.linspace(center - half, center + half, max(3, int(round(2 * half / h0)) + 1))
    left = center - half - graded(0, center - half - lo, h0, h1)[::-1]
    right = center + half + graded(0, hi - center - half, h0, h1)
    return np.unique(np.round(np.concatenate([left, core, right]), 9))


# ---------------------------------------------------------------- axisymmetric model
rr = sym(R, W / 2, 0.025, 1.5, 0.0, R_MAX)
zz = np.unique(np.round(np.concatenate([sym(H / 2, H / 2, 0.025, 1.0, Z_BOX, Z_TOP),
                                        graded(0, Z_BOX - Z_SUB, 0.1, 2.0)[::-1] * -1 + Z_BOX]), 9))
mesh = MeshTri.init_tensor(rr, zz)
basis = Basis(mesh, ElementTriP1())
xc = mesh.p[:, mesh.t].mean(axis=1)                      # element centroids (r, z)
ring = (np.abs(xc[0] - R) < W / 2) & (xc[1] > 0) & (xc[1] < H)
sub = xc[1] < Z_BOX
k_el = np.where(ring, 100e-3, np.where(sub, 148e-3, 1.4e-3))
rc_el = np.where(ring | sub, 1.66e-3, 1.62e-3)           # rho c * 1e-9 (mW us / um^3 K)


@BilinearForm
def stiff(u, v, w):
    return 2 * np.pi * w.x[0] * w.k * dot(grad(u), grad(v))


@BilinearForm
def mass(u, v, w):
    return 2 * np.pi * w.x[0] * w.rc * u * v


@LinearForm
def ring_load(v, w):
    return 2 * np.pi * w.x[0] * w.ind * v


nq = basis.quadrature[0].shape[1]
kq, rcq, indq = (np.repeat(a[:, None], nq, axis=1) for a in (k_el, rc_el, ring.astype(float)))
K = stiff.assemble(basis, k=kq)
M = mass.assemble(basis, rc=rcq)
g = ring_load.assemble(basis, ind=indq)                  # = int_ring 2 pi r phi_i dA
ring_vol = g.sum()                                        # um^3
b_full = g / ring_vol                                     # q = 1 mW / ring volume, uniform in the core
c_full = g / ring_vol                                     # volume-average temperature of the core
bottom = basis.get_dofs(lambda x: np.isclose(x[1], Z_SUB))
I = basis.complement_dofs(bottom)
Kc, Mc = K[I][:, I].tocsc(), M[I][:, I].tocsc()
B, C = b_full[I][:, None], c_full[I][None, :]
n = Kc.shape[0]
out["axisym"] = dict(nodes=int(mesh.p.shape[1]), triangles=int(mesh.t.shape[1]), states=int(n),
                     ring_volume_um3=float(ring_vol))
lu = spla.splu(Kc)
Rth = float((C @ lu.solve(B))[0, 0])
out["axisym"]["Rth_K_per_mW"] = Rth
# slowest thermal time constants: K x = lambda M x, smallest lambda
lam = spla.eigsh(Kc, k=4, M=Mc, sigma=0, which="LM", return_eigenvectors=False)
out["axisym"]["slowest_time_constants_us"] = sorted((1 / lam).tolist(), reverse=True)


def H_full(s):
    return complex((C @ spla.spsolve((s * Mc + Kc).tocsc(), B[:, 0].astype(complex)))[0])


freqs = np.logspace(1, 8, 36)                            # Hz
s_list = 2j * np.pi * freqs * 1e-6                       # 1/us
t1 = time.time()
Hf = np.array([H_full(s) for s in s_list])
out["axisym"]["full_freq_sweep_s"] = round(time.time() - t1, 1)
f3 = np.interp(1 / np.sqrt(2), np.abs(Hf[::-1]) / Rth, freqs[::-1])
out["axisym"]["f_3dB_Hz"] = float(f3)


def rel_err(Hr):
    return float(np.max(np.abs(Hr - Hf)) / np.max(np.abs(Hf)))


def eval_red(Er, Ar, Br, Cr):
    return np.array([complex((Cr @ np.linalg.solve(s * Er - Ar, Br))[0, 0]) for s in s_list])


# (a) one-sided Krylov / moment matching (Galerkin, V = W): at each real shift s0 the block
#     (s0 M + K)^-1 B, (s0 M + K)^-1 M (s0 M + K)^-1 B, ... matches the first moments of H(s) at s0.
red = {}
for order in (2, 4, 6, 8):
    t1 = time.time()
    shifts = [0.0] if order <= 2 else [0.0, 1e-2] if order <= 4 else [0.0, 1e-3, 1e-1]
    per = int(np.ceil(order / len(shifts)))
    vs = []
    for s0 in shifts:
        lu_s = spla.splu((s0 * Mc + Kc).tocsc())
        v = lu_s.solve(B[:, 0])
        vs.append(v)
        for _ in range(per - 1):
            v = lu_s.solve(Mc @ v)
            vs.append(v)
    V, _ = np.linalg.qr(np.array(vs[:order]).T)
    Er, Ar = V.T @ (Mc @ V), -(V.T @ (Kc @ V))
    Br, Cr = V.T @ B, C @ V
    red[f"krylov_r{order}"] = dict(rel_Hinf_err=rel_err(eval_red(Er, Ar, Br, Cr)),
                                   dc_gain=float((Cr @ np.linalg.solve(-Ar, Br))[0, 0]),
                                   shifts_per_us=shifts, seconds=round(time.time() - t1, 2))

# (b), (c) pyMOR
try:
    import pymor
    from pymor.models.iosys import LTIModel
    from pymor.reductors.bt import BTReductor
    from pymor.reductors.h2 import IRKAReductor
    from pymor.core.logger import set_log_levels
    set_log_levels({"pymor": "ERROR"})
    fom = LTIModel.from_matrices(-Kc, B, C, E=Mc)
    out["pymor_version"] = pymor.__version__
    for order in (2, 4, 6, 8):
        t1 = time.time()
        rom = BTReductor(fom).reduce(order)
        Hr = np.array([rom.transfer_function.eval_tf(s)[0, 0] for s in s_list])
        red[f"pymor_BT_r{order}"] = dict(rel_Hinf_err=rel_err(Hr), seconds=round(time.time() - t1, 2))
        t1 = time.time()
        rom_i = IRKAReductor(fom).reduce(order, conv_crit="h2", tol=1e-6, maxit=60)
        Hr = np.array([rom_i.transfer_function.eval_tf(s)[0, 0] for s in s_list])
        red[f"pymor_IRKA_r{order}"] = dict(rel_Hinf_err=rel_err(Hr), seconds=round(time.time() - t1, 2))
    rom6 = BTReductor(fom).reduce(6)
    A6, B6, C6, D6, E6 = rom6.to_matrices()
except Exception as e:  # noqa
    out["pymor_error"] = repr(e)
    A6 = None
out["axisym"]["reductions"] = red

# python-control on the 6-state BT model (and slycot-based balred from 6 to 3)
try:
    import control as ct
    out["control_version"] = ct.__version__
    if A6 is not None:
        Ad = np.linalg.solve(E6, A6) if E6 is not None else A6
        Bd = np.linalg.solve(E6, B6) if E6 is not None else B6
        sys6 = ct.ss(Ad, Bd, C6, 0)
        out["control_rom6"] = dict(dc_gain=float(ct.dcgain(sys6)),
                                   poles_time_constants_us=sorted((-1 / np.real(ct.poles(sys6))).tolist(), reverse=True))
        sys3 = ct.balred(sys6, 3, method="truncate")
        out["control_balred_3_dc_gain"] = float(ct.dcgain(sys3))
except Exception as e:  # noqa
    out["control_error"] = repr(e)

# ---------------------------------------------------------------- 3-D block (tiny): straight heater
t1 = time.time()
xs = sym(0.0, W / 2, 0.05, 2.0, -12.0, 12.0)
ys = np.linspace(-6, 6, 13)
zs = np.unique(np.round(np.concatenate([sym(H / 2, H / 2, 0.05, 1.0, Z_BOX, Z_TOP),
                                        graded(0, Z_BOX - Z_SUB, 0.25, 4.0)[::-1] * -1 + Z_BOX]), 9))
m3 = MeshTet.init_tensor(xs, ys, zs)
b3 = Basis(m3, ElementTetP1())
xc3 = m3.p[:, m3.t].mean(axis=1)
core3 = (np.abs(xc3[0]) < W / 2) & (xc3[2] > 0) & (xc3[2] < H)
sub3 = xc3[2] < Z_BOX
nq3 = b3.quadrature[0].shape[1]
k3 = np.repeat(np.where(core3, 100e-3, np.where(sub3, 148e-3, 1.4e-3))[:, None], nq3, axis=1)
i3 = np.repeat(core3.astype(float)[:, None], nq3, axis=1)


@BilinearForm
def stiff3(u, v, w):
    return w.k * dot(grad(u), grad(v))


@LinearForm
def load3(v, w):
    return w.ind * v


K3 = stiff3.assemble(b3, k=k3)
g3 = load3.assemble(b3, ind=i3)
bot3 = b3.get_dofs(lambda x: np.isclose(x[2], Z_SUB))
T3 = np.zeros(K3.shape[0])
T3 = __import__("skfem").solve(*condense(K3, g3 / g3.sum(), D=bot3))
out["block3d"] = dict(nodes=int(m3.p.shape[1]), tets=int(m3.t.shape[1]),
                      Rth_K_per_mW_12um_heater=float(g3 @ T3 / g3.sum()), seconds=round(time.time() - t1, 1))
out["total_seconds"] = round(time.time() - t0, 1)
print(json.dumps(out, indent=1))
os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "out", "t05_heat_mor.json"), "w"), indent=1)
