"""Smoke test: SAX (JAX) S-matrix circuit of a WDM bus with 8 add-drop microrings in series, checked
against a closed-form numpy cascade, plus JAX float64, jit and grad (d T_through / d phase) checks.

Toy values (ASSUMED except where tagged): 8 rings, resonances on a 400 GHz grid (G26 printed: 400 GHz
per direction), n_g = 4.34 (derived, femwell t02), L chosen for FSR = 2.8 THz (G26 requirement, inferred
in REF.fsrRequiredG26THz), round-trip amplitude a = 0.95, self-coupling t1 = 0.95, t2 = 0.99 (drop port
weakly coupled, like a monitor-PD tap).
Run:
  uv run --isolated --no-project --python 3.12 --with sax==0.18.2 python docs/research/05_tools_smoke/t06_sax_wdm_bus.py
"""
import json, os, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import sax

t0 = time.time()
C0 = 299792458.0
NG, NEFF0, LAM0 = 4.34, 2.4856, 1.31            # um
FSR_HZ = 2.8e12
L = C0 / (NG * FSR_HZ) * 1e6                      # um
A_RT, T1, T2 = 0.95, 0.95, 0.99
NR = 8
grid_hz = 400e9


def phase(wl, dphi):
    neff = NEFF0 - (wl - LAM0) * (NG - NEFF0) / LAM0
    return 2 * jnp.pi * neff * L / wl + dphi


def add_drop(wl=1.31, dphi=0.0):
    """Add-drop ring: ports in, through, add, drop (field amplitudes, unidirectional)."""
    p = phase(wl, dphi)
    k1, k2 = jnp.sqrt(1 - T1**2), jnp.sqrt(1 - T2**2)
    e = A_RT * jnp.exp(1j * p)
    den = 1 - T1 * T2 * e
    thru = (T1 - T2 * e) / den
    drop = -k1 * k2 * jnp.sqrt(A_RT) * jnp.exp(1j * p / 2) / den
    thru_add = (T2 - T1 * e) / den
    return sax.reciprocal({("in", "through"): thru, ("in", "drop"): drop,
                           ("add", "drop"): thru_add, ("add", "through"): drop})


# detune ring i so that its resonance sits on grid line i (resonance m at lam0 first, then shift)
m0 = jnp.round(NEFF0 * L / LAM0)
lam_res0 = NEFF0 * L / m0                          # um, approximate
dphi = [float(2 * np.pi * NG * L * (i * grid_hz) / C0) for i in range(NR)]   # frequency offset -> phase

instances = {f"r{i}": "ring" for i in range(NR)}
connections = {f"r{i},through": f"r{i+1},in" for i in range(NR - 1)}
ports = {"in": "r0,in", "out": f"r{NR-1},through"}
ports.update({f"drop{i}": f"r{i},drop" for i in range(NR)})
circuit, _ = sax.circuit(netlist={"instances": instances, "connections": connections, "ports": ports},
                         models={"ring": add_drop})
wl = jnp.linspace(1.300, 1.330, 30001)
kw = {f"r{i}": {"dphi": dphi[i]} for i in range(NR)}
t1 = time.time()
S = circuit(wl=wl, **kw)
T_sax = np.abs(np.asarray(S["in", "out"])) ** 2
first = time.time() - t1
t1 = time.time()
S = circuit(wl=wl, **kw)
T_sax = np.abs(np.asarray(S["in", "out"])) ** 2
second = time.time() - t1

# closed-form numpy cascade of the through ports
T_np = np.ones_like(np.asarray(wl), dtype=complex)
for i in range(NR):
    T_np *= np.asarray(add_drop(wl, dphi[i])[("in", "through")])
err = float(np.max(np.abs(T_sax - np.abs(T_np) ** 2)))

# jit + grad: slope of ring-0 through power vs its heater phase at a fixed laser wavelength
f = jax.jit(lambda d, w: jnp.abs(add_drop(w, d)[("in", "through")]) ** 2)
g = jax.jit(jax.grad(lambda d, w: jnp.abs(add_drop(w, d)[("in", "through")]) ** 2))
w_l = float(wl[int(np.argmin(T_sax))]) + 0.0001
fd = (f(1e-6, w_l) - f(-1e-6, w_l)) / 2e-6
out = dict(sax=sax.__version__, jax=jax.__version__, backend=jax.default_backend(),
           x64=bool(jax.config.read("jax_enable_x64")), L_um=L, n_wavelengths=int(wl.size),
           first_call_s=round(first, 2), second_call_s=round(second, 3),
           max_abs_err_vs_numpy=err, min_T_dB=float(10 * np.log10(T_sax.min())),
           grad_vs_finite_diff=[float(g(0.0, w_l)), float(fd)], total_s=round(time.time() - t0, 1))
print(json.dumps(out, indent=1))
os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "out", "t06_sax_wdm_bus.json"), "w"), indent=1)
