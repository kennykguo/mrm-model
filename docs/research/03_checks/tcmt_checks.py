"""Numerical checks for docs/research/03_mrm_compact_models.md (all results are 'derived').
Light: numpy/scipy only, < 200 MB.
Conventions: e^{+i w t}; a = mode amplitude with |a|^2 = stored energy [J]; |s|^2 = power [W];
kappa_* = ENERGY decay rates [1/s]; Delta = w_laser - w_res [rad/s].
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares

c = 299792458.0
h = 6.62607015e-34
lam0 = 1310e-9
w0 = 2 * np.pi * c / lam0
f0 = c / lam0

# ---------------- reference ring (REF values + derived) ----------------
ng = 4.2            # REF assumed (mode solve of 400x200 strip gives 4.33, derived)
L = 2 * np.pi * 6.3e-6  # REF derived radius 6.3 um
Trt = ng * L / c
FSR = 1 / Trt
QL = 3500.0         # G25 printed on talk
kap = w0 / QL       # total energy decay rate
print("== reference ring ==")
print(f"T_rt = {Trt*1e12:.3f} ps, FSR = {FSR/1e12:.3f} THz = {lam0**2*FSR/c*1e9:.2f} nm")
print(f"kappa = {kap:.3e} 1/s, FWHM = {kap/2/np.pi/1e9:.1f} GHz = {lam0/QL*1e12:.0f} pm")
print(f"energy lifetime tau_p = 1/kappa = {1/kap*1e12:.2f} ps; field (amplitude) lifetime = {2/kap*1e12:.2f} ps")
F = FSR / (kap / 2 / np.pi)
print(f"finesse = {F:.1f}; build-up at critical coupling (all-pass) ~ F/pi = {F/np.pi:.2f}")

# ---------------- TCMT <-> transfer-matrix mapping and Airy comparison ----------------
def tmm_rates(a, r1, r2):
    return -2 * np.log(a) / Trt, -2 * np.log(r1) / Trt, -2 * np.log(r2) / Trt

def airy_adddrop(phi, a, r1, r2):
    den = 1 - 2 * r1 * r2 * a * np.cos(phi) + (r1 * r2 * a) ** 2
    thru = (r2**2 * a**2 - 2 * r1 * r2 * a * np.cos(phi) + r1**2) / den
    drop = (1 - r1**2) * (1 - r2**2) * a / den
    return thru, drop

def tcmt_adddrop(Delta, ki, ke, kd):
    k = ki + ke + kd
    t = 1 - ke / (k / 2 + 1j * Delta)
    d = ke * kd / (Delta**2 + (k / 2) ** 2)
    return np.abs(t) ** 2, d

# G26-like add-drop: kappa_d = 1% of kappa, near critical through notch
for label, kd_frac in (("all-pass (G25)", 0.0), ("add-drop, kd=1% (G26-like, derived)", 0.01)):
    kd = kd_frac * kap
    ke = 0.5 * kap               # input coupling = half of total (critical when ki+kd = ke)
    ki = kap - ke - kd
    a = np.exp(-ki * Trt / 2); r1 = np.exp(-ke * Trt / 2); r2 = np.exp(-kd * Trt / 2)
    print(f"\n== {label}: a={a:.5f} r1={r1:.5f} r2={r2:.5f}  Q_i={w0/ki:.0f} Q_e={w0/ke:.0f}"
          + (f" Q_d={w0/kd:.0f}" if kd > 0 else ""))
    for df in (0.0, 32.7e9, 100e9, 200e9, 400e9, FSR / 2):
        Delta = 2 * np.pi * df
        phi = 2 * np.pi * df * Trt  # round-trip phase offset from resonance
        T_a, D_a = airy_adddrop(phi, a, r1, r2)
        T_c, D_c = tcmt_adddrop(Delta, ki, ke, kd)
        print(f"  offset {df/1e9:7.1f} GHz: thru Airy {10*np.log10(T_a):8.3f} dB  TCMT {10*np.log10(T_c):8.3f} dB"
              + (f" | drop Airy {10*np.log10(D_a):7.2f} dB TCMT {10*np.log10(D_c):7.2f} dB" if kd > 0 else ""))
    # linewidth comparison
    fw_airy = 2 * np.arccos(1 - (1 - a * r1 * r2) ** 2 / (2 * a * r1 * r2)) / (2 * np.pi * Trt)
    print(f"  FWHM: TCMT {kap/2/np.pi/1e9:.3f} GHz vs Airy exact {fw_airy/1e9:.3f} GHz")
    if kd > 0:
        print(f"  peak drop fraction 4 ke kd / k^2 = {4*ke*kd/kap**2:.4f}")

# ---------------- Backscatter: splitting and reflection ----------------
print("\n== backscatter (lumped field reflectance r_bs per round trip -> g = r_bs/T_rt) ==")
for rbs in (0.003, 0.0124, 0.03):
    g = rbs / Trt
    ke = kap / 2
    R0 = (ke * g) ** 2 / ((kap / 2) ** 2 + g**2) ** 2
    print(f"  r_bs={rbs}: g/2pi={g/2/np.pi/1e9:.2f} GHz, splitting 2g/2pi={2*g/2/np.pi/1e9:.2f} GHz "
          f"(linewidth {kap/2/np.pi/1e9:.1f} GHz); on-resonance reflection {10*np.log10(R0):.1f} dB")

# ---------------- small-signal response: closed form vs ODE ----------------
print("\n== small-signal EO response: closed form (Karimelahi & Sheikholeslami 2016 form) vs TCMT ODE ==")
def H_closed(Om, Delta, ki, ke, q=0.0):
    """Optical part only. Resonance modulation dw_r(t) = eps*cos(Om t) with optional loss modulation
    d(kappa_i/2) = q*eps*cos(Om t). Returns complex dP/eps normalised to DC value (computed same way)."""
    k = ki + ke
    s = 1j * Om
    # derived directly: linearise da/dt = (-i(Delta - dw) - k/2 - dk/2) a - i sqrt(ke) S
    a0 = -1j * np.sqrt(ke) / (k / 2 + 1j * Delta)
    t0 = 1 - 1j * np.sqrt(ke) * a0
    # perturbation source: (+i dw - q dw) a0  (resonance up => Delta down)
    src = (1j - q) * a0
    A_p = src / (s + k / 2 + 1j * Delta) / 2          # e^{+i Om t} component of a1
    A_m = src / (-s + k / 2 + 1j * Delta) / 2         # e^{-i Om t} component (s -> -s)
    # P = |t0 - i sqrt(ke) a1|^2 -> dP(+Om) = -i sqrt(ke) A_p t0* + (i sqrt(ke) A_m t0*)^*  ... e^{+iOm t} coefficient
    dP = (-1j * np.sqrt(ke) * A_p) * np.conj(t0) + np.conj(-1j * np.sqrt(ke) * A_m) * t0
    return dP

def ode_response(fm, Delta, ki, ke, eps, ncyc=12):
    k = ki + ke
    Om = 2 * np.pi * fm
    def rhs(t, y):
        a = y[0] + 1j * y[1]
        dw = eps * np.cos(Om * t)
        da = (-1j * (Delta - dw) - k / 2) * a - 1j * np.sqrt(ke)
        return [da.real, da.imag]
    a0 = -1j * np.sqrt(ke) / (k / 2 + 1j * Delta)
    T = ncyc / fm
    tt = np.linspace(0, T, 1000 * ncyc, endpoint=False)
    sol = solve_ivp(rhs, (0, T), [a0.real, a0.imag], t_eval=tt, rtol=1e-11, atol=1e-16, method='DOP853', max_step=min(1 / fm / 200, 0.2e-12))
    a = sol.y[0] + 1j * sol.y[1]
    P = np.abs(1 - 1j * np.sqrt(ke) * a) ** 2
    m = tt >= T / 2
    # lock-in at Om
    return np.mean(P[m] * np.exp(-1j * Om * tt[m]))

ki = ke = kap / 2  # critically coupled all-pass
x_opt = 1 / np.sqrt(3)  # 2 delta / FWHM at steepest slope of a Lorentzian notch
for x in (x_opt, 0.813, 2.0):
    Delta = x * kap / 2   # since 2*Delta/kappa = 2 delta/FWHM
    eps = 1e-4 * kap
    H0 = H_closed(1e-6, Delta, ki, ke)
    for fm in (5e9, 30e9, 60e9, 100e9):
        Hc = H_closed(2 * np.pi * fm, Delta, ki, ke) / H0
        Ho = ode_response(fm, Delta, ki, ke, eps) / (eps * H0)
        print(f"  x={x:.3f} f={fm/1e9:5.0f} GHz: closed |H|^2={abs(Hc)**2:.4f}  ODE |H|^2={abs(Ho)**2:.4f}")

def opt_bw(x, ki, ke, q=0.0):
    k = ki + ke; Delta = x * k / 2
    H0 = H_closed(1e-6, Delta, ki, ke, q)
    f = lambda fm: abs(H_closed(2 * np.pi * fm, Delta, ki, ke, q) / H0) ** 2 - 0.5
    fs = np.linspace(1e8, 1e12, 20000)
    v = np.array([f(z) for z in fs])
    i = np.where(v < 0)[0][0]
    return brentq(f, fs[i - 1], fs[i])

def peak_db(x, ki, ke, q=0.0):
    k = ki + ke; Delta = x * k / 2
    H0 = H_closed(1e-6, Delta, ki, ke, q)
    fs = np.linspace(1e8, 3e11, 6000)
    return 10 * np.log10(max(abs(H_closed(2 * np.pi * fm, Delta, ki, ke, q) / H0) ** 2 for fm in fs))

FW = kap / 2 / np.pi
print(f"\n  FWHM = {FW/1e9:.1f} GHz")
for x in (0.3, x_opt, 0.813, 1.0, 1.5, 2.0):
    print(f"  x=2delta/FWHM={x:.3f}: optical-only electrical-3dB BW = {opt_bw(x, ki, ke)/1e9:.1f} GHz "
          f"({opt_bw(x, ki, ke)/FW:.2f} FWHM), peaking {peak_db(x, ki, ke):.2f} dB")
for ratio, lab in ((3.0, "over-coupled ke=3ki"), (1/3, "under-coupled ke=ki/3")):
    kie = kap / (1 + ratio); kee = kap - kie
    print(f"  {lab}, x=0.813: BW {opt_bw(0.813, kie, kee)/1e9:.1f} GHz, peaking {peak_db(0.813, kie, kee):.2f} dB")
for q in (-0.1, 0.06, 0.13):
    print(f"  loss-mod ratio q={q}, x=0.577: BW {opt_bw(x_opt, ki, ke, q)/1e9:.1f} GHz peaking {peak_db(x_opt, ki, ke, q):.2f} dB")

# What RC pole does a measured 30.9 GHz total EO bandwidth imply at x = 1/sqrt(3)?
def total_bw(fRC, x):
    k = ki + ke; Delta = x * k / 2
    H0 = H_closed(1e-6, Delta, ki, ke)
    g = lambda fm: abs(H_closed(2 * np.pi * fm, Delta, ki, ke) / H0) ** 2 / (1 + (fm / fRC) ** 2) - 0.5
    return brentq(g, 1e8, 5e11)
for x in (x_opt, 0.813):
    fRC = brentq(lambda fr: total_bw(fr, x) - 30.9e9, 10e9, 1e12)
    print(f"  measured total EO 30.9 GHz at x={x:.3f} implies RC pole {fRC/1e9:.1f} GHz")
print(f"  textbook cascade (30.9 GHz RC pole x 65 GHz real pole) gives "
      f"{brentq(lambda f: 1/((1+(f/30.9e9)**2)*(1+(f/65.4e9)**2)) - 0.5, 1e9, 1e11)/1e9:.1f} GHz")

# ---------------- nonlinear magnitudes ----------------
print("\n== nonlinear magnitudes at 1310 nm, critical all-pass, laser on resonance ==")
Aeff = 0.094e-12   # m^2, textbook derived (400x200/0.85)
Gam = 0.85
hw = h * f0
alpha_i = ki * ng / c   # 1/m
print(f"  intrinsic power loss alpha_i = {alpha_i/100:.1f} /cm = {4.343*alpha_i/100:.0f} dB/cm (Q_i={w0/ki:.0f})")
def nsm13(N, P):
    dn = -(2.98e-22 * N**1.016 + 1.25e-18 * P**0.835)
    da = 3.48e-22 * N**1.229 + 1.02e-19 * P**1.089
    return dn, da
for Pin_mW in (1.0, 2.5, 8.0, 10.0):
    Pin = Pin_mW * 1e-3
    U = 2 * Pin / kap   # stored energy on resonance at critical coupling: ke|s|^2/(k/2)^2 with ke=k/2
    Pc = U / Trt
    I = Pc / Aeff        # W/m^2
    for beta_cmGW in (0.57, 2.0):
        beta = beta_cmGW * 1e-11  # m/W
        a_tpa = beta * I         # 1/m
        G = beta * I**2 / (2 * hw)  # pairs / m^3 / s
        for tau in (10e-12, 1e-9):
            N = G * tau * 1e-6  # cm^-3
            dn, da = nsm13(N, N)
            dlam_fc = lam0 * Gam * dn / ng * 1e12  # pm
            print(f"  Pin={Pin_mW:4.1f} mW Pcirc={Pc*1e3:5.1f} mW I={I*1e-13:.4f} GW/cm2 beta={beta_cmGW}: "
                  f"alpha_TPA/alpha_i={a_tpa/alpha_i:.1e}; tau_fc={tau*1e12:5.0f} ps N={N:.1e} cm-3 "
                  f"FCD shift={dlam_fc:7.2f} pm FCA/alpha_i={da*100/alpha_i:.1e}")
    n2 = 4.5e-18  # m^2/W
    print(f"     Kerr shift (n2=4.5e-14 cm2/W): {lam0*Gam*n2*I/ng*1e12:.2f} pm")
    for eta in (0.3, 0.63, 0.9):
        print(f"     self-heating, P_abs = eta*Pin = {eta*Pin_mW:.2f} mW -> {eta*Pin_mW*405:.0f} pm at 405 pm/mW "
              f"(G25 heater-derived), {eta*Pin_mW*160:.0f} pm at 160 pm/mW (HPE assumed)")

# ---------------- G25 depletion shift fit ----------------
print("\n== fit of G25 read-off resonance shift vs reverse bias ==")
V = np.array([0.0, 0.8, 1.2, 1.6]); S = np.array([0.0, 56.4, 66.8, 84.4])
for m_exp in (0.5, 1 / 3):
    def res(p):
        A, Vbi = p
        return A * ((Vbi + V) ** (1 - m_exp) - Vbi ** (1 - m_exp)) - S
    sol = least_squares(res, [100, 0.9], bounds=([1, 0.3], [1e4, 1.5]))
    A, Vbi = sol.x
    eff = lambda v: A * (1 - m_exp) * (Vbi + v) ** (-m_exp)
    print(f"  m={m_exp:.3f}: A={A:.1f} Vbi={Vbi:.2f} V rms resid={np.sqrt(np.mean(sol.fun**2)):.1f} pm; "
          f"pm/V at 0,1,2,3.75 V = {eff(0):.1f}, {eff(1):.1f}, {eff(2):.1f}, {eff(3.75):.1f}")
    # fixed Vbi = 0.9 V physically motivated
    sol2 = least_squares(lambda p: p[0] * ((0.9 + V) ** (1 - m_exp) - 0.9 ** (1 - m_exp)) - S, [100])
    print(f"     with Vbi fixed 0.9 V: A={sol2.x[0]:.1f}, rms resid {np.sqrt(np.mean(sol2.fun**2)):.1f} pm")

# ---------------- G25 ER vs resonance position fit ----------------
print("\n== fit of G25 notch depth vs resonance wavelength: x = ke/ki = exp(s (lam - lam_c)) ==")
lam_r = np.array([1301.9, 1302.21, 1304.45, 1306.27, 1308.43, 1310.84, 1313.49, 1314.9])
er = np.array([28, 30, 30, 24, 22, 20, 18, 17.0])
def model_er(p, lr):
    s, lc = p
    x = np.exp(s * (lr - lc))
    return -20 * np.log10(np.abs(1 - x) / (1 + x) + 1e-6)
sol = least_squares(lambda p: np.clip(model_er(p, lam_r), 0, 40) - er, [0.02, 1302.0])
print(f"  s = {sol.x[0]*100:.2f} %/nm, critical at {sol.x[1]:.2f} nm, rms resid {np.sqrt(np.mean(sol.fun**2)):.1f} dB")
print("  model:", np.round(np.clip(model_er(sol.x, lam_r), 0, 40), 1))
# what a straight point coupler gives (derived estimate): d ln kappa^2/d lambda
neff, ncl = 2.49, 1.447
k0 = 2 * np.pi / 1.31
gam = k0 * np.sqrt(neff**2 - ncl**2)
dneff = -(4.33 - 2.49) / 1.31
dgam = -gam / 1.31 + k0 * neff * dneff / np.sqrt(neff**2 - ncl**2)
for gap in (0.15, 0.2, 0.25):
    dlnk2 = 2 * (gap * (-dgam) + 0.5 * (-dgam / gam))  # exp tail + point-coupler length sqrt(pi R/gam)
    print(f"  evanescent estimate gap {gap*1e3:.0f} nm: d ln(kappa^2)/d lambda = {dlnk2/10:.2f} %/nm")

# ---------------- adiabatic elimination check ----------------
print("\n== adiabatic elimination ==")
ramp = 2000 * 86e-12 * 2 * np.pi * c / lam0**2  # rad/s^2: 2000 K/s * 86 pm/K -> frequency sweep rate
lag = ramp / (kap / 2) ** 2   # detuning lag ~ (dDelta/dt) * tau_field / ... normalized
print(f"  2000 K/s x 86 pm/K = {2000*86e-3:.0f} nm/s; resonance moves {2000*86e-12*1e12*2/kap:.2e} pm per field lifetime")
print(f"  quasi-static error (dDelta/dt)/(kappa/2)^2 = {lag:.1e} (relative)")

# ---------------- verify the compact pole-zero formula used in the report ----------------
print("\n== pole-zero formula check: H(s) = -dP/dDelta * (1 + s/(ki+kd)) * wn^2/(s^2 + k s + wn^2) ==")
def H_formula(Om, Delta, ki, ke, kd=0.0):
    k = ki + ke + kd; s = 1j * Om; wn2 = (k / 2) ** 2 + Delta**2
    dPdD = 2 * Delta * ke * (ki + kd) / wn2**2
    return -dPdD * (1 + s / (ki + kd)) * wn2 / (s**2 + k * s + wn2)
for (kif, kef) in ((0.5, 0.5), (0.25, 0.75), (0.75, 0.25)):
    kii, kee = kif * kap, kef * kap
    for x in (0.4, 1.3):
        D = x * kap / 2
        err = max(abs(H_formula(2*np.pi*f, D, kii, kee) / 2 - H_closed(2*np.pi*f, D, kii, kee)) / abs(H_closed(2*np.pi*f, D, kii, kee))
                  for f in (1e9, 20e9, 50e9, 120e9))
        print(f"  ki={kif}k ke={kef}k x={x}: max rel err {err:.1e}")
