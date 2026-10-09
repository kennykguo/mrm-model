"""Curvature k (dB per K^2) of OMA about its optimum bias for a Lorentzian notch, versus swing, linewidth and
dlambda/dT. Used in docs/research/06_textbook_audit.md to show that the published relOMA curvature constrains
FWHM/(dlambda/dT), not dlambda/dT alone.  Run: .venv/bin/python docs/research/scripts/06_reloma_curvature.py
"""
import numpy as np
from scipy.optimize import brentq, minimize_scalar
def T(d, fw, tmin): x=2*d/fw; return 1-(1-tmin)/(1+x*x)
def oma(d, s, fw, tmin): return abs(T(d+s/2,fw,tmin)-T(d-s/2,fw,tmin))
def best(s, fw, tmin):
    r = minimize_scalar(lambda d: -oma(d,s,fw,tmin), bounds=(0, s+3*fw), method='bounded', options={'xatol':1e-6})
    return r.x, -r.fun
def curv_db(s, fw, tmin):
    d0, o0 = best(s, fw, tmin); h = 0.5
    f = lambda d: 10*np.log10(oma(d,s,fw,tmin))
    return -(f(d0+h)-2*f(d0)+f(d0-h))/h**2/2, d0, o0   # dB per pm^2 (penalty = k*dd^2)
for fw in (374, 500, 650):
  for tmin in (0.02, 0.01):
    for target in (None, -3.0, -2.55, -2.1):
        if target is None: s = 65.0
        else:
            try: s = brentq(lambda s: 10*np.log10(best(s,fw,tmin)[1]) - target, 5, 1500)
            except ValueError: continue
        kpm, d0, o0 = curv_db(s, fw, tmin)
        out = [f"FWHM={fw} Tmin={tmin} swing={s:.0f}pm relOMA={10*np.log10(o0):.2f}dB dopt={d0:.0f}pm"]
        for dl in (50, 63.5, 66, 73, 86):
            out.append(f"k@{dl}={kpm*dl*dl:.2f}")
        print("  ".join(out))
