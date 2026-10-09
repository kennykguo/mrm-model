"""What the G26 heater code range (19,500 to 49,300 of 16 bits over an ~83 C excursion) implies for dlambda/dT
through the G25 measured V-to-lambda curve, as a function of the assumed voltage-DAC full scale.
Used in docs/research/06_textbook_audit.md.  Run: .venv/bin/python docs/research/scripts/06_dac_code_check.py
"""
import numpy as np
V=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
L=[1301.9,1301.98,1302.21,1302.58,1303.06,1303.7,1304.45,1305.31,1306.27,1307.3,1308.43,1309.6,1310.84,1312.15,1313.49,1314.9]
lam=lambda v: np.interp(v,V,L)
for vfs in (1.6,1.5,1.44,1.4):
    v1=vfs*19500/65535; v2=vfs*49300/65535
    dl=(lam(v2)-lam(v1))*1000
    print(f"V_FS={vfs}: V {v1:.3f}->{v2:.3f} V, dlambda={dl:.0f} pm, implied dl/dT over 83 K = {dl/83:.1f} pm/K")
# TCR-model power at endpoints, R0=45.4, y=0.00904
R0=45.4; y=0.00904
P=lambda v: 2*(v*v/R0*1e3)/(1+np.sqrt(1+4*y*(v*v/R0*1e3)))
for vfs in (1.6,1.44):
    v1=vfs*19500/65535; v2=vfs*49300/65535
    dP=P(v2)-P(v1); print(f"V_FS={vfs}: P {P(v1):.2f}->{P(v2):.2f} mW, dP={dP:.2f} mW, Rth=83/dP={83/dP:.2f} K/mW")
# heater efficiency of swept device at 1.195 V
print("swept device shift at 1.195 V:", (lam(1.195)-lam(0))*1000, "pm; P=", 1.195**2/55.9*1e3, "mW; eta=", (lam(1.195)-lam(0))/(1.195**2/55.9*1e3), "nm/mW")
# lag budgets
for d in (62.5,66,73,80,86):
    print(d, "pm/K: lag for 5 pm at 2000 C/s =", 5/(2000*d)*1e6, "us; at 808 C/s =", 5/(808*d)*1e6, "us")
