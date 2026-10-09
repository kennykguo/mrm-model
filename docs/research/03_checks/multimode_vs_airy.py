import numpy as np
c=299792458.0; lam0=1310e-9; w0=2*np.pi*c/lam0
ng=4.2; L=2*np.pi*6.3e-6; Trt=ng*L/c; FSR=1/Trt; kap=w0/3500
kd=0.01*kap; ke=0.5*kap; ki=kap-ke-kd
a=np.exp(-ki*Trt/2); r1=np.exp(-ke*Trt/2); r2=np.exp(-kd*Trt/2)
def airy(phi):
    den=1-2*r1*r2*a*np.cos(phi)+(r1*r2*a)**2
    return (r2**2*a**2-2*r1*r2*a*np.cos(phi)+r1**2)/den, (1-r1**2)*(1-r2**2)*a/den
def tcmt(df, M, alt):
    D=2*np.pi*df; t=1+0j; d=0j
    for m in range(-M,M+1):
        am=-1j*np.sqrt(ke)/(kap/2+1j*(D-2*np.pi*m*FSR))
        t+= -1j*np.sqrt(ke)*am
        d+= -1j*np.sqrt(kd)*am*((-1)**m if alt else 1)
    return abs(t)**2, abs(d)**2
for df in (200e9, 400e9, FSR/2, 1.4e12):
    Ta,Da=airy(2*np.pi*df*Trt)
    out=[f"{df/1e9:6.0f} GHz Airy thru {10*np.log10(Ta):7.3f} drop {10*np.log10(Da):7.2f}"]
    for M,alt in ((0,False),(3,False),(3,True)):
        T,Dd=tcmt(df,M,alt); out.append(f"M={M} alt={alt}: thru {10*np.log10(T):7.3f} drop {10*np.log10(Dd):7.2f}")
    print(" | ".join(out))
