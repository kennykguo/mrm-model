"""Physical constants and unit conversions shared by every module (SI inside the code)."""
import math

C0 = 299_792_458.0            # speed of light in vacuum, m/s
Q_E = 1.602_176_634e-19       # elementary charge, C
H_PLANCK = 6.626_070_15e-34   # Planck constant, J s
K_B = 1.380_649e-23           # Boltzmann constant, J/K
EPS0 = 8.854_187_8128e-12     # vacuum permittivity, F/m
T0_K = 273.15                 # 0 degC in K


def c_to_k(t_c):
    return t_c + T0_K


def k_to_c(t_k):
    return t_k - T0_K


def db_to_lin(db):
    return 10.0 ** (db / 10.0)


def lin_to_db(x):
    return 10.0 * math.log10(max(x, 1e-300))


def dbm_to_w(dbm):
    return 1e-3 * 10.0 ** (dbm / 10.0)


def w_to_dbm(w):
    return 10.0 * math.log10(max(w, 1e-300) / 1e-3)


def wavelength_to_freq(lam_m):
    return C0 / lam_m


def freq_to_wavelength(f_hz):
    return C0 / f_hz


def dlambda_from_df(df_hz, lam_m):
    """Magnitude of the wavelength change for a small frequency change at wavelength lam_m."""
    return df_hz * lam_m * lam_m / C0


def df_from_dlambda(dlam_m, lam_m):
    """Magnitude of the frequency change for a small wavelength change at wavelength lam_m."""
    return dlam_m * C0 / (lam_m * lam_m)
