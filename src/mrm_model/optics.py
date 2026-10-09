"""Passive periodic microring optics and energy-normalized local coupled modes.

All quantities use SI units. Defaults illustrate a silicon add-drop ring; they
are NOT a calibrated Lightmatter device. Couplers are reciprocal, lossless and
frequency independent, and the drop coupler sits halfway around a uniform ring.
The static transfer is exact for those assumptions; TCMT is a local, high-finesse
approximation and is deliberately distinguished from the periodic spectrum.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial

C = 299_792_458.0
H_PLANCK = 6.62607015e-34


@dataclass(frozen=True)
class OpticalParams:
    """Independent optical parameters; loss is POWER attenuation in Np/m.

    Voltage is signed junction voltage relative to zero bias. The illustrative
    positive df/dV means negative reverse bias red-shifts the resonance; actual
    polarity and bias dependence must be measured. Tables override linear
    voltage coefficients and reject extrapolation outside their supplied range.
    Intrinsic loss includes scattering: only its declared absorptive fraction
    deposits local heat. No independent Q, FSR or linewidth is accepted.
    """

    reference_frequency_hz: float = C / 1310e-9
    length_m: float = 40e-6
    group_index: float = 4.2
    loss_power_np_per_m: float = 1000.0
    coupling_bus_power: float = 0.08
    coupling_drop_power: float = 0.01
    df_dtemperature_hz_per_k: float = -15e9
    reference_temperature_k: float = 298.15
    df_dvoltage_hz_per_v: float = 8e9
    loss_voltage_np_per_m_per_v: float = 0.0
    phase_dispersion_s2: float = 0.0
    intrinsic_absorption_fraction: float = 1.0
    voltage_points_v: tuple[float, ...] = ()
    resonance_voltage_shifts_hz: tuple[float, ...] = ()
    loss_voltage_values_np_per_m: tuple[float, ...] = ()

    def __post_init__(self):
        for name in ("reference_frequency_hz", "length_m", "group_index", "reference_temperature_k"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("loss_power_np_per_m",):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("coupling_bus_power", "coupling_drop_power"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must lie in [0, 1)")
        for name in (
            "df_dtemperature_hz_per_k",
            "df_dvoltage_hz_per_v",
            "loss_voltage_np_per_m_per_v",
            "phase_dispersion_s2",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.intrinsic_absorption_fraction <= 1:
            raise ValueError("intrinsic_absorption_fraction must lie in [0, 1]")
        v = np.asarray(self.voltage_points_v, dtype=float)
        if v.size:
            if v.size < 2 or not np.all(np.isfinite(v)) or np.any(np.diff(v) <= 0):
                raise ValueError("voltage_points_v needs at least two increasing finite entries")
            for name in ("resonance_voltage_shifts_hz", "loss_voltage_values_np_per_m"):
                a = np.asarray(getattr(self, name), dtype=float)
                if a.size and (a.shape != v.shape or not np.all(np.isfinite(a))):
                    raise ValueError(f"{name} must match voltage_points_v")
            if np.any(np.asarray(self.loss_voltage_values_np_per_m) < 0):
                raise ValueError("voltage table losses must be nonnegative")
        elif self.resonance_voltage_shifts_hz or self.loss_voltage_values_np_per_m:
            raise ValueError("voltage tables need voltage_points_v")

    @property
    def roundtrip_time_s(self):
        return self.group_index * self.length_m / C

    @property
    def fsr_hz(self):
        """Reference first-order FSR; constant only when phase dispersion is zero."""
        return 1 / self.roundtrip_time_s

    @property
    def kappa_intrinsic_s_inv(self):
        return self.loss_power_np_per_m * C / self.group_index

    @property
    def kappa_bus_s_inv(self):
        return -math.log1p(-self.coupling_bus_power) / self.roundtrip_time_s

    @property
    def kappa_drop_s_inv(self):
        return -math.log1p(-self.coupling_drop_power) / self.roundtrip_time_s

    @property
    def kappa_total_s_inv(self):
        return self.kappa_intrinsic_s_inv + self.kappa_bus_s_inv + self.kappa_drop_s_inv

    @property
    def photon_lifetime_s(self):
        """Energy lifetime; field amplitude lifetime is twice this value."""
        return 1 / self.kappa_total_s_inv if self.kappa_total_s_inv else math.inf

    @property
    def linewidth_hz(self):
        """Pole/TCMT linewidth; exact periodic FWHM may differ at low finesse."""
        return self.kappa_total_s_inv / (2 * math.pi)

    @property
    def loaded_q(self):
        return self.reference_frequency_hz / self.linewidth_hz if self.linewidth_hz else math.inf

    @property
    def exact_fwhm_hz(self):
        """Local FWHM of drop enhancement (or all-pass notch depression).

        Returns NaN when a resonance never falls to half peak within one FSR.
        This prevents calling a low-finesse periodic spectrum a Lorentzian.
        """
        rho = math.exp(-self.kappa_total_s_inv * self.roundtrip_time_s / 2)
        if rho == 1:
            return 0.0
        if rho == 0:
            return math.nan
        argument = (1 - rho) / (2 * math.sqrt(rho))
        if argument > 1:
            return math.nan
        half_phase = 2 * math.asin(argument)
        tau = self.roundtrip_time_s
        frequencies = []
        for phi in (-half_phase, half_phase):
            discriminant = tau**2 + 2 * self.phase_dispersion_s2 * phi
            if discriminant <= 0:
                return math.nan
            # Cancellation-free local root with positive dphase/domega.
            frequencies.append(2 * phi / (tau + math.sqrt(discriminant)) / (2 * math.pi))
        return frequencies[1] - frequencies[0]


@dataclass(frozen=True)
class OpticalResponse:
    through_power_w: np.ndarray
    drop_power_w: np.ndarray
    absorbed_power_w: np.ndarray
    cavity_energy_j: np.ndarray
    resonance_frequency_hz: np.ndarray
    through_field: np.ndarray
    drop_field: np.ndarray
    dissipated_power_w: np.ndarray
    scattered_power_w: np.ndarray


def voltage_response(params: OpticalParams, voltage_v):
    """Return signed resonance shift (Hz) and total propagation loss (Np/m)."""
    v = np.asarray(voltage_v, dtype=float)
    if not np.all(np.isfinite(v)):
        raise ValueError("voltage must be finite")
    shift = params.df_dvoltage_hz_per_v * v
    loss = params.loss_power_np_per_m + params.loss_voltage_np_per_m_per_v * v
    if len(params.voltage_points_v):
        points = params.voltage_points_v
        if np.any(v < points[0]) or np.any(v > points[-1]):
            raise ValueError("junction voltage is outside supplied calibration table")
        if len(params.resonance_voltage_shifts_hz):
            shift = np.interp(v, points, params.resonance_voltage_shifts_hz)
        if len(params.loss_voltage_values_np_per_m):
            loss = np.interp(v, points, params.loss_voltage_values_np_per_m)
    if np.any(loss < 0) or not np.all(np.isfinite(loss)):
        raise ValueError("voltage-dependent loss is negative or nonfinite")
    return shift, loss


def resonance_frequency(params: OpticalParams, temperature_k=None, junction_voltage_v=0.0):
    temperature = (
        params.reference_temperature_k if temperature_k is None else np.asarray(temperature_k, dtype=float)
    )
    if np.any(np.asarray(temperature) <= 0) or not np.all(np.isfinite(temperature)):
        raise ValueError("temperature_k must be finite and positive")
    shift, _ = voltage_response(params, junction_voltage_v)
    frequency = (
        params.reference_frequency_hz
        + params.df_dtemperature_hz_per_k * (temperature - params.reference_temperature_k)
        + shift
    )
    if np.any(frequency <= 0):
        raise ValueError("resonance frequency must stay positive")
    return frequency


def optical_response(
    params: OpticalParams, laser_frequency_hz, temperature_k=None, junction_voltage_v=0.0, input_power_w=1.0
):
    """Exact periodic add-drop response, broadcasting frequency/T/V/power arrays.

    Complex fields have sqrt(W) units and powers W. The energy integrates the
    traveling power on both half-rings, including loss and the drop coupler.
    Phase dispersion uses the local group delay dphase/domega; a nonpositive
    local delay rejects invalid extrapolation.
    A completely isolated lossless ring has zero driven stored energy. No
    backscattering or nonlinear branches are silently folded into this model.
    """
    frequency, temperature, voltage, power = np.broadcast_arrays(
        np.asarray(laser_frequency_hz, dtype=float),
        params.reference_temperature_k if temperature_k is None else np.asarray(temperature_k, dtype=float),
        np.asarray(junction_voltage_v, dtype=float),
        np.asarray(input_power_w, dtype=float),
    )
    if np.any(frequency <= 0) or not np.all(np.isfinite(frequency)):
        raise ValueError("laser frequency must be finite and positive")
    if np.any(power < 0) or not np.all(np.isfinite(power)):
        raise ValueError("input power must be finite and nonnegative")
    f_res = resonance_frequency(params, temperature, voltage)
    _, alpha = voltage_response(params, voltage)
    omega = 2 * np.pi * (frequency - f_res)
    phase = omega * params.roundtrip_time_s + 0.5 * params.phase_dispersion_s2 * omega**2
    local_roundtrip_s = params.roundtrip_time_s + params.phase_dispersion_s2 * omega
    if np.any(local_roundtrip_s <= 0) or not np.all(np.isfinite(local_roundtrip_s)):
        raise ValueError("phase dispersion extrapolation gives nonpositive or nonfinite local group delay")
    a = np.exp(-alpha * params.length_m / 2)
    t1, t2 = np.sqrt(1 - params.coupling_bus_power), np.sqrt(1 - params.coupling_drop_power)
    z = a * np.exp(1j * phase)
    denominator = 1 - t1 * t2 * z
    if params.coupling_bus_power == 0:
        through = np.sqrt(power).astype(complex)
        zeros = np.zeros_like(power)
        return OpticalResponse(
            power, zeros, zeros, zeros, f_res, through, zeros.astype(complex), zeros, zeros
        )
    through = np.sqrt(power) * (t1 - t2 * z) / denominator
    drop = (
        -np.sqrt(power * params.coupling_bus_power * params.coupling_drop_power * a)
        * np.exp(1j * phase / 2)
        / denominator
    )
    p0 = power * params.coupling_bus_power / np.abs(denominator) ** 2
    # The first half-integral is length/2 at zero loss; expm1 avoids cancellation.
    half_integral = np.full_like(alpha, params.length_m / 2)
    np.divide(-np.expm1(-alpha * params.length_m / 2), alpha, out=half_integral, where=alpha != 0)
    energy = p0 * half_integral * (1 + t2**2 * a) * local_roundtrip_s / params.length_m
    dissipated = alpha * params.length_m / local_roundtrip_s * energy
    absorbed = params.intrinsic_absorption_fraction * dissipated
    return OpticalResponse(
        np.abs(through) ** 2,
        np.abs(drop) ** 2,
        absorbed,
        energy,
        f_res,
        through,
        drop,
        dissipated,
        dissipated - absorbed,
    )


@dataclass(frozen=True)
class NonlinearParams:
    """Effective TCMT coefficients, all disabled by default (uncalibrated).

    kappa_TPA = tpa_rate_per_j_s * U; P_TPA = kappa_TPA * U.
    TPA creates N at P_TPA/(2*h*f*carrier_volume_m3), with a quantum yield.
    FCA rate = sigma * v_g * N. FCD coefficient units are Hz per (m^-3);
    Kerr coefficient units Hz/J; both shift resonance, not laser frequency.
    Counterpropagating nonlinearity uses total energy and a uniform carrier
    density; spatial standing-wave gratings, Raman and FWM are unresolved.
    """

    tpa_rate_per_j_s: float = 0.0
    carrier_lifetime_s: float = 1e-9
    carrier_volume_m3: float = 1e-17
    carrier_quantum_yield: float = 1.0
    fca_cross_section_m2: float = 0.0
    fcd_hz_per_m3: float = 0.0
    kerr_hz_per_j: float = 0.0

    def __post_init__(self):
        for name in ("carrier_lifetime_s", "carrier_volume_m3"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("tpa_rate_per_j_s", "fca_cross_section_m2", "carrier_quantum_yield"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.carrier_quantum_yield > 1:
            raise ValueError("carrier_quantum_yield must lie in [0, 1]")
        for name in ("fcd_hz_per_m3", "kerr_hz_per_j"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")


def cavity_rhs(
    params,
    field_forward,
    field_reverse,
    carrier_density_m3,
    *,
    laser_frequency_hz,
    temperature_k=None,
    junction_voltage_v=0.0,
    input_field=0j,
    reverse_input_field=0j,
    backscatter_rad_per_s=0j,
    nonlinearity=None,
):
    """Continuous local two-mode dynamics; a has sqrt(J), input sqrt(W).

    da/dt=(i*(omega_laser-omega_res)-kappa/2)*a+i*g*b+sqrt(k_bus)*s.
    Reciprocal coupling uses conjugate(g) in the reverse equation so g alone
    never creates/destroys energy. Returns (da_forward, da_reverse, dN).
    """
    nl = NonlinearParams() if nonlinearity is None else nonlinearity
    if not math.isfinite(laser_frequency_hz) or laser_frequency_hz <= 0:
        raise ValueError("laser_frequency_hz must be finite and positive")
    if not all(
        np.isfinite(complex(value))
        for value in (field_forward, field_reverse, input_field, reverse_input_field, backscatter_rad_per_s)
    ):
        raise ValueError("cavity fields and backscatter must be finite")
    U = abs(field_forward) ** 2 + abs(field_reverse) ** 2
    N = float(carrier_density_m3)
    if N < 0 or not math.isfinite(N):
        raise ValueError("carrier density must be finite and nonnegative")
    _, alpha = voltage_response(params, junction_voltage_v)
    intrinsic_rate = float(alpha) * C / params.group_index
    k_tpa = nl.tpa_rate_per_j_s * U
    k_fca = nl.fca_cross_section_m2 * C / params.group_index * N
    f_res = (
        float(resonance_frequency(params, temperature_k, junction_voltage_v))
        + nl.kerr_hz_per_j * U
        + nl.fcd_hz_per_m3 * N
    )
    delta = 2 * np.pi * (laser_frequency_hz - f_res)
    decay = intrinsic_rate + params.kappa_bus_s_inv + params.kappa_drop_s_inv + k_tpa + k_fca
    lam = 1j * delta - decay / 2
    g = complex(backscatter_rad_per_s)
    da = lam * field_forward + 1j * g * field_reverse + np.sqrt(params.kappa_bus_s_inv) * input_field
    db = (
        lam * field_reverse
        + 1j * g.conjugate() * field_forward
        + np.sqrt(params.kappa_bus_s_inv) * reverse_input_field
    )
    dN = (
        nl.carrier_quantum_yield * k_tpa * U / (2 * H_PLANCK * laser_frequency_hz * nl.carrier_volume_m3)
        - N / nl.carrier_lifetime_s
    )
    return da, db, dN


@dataclass(frozen=True)
class NonlinearSteadyState:
    cavity_energy_j: float
    carrier_density_m3: float
    field: complex
    stable: bool
    eigenvalues_s_inv: np.ndarray
    relative_residual: float


def nonlinear_steady_states(
    params,
    laser_frequency_hz,
    input_power_w,
    *,
    temperature_k=None,
    junction_voltage_v=0.0,
    nonlinearity=None,
):
    """Enumerate single-mode nonlinear branches and Jacobian stability.

    The energy equation is a real polynomial of at most degree five. Carrier
    equilibrium and complex-field dynamics are retained in the stability test;
    the positive-slope energy rule alone cannot diagnose self-pulsation. No
    thermal state or backscattering is included in this branch calculation.
    """
    nl = NonlinearParams() if nonlinearity is None else nonlinearity
    if not math.isfinite(input_power_w) or input_power_w <= 0 or params.kappa_bus_s_inv == 0:
        raise ValueError("branch analysis requires positive power and bus coupling")
    if not math.isfinite(laser_frequency_hz) or laser_frequency_hz <= 0:
        raise ValueError("laser_frequency_hz must be finite and positive")
    _, alpha = voltage_response(params, junction_voltage_v)
    k0 = float(alpha) * C / params.group_index + params.kappa_bus_s_inv + params.kappa_drop_s_inv
    drive = math.sqrt(params.kappa_bus_s_inv * input_power_w)
    Uscale = drive**2 / (k0 / 2) ** 2
    carrier_per_u2 = (
        nl.carrier_quantum_yield
        * nl.tpa_rate_per_j_s
        * nl.carrier_lifetime_s
        / (2 * H_PLANCK * laser_frequency_hz * nl.carrier_volume_m3)
    )
    Nscale = max(carrier_per_u2 * Uscale**2, 1.0)
    base_delta = (
        2
        * np.pi
        * (laser_frequency_hz - float(resonance_frequency(params, temperature_k, junction_voltage_v)))
    )
    x = Polynomial([0.0, 1.0])
    Npoly = carrier_per_u2 * Uscale**2 * x**2
    decay = (
        Polynomial([k0])
        + nl.tpa_rate_per_j_s * Uscale * x
        + nl.fca_cross_section_m2 * C / params.group_index * Npoly
    )
    delta = (
        Polynomial([base_delta])
        - 2 * np.pi * nl.kerr_hz_per_j * Uscale * x
        - 2 * np.pi * nl.fcd_hz_per_m3 * Npoly
    )
    equation = x * ((decay / k0 / 2) ** 2 + (delta / k0) ** 2) - drive**2 / Uscale / k0**2
    result = []
    for root in equation.roots():
        if abs(root.imag) > 1e-7 * max(1, abs(root.real)) or root.real <= 0:
            continue
        U = float(root.real * Uscale)
        N = carrier_per_u2 * U**2
        k = k0 + nl.tpa_rate_per_j_s * U + nl.fca_cross_section_m2 * C / params.group_index * N
        d = base_delta - 2 * np.pi * (nl.kerr_hz_per_j * U + nl.fcd_hz_per_m3 * N)
        a = drive / (k / 2 - 1j * d)
        residual = abs(U * ((k / 2) ** 2 + d**2) - drive**2) / drive**2
        if residual > 1e-5:
            continue
        # Scaled coordinates avoid differencing 1e-8 amplitudes against 1e22 densities.
        point = np.array([a.real / math.sqrt(Uscale), a.imag / math.sqrt(Uscale), N / Nscale])

        def rhs_scaled(y):
            field = complex(y[0], y[1]) * math.sqrt(Uscale)
            energy = abs(field) ** 2
            density = y[2] * Nscale
            k = k0 + nl.tpa_rate_per_j_s * energy + nl.fca_cross_section_m2 * C / params.group_index * density
            delta = base_delta - 2 * np.pi * (nl.kerr_hz_per_j * energy + nl.fcd_hz_per_m3 * density)
            da = (1j * delta - k / 2) * field + drive
            dn = (carrier_per_u2 * energy**2 - density) / nl.carrier_lifetime_s
            return np.array([da.real / math.sqrt(Uscale), da.imag / math.sqrt(Uscale), dn / Nscale])

        jac = np.empty((3, 3))
        for j in range(3):
            step = 1e-6 * max(1, abs(point[j]))
            perturb = np.zeros(3)
            perturb[j] = step
            jac[:, j] = (rhs_scaled(point + perturb) - rhs_scaled(point - perturb)) / (2 * step)
        eigenvalues = np.linalg.eigvals(jac)
        result.append(
            NonlinearSteadyState(U, N, a, bool(np.all(eigenvalues.real < 0)), eigenvalues, residual)
        )
    return sorted(result, key=lambda state: state.cavity_energy_j)


def select_nonlinear_branch(branches, previous_energy_j=None):
    """Follow the nearest stable branch; require a history when ambiguous."""
    stable = [state for state in branches if state.stable]
    if not stable:
        raise ValueError("no dynamically stable branch exists")
    if len(stable) > 1 and previous_energy_j is None:
        raise ValueError("multiple stable branches: supply previous_energy_j or evolve the dynamic model")
    return (
        min(stable, key=lambda state: abs(state.cavity_energy_j - previous_energy_j))
        if previous_energy_j is not None
        else stable[0]
    )
