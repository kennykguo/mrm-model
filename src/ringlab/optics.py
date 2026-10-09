"""Passive, exact per-pass ring optics with explicitly stated compact assumptions.

Loss is a *power* attenuation coefficient in Np/m; couplings are power fractions.
This is an add-drop traveling-wave ring with two lossless point couplers separated
by half a circumference. Set ``drop_coupling_power=0`` for an all-pass ring.
Defaults are illustrative, not a reconstruction of a Lightmatter device.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

C0 = 299_792_458.0


@dataclass(frozen=True)
class RingParameters:
    reference_wavelength_m: float = 1310e-9
    roundtrip_length_m: float = 2 * np.pi * 10e-6
    group_index: float = 4.0
    intrinsic_loss_per_m: float = 250.0
    input_coupling_power: float = 0.025
    drop_coupling_power: float = 0.005
    absorption_fraction: float = 0.8
    reference_temperature_k: float = 300.0
    thermal_frequency_slope_hz_per_k: float = -15e9
    thermal_frequency_curvature_hz_per_k2: float = 0.0
    beta2_s2_per_m: float = 0.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("reference_wavelength_m", "roundtrip_length_m", "group_index", "reference_temperature_k"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.intrinsic_loss_per_m < 0:
            raise ValueError("intrinsic_loss_per_m must be nonnegative")
        if not 0 < self.input_coupling_power < 1:
            raise ValueError("input_coupling_power must lie strictly between 0 and 1")
        if not 0 <= self.drop_coupling_power < 1:
            raise ValueError("drop_coupling_power must lie in [0, 1)")
        if not 0 <= self.absorption_fraction <= 1:
            raise ValueError("absorption_fraction must lie in [0, 1]")

    @property
    def reference_frequency_hz(self) -> float:
        return C0 / self.reference_wavelength_m

    @property
    def roundtrip_time_s(self) -> float:
        return self.group_index * self.roundtrip_length_m / C0

    @property
    def fsr_hz(self) -> float:
        """Local free spectral range at the reference resonance."""
        return 1.0 / self.roundtrip_time_s


@dataclass(frozen=True)
class OpticalResponse:
    through_fraction: NDArray
    drop_fraction: NDArray
    intrinsic_loss_fraction: NDArray
    absorbed_fraction: NDArray
    through_power_w: NDArray
    drop_power_w: NDArray
    intrinsic_loss_power_w: NDArray
    absorbed_power_w: NDArray
    stored_energy_j: NDArray
    resonance_frequency_hz: NDArray
    roundtrip_phase_rad: NDArray
    through_field: NDArray
    drop_field: NDArray


@dataclass(frozen=True)
class NonlinearScreening:
    """First-order diagnostics evaluated on the *linear* steady-state field.

    These are not guaranteed bounds on a nonlinear solution and do not update
    the returned linear spectrum or thermal state.
    """

    peak_circulating_power_w: NDArray
    peak_effective_intensity_w_per_m2: NDArray
    estimated_tpa_power_w: NDArray
    estimated_tpa_pair_generation_per_s: NDArray
    tpa_to_linear_intrinsic_loss_ratio: NDArray
    tpa_decay_to_loaded_decay_ratio: NDArray
    estimated_kerr_frequency_shift_hz: NDArray
    absolute_kerr_shift_over_linewidth: NDArray


def _finite(value: ArrayLike, name: str) -> NDArray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def resonance_frequency(
    params: RingParameters, temperature_k: ArrayLike, resonance_shift_hz: ArrayLike = 0.0
) -> NDArray:
    """Empirical thermal resonance law, plus an externally supplied junction shift.

    The curvature coefficient multiplies delta_T**2 (there is no factor 1/2).
    A measured frequency slope already incorporates thermo-optic and expansion
    effects; an additional expansion shift must not be added on top of it.
    """
    temperature = _finite(temperature_k, "temperature_k")
    shift = _finite(resonance_shift_hz, "resonance_shift_hz")
    if np.any(temperature <= 0):
        raise ValueError("temperature_k must be positive")
    delta = temperature - params.reference_temperature_k
    frequency = (
        params.reference_frequency_hz
        + params.thermal_frequency_slope_hz_per_k * delta
        + params.thermal_frequency_curvature_hz_per_k2 * delta**2
        + shift
    )
    if np.any(frequency <= 0):
        raise ValueError("resonance frequency must remain positive")
    return np.asarray(frequency)


def decay_rates(params: RingParameters, additional_loss_per_m: ArrayLike = 0.0) -> tuple[NDArray, float, float]:
    """(intrinsic, input, drop) *energy* decay rates, in s^-1, at resonance.

    Logarithms match the per-roundtrip energy decay, not an approximate k/tau.
    These rates give an isolated-resonance TCMT model, not an exact broadband
    substitute for the discrete couplers and roundtrip delay.
    """
    alpha = params.intrinsic_loss_per_m + _finite(additional_loss_per_m, "additional_loss_per_m")
    if np.any(alpha < 0):
        raise ValueError("total intrinsic power loss must be nonnegative")
    tau = params.roundtrip_time_s
    return (
        np.asarray(alpha * params.roundtrip_length_m / tau),
        float(-np.log1p(-params.input_coupling_power) / tau),
        float(-np.log1p(-params.drop_coupling_power) / tau),
    )


def loaded_linewidth_hz(params: RingParameters, additional_loss_per_m: ArrayLike = 0.0) -> NDArray:
    """Exact half-maximum width of intracavity/drop *power* about this resonance.

    The zero-dispersion result is 2*FSR/pi*asin((1-rho)/(2*sqrt(rho))).
    A very broad ring may never reach half its maximum within an FSR; in that
    case a linewidth is not defined and ValueError is raised.
    """
    intrinsic, k1, k2 = decay_rates(params, additional_loss_per_m)
    rho = np.exp(-0.5 * (intrinsic + k1 + k2) * params.roundtrip_time_s)
    argument = (1 - rho) / (2 * np.sqrt(rho))
    if np.any(argument > 1):
        raise ValueError("ring has no resolved half-maximum within one FSR")
    phase_half = 2 * np.arcsin(argument)
    tau = params.roundtrip_time_s
    dispersion = params.beta2_s2_per_m * params.roundtrip_length_m

    def detuning(phase: NDArray) -> NDArray:
        discriminant = tau**2 + 2 * dispersion * phase
        if np.any(discriminant <= 0):
            raise ValueError("dispersion expansion invalid at half-maximum")
        return 2 * phase / (tau + np.sqrt(discriminant))

    return np.asarray((detuning(phase_half) - detuning(-phase_half)) / (2 * np.pi))


def optical_response(
    params: RingParameters,
    frequency_hz: ArrayLike,
    *,
    temperature_k: ArrayLike = 300.0,
    resonance_shift_hz: ArrayLike = 0.0,
    additional_loss_per_m: ArrayLike = 0.0,
    input_power_w: ArrayLike = 1.0,
) -> OpticalResponse:
    """Exact passive steady state; array-valued arguments follow NumPy broadcasting.

    Complex port fields are transfer amplitudes normalized to the input field.
    Heat is only ``absorption_fraction`` times intrinsic lost power: scattered
    light is not silently treated as local heat. Stored energy integrates the
    decaying traveling-wave power around both half-ring sections.
    """
    frequency = _finite(frequency_hz, "frequency_hz")
    pin = _finite(input_power_w, "input_power_w")
    alpha = params.intrinsic_loss_per_m + _finite(additional_loss_per_m, "additional_loss_per_m")
    if np.any(frequency <= 0) or np.any(pin < 0) or np.any(alpha < 0):
        raise ValueError("frequency must be positive, and power/total loss nonnegative")
    fres = resonance_frequency(params, temperature_k, resonance_shift_hz)
    frequency, fres, pin, alpha = np.broadcast_arrays(frequency, fres, pin, alpha)
    domega = 2 * np.pi * (frequency - fres)
    phase = domega * params.roundtrip_time_s + 0.5 * params.beta2_s2_per_m * params.roundtrip_length_m * domega**2
    group_delay = params.roundtrip_time_s + params.beta2_s2_per_m * params.roundtrip_length_m * domega
    if np.any(group_delay <= 0):
        raise ValueError("dispersion expansion gives nonpositive group delay; outside model domain")
    a = np.exp(-0.5 * alpha * params.roundtrip_length_m)
    r1 = np.sqrt(1 - params.input_coupling_power)
    r2 = np.sqrt(1 - params.drop_coupling_power)
    phase_factor = np.exp(-1j * phase)
    denominator = 1 - r1 * r2 * a * phase_factor
    through = (r1 - r2 * a * phase_factor) / denominator
    drop = -np.sqrt(params.input_coupling_power * params.drop_coupling_power * a) * np.exp(-0.5j * phase) / denominator
    circulating_fraction = params.input_coupling_power / np.abs(denominator) ** 2
    half_loss = -np.expm1(-0.5 * alpha * params.roundtrip_length_m)
    loss_fraction = circulating_fraction * half_loss * (1 + r2**2 * a)
    # Integral of exp(-alpha*z) over a half-circle; the series limit handles alpha=0.
    half_integral = np.full(alpha.shape, 0.5 * params.roundtrip_length_m)
    np.divide(half_loss, alpha, out=half_integral, where=alpha != 0)
    energy = pin * circulating_fraction * half_integral * (1 + r2**2 * a) * group_delay / params.roundtrip_length_m
    t_fraction, d_fraction = np.abs(through) ** 2, np.abs(drop) ** 2
    heat_fraction = params.absorption_fraction * loss_fraction
    return OpticalResponse(
        t_fraction, d_fraction, loss_fraction, heat_fraction,
        pin * t_fraction, pin * d_fraction, pin * loss_fraction,
        pin * heat_fraction, energy, fres, phase, through, drop,
    )


def nonlinear_screening(
    params: RingParameters,
    frequency_hz: ArrayLike,
    *,
    effective_mode_area_m2: float,
    tpa_coefficient_m_per_w: float,
    kerr_index_m2_per_w: float,
    temperature_k: ArrayLike = 300.0,
    resonance_shift_hz: ArrayLike = 0.0,
    additional_loss_per_m: ArrayLike = 0.0,
    input_power_w: ArrayLike = 1.0,
) -> NonlinearScreening:
    """Screen omitted TPA/Kerr effects with caller-supplied material/mode data.

    No wavelength-dependent nonlinear coefficient or mode area is invented.
    Coefficients and the effective area must use mutually compatible modal
    overlap conventions; one common area is assumed for Kerr and TPA here.
    A detailed mode extraction may require different nonlinear overlap areas.

    P_TPA = integral(beta_TPA*P(z)^2/A_eff dz), evaluated on the linear field.
    delta_f_Kerr = -f/ng * n2*mean_z(P(z))/A_eff. The integration accounts for
    distributed loss and the drop-coupler discontinuity. Peak intensity is an
    effective traveling-wave value, not a resolved material hot-spot intensity.
    The photon-pair generation estimate divides TPA power by 2*h*f; no carrier
    density, lifetime, FCA/FCD or thermalization fraction is inferred.

    Large ratios call for a self-consistent nonlinear model. Even small ratios
    at one linear operating point do not exclude a distant nonlinear branch,
    optical bistability, carrier accumulation or multimode effects. These
    diagnostics are therefore perturbation estimates, NOT validity bounds.
    """
    if not np.isfinite(effective_mode_area_m2) or effective_mode_area_m2 <= 0:
        raise ValueError("effective_mode_area_m2 must be positive and finite")
    if not np.isfinite(tpa_coefficient_m_per_w) or tpa_coefficient_m_per_w < 0:
        raise ValueError("tpa_coefficient_m_per_w must be finite and nonnegative")
    if not np.isfinite(kerr_index_m2_per_w):
        raise ValueError("kerr_index_m2_per_w must be finite")
    linear = optical_response(
        params, frequency_hz, temperature_k=temperature_k,
        resonance_shift_hz=resonance_shift_hz,
        additional_loss_per_m=additional_loss_per_m, input_power_w=input_power_w,
    )
    frequency, pin, alpha, phase = np.broadcast_arrays(
        _finite(frequency_hz, "frequency_hz"), _finite(input_power_w, "input_power_w"),
        params.intrinsic_loss_per_m + _finite(additional_loss_per_m, "additional_loss_per_m"),
        linear.roundtrip_phase_rad,
    )
    attenuation = np.exp(-alpha * params.roundtrip_length_m / 2)
    r2_squared = 1 - params.drop_coupling_power
    denominator = np.abs(1 - np.sqrt((1 - params.input_coupling_power) * r2_squared) * attenuation * np.exp(-1j * phase))**2
    peak_power = pin * params.input_coupling_power / denominator
    half_squared_power_integral = np.full(alpha.shape, params.roundtrip_length_m / 2)
    np.divide(-np.expm1(-alpha * params.roundtrip_length_m), 2 * alpha, out=half_squared_power_integral, where=alpha != 0)
    tpa_power = tpa_coefficient_m_per_w / effective_mode_area_m2 * peak_power**2 * half_squared_power_integral * (1 + r2_squared**2 * attenuation**2)
    domega = 2 * np.pi * (frequency - linear.resonance_frequency_hz)
    group_delay = params.roundtrip_time_s + params.beta2_s2_per_m * params.roundtrip_length_m * domega
    group_index = C0 * group_delay / params.roundtrip_length_m
    mean_power = linear.stored_energy_j / group_delay
    kerr_shift = -frequency / group_index * kerr_index_m2_per_w * mean_power / effective_mode_area_m2
    ki, k1, k2 = decay_rates(params, additional_loss_per_m)
    # Correct the reference decay rates for local group delay when beta2 != 0.
    loaded_decay = (ki + k1 + k2) * params.roundtrip_time_s / group_delay
    intrinsic_ratio = np.divide(tpa_power, linear.intrinsic_loss_power_w, out=np.full(tpa_power.shape, np.inf), where=linear.intrinsic_loss_power_w > 0)
    intrinsic_ratio = np.where(tpa_power == 0, 0.0, intrinsic_ratio)
    decay_ratio = np.divide(tpa_power, loaded_decay * linear.stored_energy_j, out=np.zeros(tpa_power.shape), where=linear.stored_energy_j > 0)
    planck_j_s = 6.62607015e-34
    return NonlinearScreening(
        peak_power, peak_power / effective_mode_area_m2, tpa_power,
        tpa_power / (2 * planck_j_s * frequency), intrinsic_ratio, decay_ratio,
        kerr_shift, np.abs(kerr_shift) / loaded_linewidth_hz(params, additional_loss_per_m),
    )
