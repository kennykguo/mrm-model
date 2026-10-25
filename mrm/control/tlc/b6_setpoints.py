"""Block B6: Red-Flank Setpoint Generator for Lightmatter G26 Thermal Locking Controller (TLC).

Calculates and manages optimal detuning setpoints on the red flank of resonance
where the second derivative of the data-averaged drop transmission vanishes:
    d^2 L_avg / d(delta)^2 = 0
providing:
1. Maximum transmission slope magnitude |dL/d(delta)| for highest lock-in demodulation SNR.
2. Passive self-stabilizing negative thermo-optic feedback (dL/d(delta) < 0) preventing thermal runaway.
3. Maximum linearity and dynamic range by eliminating 2nd-order distortion in the error signal.

Nominal Plant Parameters (Lightmatter G26):
- Optical wavelength: lambda_0 = 1310.0 nm
- Loaded quality factor: Q_loaded = 3850.0
- Resonance FWHM: Delta lambda = lambda_0 / Q = 340.26 pm
- Resonance HWHM: gamma = FWHM / 2 = 170.13 pm
- NRZ modulation shift: mod_shift = 50.0 pm
- Nominal inflection setpoint: delta* ~ +124.62 pm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

try:
    from scipy.optimize import brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

__all__ = ["SetpointConfig", "SetpointGenerator", "B6Setpoints"]


@dataclass
class SetpointConfig:
    """Configuration parameters for Block B6 Setpoint Generator.

    Attributes:
        n_rings: Number of microring resonator channels (default 8 for G26).
        q_loaded: Nominal loaded quality factor (default 3850.0).
        wavelength_nm: Laser carrier wavelength in nm (default 1310.0 nm).
        fwhm_pm: Optical resonance full-width at half-maximum in pm. If None,
            derived from wavelength_nm and q_loaded (1310.0 / 3850.0 * 1000.0 = 340.26 pm).
        mod_shift_pm: Electro-optic NRZ modulation detuning shift in pm (default 50.0 pm).
        frac_ones: Fraction of mark bits ('1') in high-speed data stream (default 0.5).
        target_override_pm: Optional manual override setpoint in pm.
        channel_offsets_pm: Optional per-channel calibration or OMA-balancing offsets in pm.
        solver_tol_pm: Root-finding convergence tolerance in pm (default 1e-6).
        max_iter: Maximum iterations for root solver (default 100).
    """

    n_rings: int = 8
    q_loaded: float = 3850.0
    wavelength_nm: float = 1310.0
    fwhm_pm: float | None = None
    mod_shift_pm: float = 50.0
    frac_ones: float = 0.5
    target_override_pm: float | np.ndarray | None = None
    channel_offsets_pm: np.ndarray | None = None
    solver_tol_pm: float = 1e-6
    max_iter: int = 100


class SetpointGenerator:
    """Block B6: Red-Flank Setpoint Generator for multi-ring microring modulators.

    Generates optimal operating detunings delta* on the red flank (delta > 0) of the
    data-averaged optical resonance lineshape. Supports vectorized batch generation (B, N)
    and optional per-channel OMA balancing.
    """

    def __init__(
        self,
        n_rings: int = 8,
        q_loaded: float = 3850.0,
        wavelength_nm: float = 1310.0,
        fwhm_pm: float | Sequence[float] | np.ndarray | None = None,
        mod_shift_pm: float = 50.0,
        target_override_pm: float | Sequence[float] | np.ndarray | None = None,
        channel_offsets_pm: Sequence[float] | np.ndarray | None = None,
        config: SetpointConfig | None = None,
        **kwargs,
    ) -> None:
        """Initialize SetpointGenerator.

        Accepts either explicit keyword arguments, a SetpointConfig instance, or kwargs.
        """
        if config is not None:
            self.config = config
            self.n_rings = int(config.n_rings)
            self.q_loaded = float(config.q_loaded)
            self.wavelength_nm = float(config.wavelength_nm)
            self.mod_shift_pm = float(config.mod_shift_pm)
            self.frac_ones = float(config.frac_ones)
            override = config.target_override_pm
            offsets = config.channel_offsets_pm
            _fwhm_in = config.fwhm_pm
        else:
            self.n_rings = int(n_rings)
            self.q_loaded = float(q_loaded)
            self.wavelength_nm = float(wavelength_nm)
            self.mod_shift_pm = float(kwargs.get("mod_shift_pm", mod_shift_pm))
            self.frac_ones = float(kwargs.get("frac_ones", 0.5))
            override = kwargs.get("target_override_pm", target_override_pm)
            offsets = kwargs.get("channel_offsets_pm", channel_offsets_pm)
            _fwhm_in = kwargs.get("fwhm_pm", fwhm_pm)

        # Resolve FWHM and HWHM (scalar or vector across rings)
        if _fwhm_in is not None:
            fwhm_arr = np.asarray(_fwhm_in, dtype=np.float64)
            if fwhm_arr.ndim == 0:
                self.fwhm_pm = float(fwhm_arr)
                self._fwhm_vec = np.full(self.n_rings, self.fwhm_pm, dtype=np.float64)
            else:
                if len(fwhm_arr) != self.n_rings:
                    raise ValueError(f"fwhm_pm array length ({len(fwhm_arr)}) must match n_rings ({self.n_rings})")
                self._fwhm_vec = fwhm_arr.astype(np.float64)
                self.fwhm_pm = float(np.mean(self._fwhm_vec))
        else:
            self.fwhm_pm = (self.wavelength_nm / self.q_loaded) * 1000.0  # pm
            self._fwhm_vec = np.full(self.n_rings, self.fwhm_pm, dtype=np.float64)

        self.hwhm_pm = self.fwhm_pm / 2.0
        self._hwhm_vec = self._fwhm_vec / 2.0

        # Handle target overrides
        self.target_override_pm = override

        # Handle channel offsets
        if offsets is not None:
            off_arr = np.asarray(offsets, dtype=np.float64)
            if off_arr.ndim == 0:
                self.channel_offsets_pm = np.full(self.n_rings, float(off_arr), dtype=np.float64)
            elif len(off_arr) == self.n_rings:
                self.channel_offsets_pm = off_arr.astype(np.float64)
            else:
                raise ValueError(f"channel_offsets_pm length ({len(off_arr)}) must match n_rings ({self.n_rings})")
        else:
            self.channel_offsets_pm = np.zeros(self.n_rings, dtype=np.float64)

        # Compute per-channel raw inflection points
        self._raw_targets = np.array([
            self.compute_inflection_point(h, self.mod_shift_pm, self.frac_ones)
            for h in self._hwhm_vec
        ], dtype=np.float64)

        self.nominal_inflection_pm = float(np.mean(self._raw_targets))

    @property
    def targets_pm(self) -> np.ndarray:
        """Returns per-ring target detunings in pm, shape (N,)."""
        if self.target_override_pm is not None:
            override_arr = np.asarray(self.target_override_pm, dtype=np.float64)
            if override_arr.ndim == 0:
                return np.full(self.n_rings, float(override_arr), dtype=np.float64) + self.channel_offsets_pm
            return override_arr + self.channel_offsets_pm
        return self._raw_targets + self.channel_offsets_pm

    @classmethod
    def compute_inflection_point(
        cls,
        hwhm_pm: float,
        mod_shift_pm: float = 50.0,
        frac_ones: float = 0.5,
    ) -> float:
        """Computes the red-flank inflection point delta* where d^2 L_avg / d(delta)^2 = 0.

        For a bare Lorentzian (mod_shift_pm == 0), the analytic solution is:
            delta* = hwhm_pm / sqrt(3) ~ 0.57735 * hwhm_pm.

        For an NRZ-modulated Lorentzian with mark fraction p1:
            L_avg(delta) = p1 / (1 + ((delta + mod_shift) / hwhm)^2)
                         + (1 - p1) / (1 + ((delta - mod_shift) / hwhm)^2)
        The inflection point is solved via Newton-Raphson with exact analytic derivatives,
        with robust Brent's method / bisection bracket fallback.

        Args:
            hwhm_pm: Half-width at half-maximum of resonance in pm.
            mod_shift_pm: NRZ modulation shift in pm (default 50.0 pm).
            frac_ones: Fraction of mark bits in data stream (default 0.5).

        Returns:
            Optimal red-flank detuning setpoint delta* in pm.
        """
        gamma = float(hwhm_pm)
        m = float(mod_shift_pm)
        p1 = float(frac_ones)
        p0 = 1.0 - p1

        if gamma <= 0.0:
            raise ValueError(f"hwhm_pm must be positive, got {hwhm_pm}")

        # Bare Lorentzian fallback: exact analytic root
        if abs(m) < 1e-9:
            return gamma / np.sqrt(3.0)

        # Define 2nd derivative kernel f(u) = (3u^2 - gamma^2) / (gamma^2 + u^2)^3
        def d2_avg(delta: float) -> float:
            u1 = delta + m
            u0 = delta - m
            d2_1 = (3.0 * u1**2 - gamma**2) / (gamma**2 + u1**2)**3
            d2_0 = (3.0 * u0**2 - gamma**2) / (gamma**2 + u0**2)**3
            return p1 * d2_1 + p0 * d2_0

        # Define 3rd derivative kernel f'(u) = -12u(u^2 - gamma^2) / (gamma^2 + u^2)^4
        def d3_avg(delta: float) -> float:
            u1 = delta + m
            u0 = delta - m
            d3_1 = -12.0 * u1 * (u1**2 - gamma**2) / (gamma**2 + u1**2)**4
            d3_0 = -12.0 * u0 * (u0**2 - gamma**2) / (gamma**2 + u0**2)**4
            return p1 * d3_1 + p0 * d3_0

        # Initial guess from modulated expansion: delta_0 = sqrt(gamma^2 / 3 + m^2)
        delta = np.sqrt(gamma**2 / 3.0 + m**2)

        # High-order Newton-Raphson iteration with machine precision
        converged = False
        for _ in range(30):
            f_val = d2_avg(delta)
            df_val = d3_avg(delta)
            if abs(df_val) < 1e-30:
                break
            step = f_val / df_val
            delta -= step
            if abs(step) < 1e-12:
                converged = True
                break

        # Fallback to Brent's method if Newton steps diverge or stay invalid
        if not converged or delta <= 0.0 or not np.isfinite(delta):
            low = max(0.01, gamma / np.sqrt(3.0) * 0.5)
            high = max(low + 10.0, gamma * 3.0 + m)
            if _HAS_SCIPY:
                delta = float(brentq(d2_avg, low, high, xtol=1e-12, rtol=1e-12))
            else:
                # Bisection fallback
                for _ in range(60):
                    mid = 0.5 * (low + high)
                    if d2_avg(mid) * d2_avg(low) <= 0:
                        high = mid
                    else:
                        low = mid
                delta = 0.5 * (low + high)

        return float(delta)

    def calculate_inflection_setpoint(self, ring_idx: int | None = None) -> float:
        """Returns the calculated red-flank inflection setpoint in pm.

        Args:
            ring_idx: Optional ring channel index (0 to N-1). If None, returns
                the nominal setpoint averaged across channels.

        Returns:
            Inflection target in pm (~ +124 pm for G26 nominal Q=3850).
        """
        targets = self.targets_pm
        if ring_idx is not None:
            return float(targets[ring_idx])
        return float(np.mean(targets))

    def get_setpoints(self, batch_size: int = 1) -> np.ndarray:
        """Returns vectorized setpoint array of shape (batch_size, n_rings).

        Args:
            batch_size: Number of parallel simulation / controller worlds (default 1).

        Returns:
            Detuning setpoint targets in pm with shape (B, N), type np.float64.
        """
        b = int(batch_size)
        if b < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        targets = self.targets_pm  # shape (N,)
        return np.tile(targets, (b, 1)).astype(np.float64)

    def set_target_override(self, override_pm: float | Sequence[float] | np.ndarray | None) -> None:
        """Sets or clears an explicit detuning setpoint override."""
        self.target_override_pm = override_pm

    def set_channel_offsets(self, offsets_pm: Sequence[float] | np.ndarray) -> None:
        """Sets per-channel detuning offsets in pm (e.g. from OMA equalization)."""
        arr = np.asarray(offsets_pm, dtype=np.float64)
        if arr.ndim == 0:
            self.channel_offsets_pm = np.full(self.n_rings, float(arr), dtype=np.float64)
        elif len(arr) == self.n_rings:
            self.channel_offsets_pm = arr.astype(np.float64)
        else:
            raise ValueError(f"offsets_pm length ({len(arr)}) must match n_rings ({self.n_rings})")

    def balance_oma(
        self,
        depth: Sequence[float] | np.ndarray,
        laser_power_mw: Sequence[float] | np.ndarray,
        oma_shift_max_pm: float = 30.0,
        er_min_db: float = 6.0,
        er_check_window_pm: float = 10.0,
        grid_points: int = 81,
    ) -> np.ndarray:
        """Balances Optical Modulation Amplitude (OMA) across rings within ER constraints.

        Searches for per-ring detuning offsets s_i in [-oma_shift_max, +oma_shift_max]
        that minimize the cross-channel OMA spread while ensuring Extinction Ratio (ER)
        remains >= er_min_db over [s_i - w, s_i + w].

        Args:
            depth: Resonance optical extinction depth per ring (0 to 1).
            laser_power_mw: Laser input power per ring in mW.
            oma_shift_max_pm: Maximum permissible shift away from raw inflection in pm.
            er_min_db: Minimum required extinction ratio in dB (default 6.0 dB).
            er_check_window_pm: Stability window around shifted target for ER verification.
            grid_points: Number of discrete shift search points.

        Returns:
            Optimal per-channel shifts in pm, shape (N,). Also updates self.channel_offsets_pm.
        """
        depth_arr = np.asarray(depth, dtype=np.float64)
        p_mw_arr = np.asarray(laser_power_mw, dtype=np.float64)
        base = self._raw_targets
        hwhm = self._hwhm_vec
        mod = self.mod_shift_pm
        n = self.n_rings

        grid = np.linspace(-oma_shift_max_pm, oma_shift_max_pm, grid_points)

        # Through levels helper
        def through(delta, h, d):
            x1 = (delta + mod) / h
            x0 = (delta - mod) / h
            l1 = 1.0 / (1.0 + x1 * x1)
            l0 = 1.0 / (1.0 + x0 * x0)
            t_hi = np.maximum(1.0 - d * l1, 1.0 - d * l0)
            t_lo = np.maximum(1e-12, np.minimum(1.0 - d * l1, 1.0 - d * l0))
            return t_hi, t_lo

        # Compute OMA grid
        oma = np.zeros((n, grid_points), dtype=np.float64)
        er_ok = np.zeros((n, grid_points), dtype=bool)

        for i in range(n):
            b_i = base[i]
            h_i = hwhm[i]
            d_i = depth_arr[i]
            p_i = p_mw_arr[i]

            for g_idx, s in enumerate(grid):
                t_hi, t_lo = through(b_i + s, h_i, d_i)
                oma_lin = p_i * np.maximum(1e-12, t_hi - t_lo)
                oma[i, g_idx] = 10.0 * np.log10(oma_lin)

                # Check ER constraint over +/- window
                er_valid = True
                for e in (-er_check_window_pm, 0.0, er_check_window_pm):
                    thi_e, tlo_e = through(b_i + s + e, h_i, d_i)
                    er_val = 10.0 * np.log10(thi_e / tlo_e)
                    if er_val < er_min_db:
                        er_valid = False
                        break
                er_ok[i, g_idx] = er_valid

        oma_masked = np.where(er_ok, oma, np.nan)
        valid_targets = oma_masked[np.isfinite(oma_masked)]

        if len(valid_targets) == 0:
            # Fallback: keep unshifted targets if no point meets strict ER constraint
            return self.channel_offsets_pm

        t_min, t_max = np.min(valid_targets), np.max(valid_targets)
        best_cost = float("inf")
        best_shifts = np.zeros(n, dtype=np.float64)

        for target in np.linspace(t_min, t_max, 200):
            diff = np.abs(oma_masked - target)
            # Find nearest valid shift per ring
            if np.any(np.all(np.isnan(diff), axis=1)):
                continue
            idx = np.nanargmin(diff, axis=1)
            vals = oma[np.arange(n), idx]
            spread = np.max(vals) - np.min(vals)
            cost = spread + 0.02 * np.sum(np.abs(grid[idx]))
            if cost < best_cost:
                best_cost = cost
                best_shifts = grid[idx].copy()

        self.set_channel_offsets(best_shifts)
        return best_shifts


# Alias conforming to reference TLC nomenclature
B6Setpoints = SetpointGenerator
