"""Block B2 Signal Conditioner: FIR filtering, noise estimation, and squelch detection.

Ports the reference ECE496 B2 conditioner into mrm-model for the 8-ring Lightmatter G26
architecture, supporting vectorized processing across batch worlds (B, N, M).
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi

__all__ = ["ConditionerConfig", "SignalConditioner", "B2Conditioner"]


@dataclasses.dataclass(frozen=True)
class ConditionerConfig:
    """Configuration parameters for Block B2 Signal Conditioner.

    Attributes:
        adc_rate_hz: Monitor ADC sampling rate per ring in Hz (default 250 kS/s).
        cutoff_hz: FIR lowpass cutoff frequency in Hz (default 37.5 kHz, yielding
            exact normalized cutoff fc/fs = 0.15 and noise gain ~ 0.5227).
        filter_taps: Length of linear-phase FIR filter (default 31).
        squelch_drop_db: Noise variance drop threshold for squelch in dB (default 15.0 dB).
        noise_floor: Nominal quiescent ADC/TIA noise floor standard deviation (default 1e-3).
        squelch_hysteresis_counts: Number of consecutive blocks required to enter or exit
            squelch state (default 2).
        stuck_threshold_lsb: Standard deviation threshold below which a sample block is
            flagged as stuck (default 0.5 LSB).
        sat_margin_lsb: Distance from ADC rails in LSBs flagging saturation (default 2.0 LSB).
        sat_fraction_limit: Maximum permissible fraction of saturated samples before
            asserting invalid flag (default 0.01 = 1%).
        code_min: Minimum ADC hardware code (default 0.0).
        code_max: Maximum ADC hardware code (default 65535.0 for 16-bit ADC).
        lsb: Least significant bit code weight (default 1.0).
        decim: Downsampling factor for filtered samples (default 1 = no decimation).
    """

    adc_rate_hz: float = 250000.0
    cutoff_hz: float = 37500.0
    filter_taps: int = 31
    squelch_drop_db: float = 15.0
    noise_floor: float = 1e-3
    squelch_hysteresis_counts: int = 2
    stuck_threshold_lsb: float = 0.5
    sat_margin_lsb: float = 2.0
    sat_fraction_limit: float = 0.01
    code_min: float = 0.0
    code_max: float = 65535.0
    lsb: float = 1.0
    decim: int = 1


class SignalConditioner:
    """Conditioner for raw monitor ADC sample blocks across (B, N) microrings.

    Performs sample imputation, linear-phase FIR filtering, second-difference noise floor
    estimation, and squelch variance hysteresis detection.
    """

    def __init__(self, n_rings: int = 8, config: ConditionerConfig | None = None) -> None:
        self.n_rings = int(n_rings)
        self.config = config if config is not None else ConditionerConfig()

        # Design 31-tap linear-phase Type I FIR lowpass filter
        self.h: np.ndarray = firwin(
            self.config.filter_taps,
            self.config.cutoff_hz,
            fs=self.config.adc_rate_hz,
        )
        self.taps: np.ndarray = self.h
        self.coeffs: np.ndarray = self.h
        self.fir_coeffs: np.ndarray = self.h
        self.noise_gain: float = float(np.sqrt(np.sum(self.h ** 2)))
        self.delay_s: float = (self.config.filter_taps - 1) / (2.0 * self.config.adc_rate_hz)

        self.batch_size: int = 1
        self._primed: bool = False
        self._primed_valid: bool = False

        # Internal state buffers
        self.zi: np.ndarray = np.zeros((self.batch_size, self.n_rings, self.config.filter_taps - 1))
        self.last_valid: np.ndarray = np.full((self.batch_size, self.n_rings), np.nan)
        self.noise: np.ndarray = np.full((self.batch_size, self.n_rings), self.config.noise_floor)
        self.noise_baseline: np.ndarray = np.full((self.batch_size, self.n_rings), self.config.noise_floor)
        self.squelch: np.ndarray = np.zeros((self.batch_size, self.n_rings), dtype=bool)
        self.squelch_active: np.ndarray = np.zeros((self.batch_size, self.n_rings), dtype=bool)
        self.sq_count: np.ndarray = np.zeros((self.batch_size, self.n_rings), dtype=int)
        self.unsq_count: np.ndarray = np.zeros((self.batch_size, self.n_rings), dtype=int)
        self.invalid: np.ndarray = np.zeros((self.batch_size, self.n_rings), dtype=bool)
        self.filtered_dc: np.ndarray = np.zeros((self.batch_size, self.n_rings))
        self.filtered_samples: np.ndarray | None = None

        self.reset(batch_size=1)

    def reset(self, batch_size: int = 1) -> None:
        """Resets conditioner filter states and squelch counters.

        Args:
            batch_size: Number of parallel simulation worlds B.
        """
        self.batch_size = int(batch_size)
        self.zi = np.zeros((self.batch_size, self.n_rings, self.config.filter_taps - 1), dtype=np.float64)
        self.last_valid = np.full((self.batch_size, self.n_rings), np.nan, dtype=np.float64)
        self.noise = np.full((self.batch_size, self.n_rings), self.config.noise_floor, dtype=np.float64)
        self.noise_baseline = np.full((self.batch_size, self.n_rings), self.config.noise_floor, dtype=np.float64)
        self.squelch = np.zeros((self.batch_size, self.n_rings), dtype=bool)
        self.squelch_active = np.zeros((self.batch_size, self.n_rings), dtype=bool)
        self.sq_count = np.zeros((self.batch_size, self.n_rings), dtype=int)
        self.unsq_count = np.zeros((self.batch_size, self.n_rings), dtype=int)
        self.invalid = np.zeros((self.batch_size, self.n_rings), dtype=bool)
        self.filtered_dc = np.zeros((self.batch_size, self.n_rings), dtype=np.float64)
        self.filtered_samples = None
        self._primed = False
        self._primed_valid = False

    def process_samples(
        self,
        samples: np.ndarray,
        count: np.ndarray | None = None,
        monitor_valid: np.ndarray | None = None,
        adc_overrange: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Processes ADC monitor samples and returns conditioned metrics.

        Args:
            samples: Raw ADC monitor samples of shape (B, N, M) or (N, M).
            count: Optional valid sample counts along axis M of shape (B, N) or (N,).
            monitor_valid: Optional boolean sensor validity mask from plant.
            adc_overrange: Optional boolean hardware ADC clipping mask from plant.

        Returns:
            filtered_dc: (B, N) block mean of conditioned monitor samples.
            squelch_active: (B, N) boolean mask flagging modulation squelch.
            noise_std: (B, N) smoothed wideband noise standard deviation estimate.
        """
        x = np.asarray(samples, dtype=np.float64)
        is_2d = False
        if x.ndim == 2:
            if x.shape[0] != self.n_rings and x.shape[1] == self.n_rings:
                x = x.T[None, ...]
            else:
                x = x[None, ...]
            is_2d = True
        elif x.ndim != 3:
            raise ValueError(f"Expected samples array of shape (B, N, M) or (N, M), got {samples.shape}")

        b_dim, n_dim, m_dim = x.shape
        if n_dim != self.n_rings:
            raise ValueError(f"Expected {self.n_rings} rings along axis 1, got {n_dim}")

        if b_dim != self.batch_size or not self._primed:
            self.reset(batch_size=b_dim)

        xf = x.copy()

        # Handle sample truncation via count with robust broadcasting for B >= 1
        if count is not None:
            c = np.asarray(count)
            if c.ndim == 0:
                c = np.broadcast_to(c, (b_dim, n_dim))
            elif c.ndim == 1:
                if c.shape[0] == n_dim:
                    c = np.broadcast_to(c, (b_dim, n_dim))
                elif c.shape[0] == b_dim and b_dim != n_dim:
                    c = np.broadcast_to(c[:, None], (b_dim, n_dim))
                else:
                    c = np.broadcast_to(c, (b_dim, n_dim))
            else:
                c = np.broadcast_to(c, (b_dim, n_dim))

            m_indices = np.arange(m_dim)
            mask_truncated = m_indices >= np.maximum(0, c)[..., None]
            xf[mask_truncated] = np.nan

        # Robust sample-and-hold imputation across axis M (handles NaN and Inf)
        missing = (~np.isfinite(xf)).any(axis=-1)
        for m in range(m_dim):
            mask_bad = ~np.isfinite(xf[..., m])
            if m == 0:
                prev = np.where(np.isfinite(self.last_valid), self.last_valid, 0.0)
            else:
                prev = xf[..., m - 1]
            xf[..., m] = np.where(mask_bad, prev, xf[..., m])

        self.last_valid = xf[..., -1].copy()

        # Anomaly detection: saturation and stuck bits
        cfg = self.config
        sat = ((x >= cfg.code_max - cfg.sat_margin_lsb * cfg.lsb) |
               (x <= cfg.code_min + cfg.lsb)).mean(axis=-1) > cfg.sat_fraction_limit
        stuck = np.std(xf, axis=-1) < (cfg.stuck_threshold_lsb * cfg.lsb)
        inv = missing | sat | stuck

        if monitor_valid is not None:
            mv = np.asarray(monitor_valid, dtype=bool)
            if mv.ndim == 0:
                mv = np.broadcast_to(mv, (b_dim, n_dim))
            elif mv.ndim == 1:
                if mv.shape[0] == n_dim:
                    mv = np.broadcast_to(mv, (b_dim, n_dim))
                elif mv.shape[0] == b_dim and b_dim != n_dim:
                    mv = np.broadcast_to(mv[:, None], (b_dim, n_dim))
                else:
                    mv = np.broadcast_to(mv, (b_dim, n_dim))
            else:
                mv = np.broadcast_to(mv, (b_dim, n_dim))
            inv = inv | (~mv)

        if adc_overrange is not None:
            ao = np.asarray(adc_overrange, dtype=bool)
            if ao.ndim == 0:
                ao = np.broadcast_to(ao, (b_dim, n_dim))
            elif ao.ndim == 1:
                if ao.shape[0] == n_dim:
                    ao = np.broadcast_to(ao, (b_dim, n_dim))
                elif ao.shape[0] == b_dim and b_dim != n_dim:
                    ao = np.broadcast_to(ao[:, None], (b_dim, n_dim))
                else:
                    ao = np.broadcast_to(ao, (b_dim, n_dim))
            else:
                ao = np.broadcast_to(ao, (b_dim, n_dim))
            inv = inv | ao

        self.invalid = inv

        # Digital filter priming
        if not self._primed or not self._primed_valid:
            valid_mask = np.isfinite(x)
            if valid_mask.any():
                x_init = xf[..., 0]
                zi_single = lfilter_zi(self.h, 1.0)
                self.zi = x_init[..., None] * zi_single
                self._primed = True
                self._primed_valid = True
            elif not self._primed:
                self.zi = np.zeros((b_dim, n_dim, len(self.h) - 1), dtype=np.float64)
                self._primed = True
                self._primed_valid = False

        # Linear-phase FIR filtering across time axis (axis=-1)
        y_filt, self.zi = lfilter(self.h, 1.0, xf, axis=-1, zi=self.zi)
        self.filtered_samples = y_filt[..., ::cfg.decim]
        self.filtered_dc = np.mean(y_filt, axis=-1)

        # High-frequency noise estimation via 2nd discrete difference
        if m_dim >= 3:
            d2 = np.diff(xf, n=2, axis=-1)
            nz = np.std(d2, axis=-1) / np.sqrt(6.0)
        else:
            nz = np.std(xf, axis=-1)

        self.noise = np.where(inv, self.noise, 0.7 * self.noise + 0.3 * nz)
        noise_inst = np.where(inv, self.noise, nz)

        # Baseline initialization on cold start
        self.noise_baseline = np.where(
            (self.noise_baseline <= cfg.noise_floor * 1.01) & (noise_inst > cfg.noise_floor),
            noise_inst,
            self.noise_baseline,
        )

        # Squelch drop detection
        ratio_enter = 10.0 ** (-cfg.squelch_drop_db / 20.0)
        ratio_exit = 10.0 ** (-(cfg.squelch_drop_db - 6.0) / 20.0)

        quiet = (noise_inst < ratio_enter * self.noise_baseline) | (noise_inst < 1.25 * cfg.noise_floor) | inv
        loud = (noise_inst > ratio_exit * self.noise_baseline) & (noise_inst > 2.0 * cfg.noise_floor) & (~inv)

        # Dynamic baseline update: freeze during squelch, quiet, or invalid steps
        valid_active = (~self.squelch) & (~quiet) & (~inv) & (noise_inst > 1.5 * cfg.noise_floor)
        self.noise_baseline = np.where(
            valid_active,
            0.95 * self.noise_baseline + 0.05 * noise_inst,
            self.noise_baseline,
        )
        self.noise_baseline = np.maximum(self.noise_baseline, cfg.noise_floor)

        # Consecutive counter update
        self.sq_count = np.where(quiet, self.sq_count + 1, 0)
        self.unsq_count = np.where(loud, self.unsq_count + 1, 0)

        # Schmitt-trigger 2-stage hysteresis update
        self.squelch = np.where(
            self.squelch,
            self.unsq_count < cfg.squelch_hysteresis_counts,
            self.sq_count >= cfg.squelch_hysteresis_counts,
        )

        squelch_active = self.squelch | inv
        self.squelch_active = squelch_active

        out_dc = self.filtered_dc[0] if is_2d else self.filtered_dc
        out_sq = squelch_active[0] if is_2d else squelch_active
        out_nz = self.noise[0] if is_2d else self.noise
        return out_dc, out_sq, out_nz

    # Canonical alias for compatibility with both process and process_samples callers
    process = process_samples


# B2Conditioner alias for spec alignment
B2Conditioner = SignalConditioner
