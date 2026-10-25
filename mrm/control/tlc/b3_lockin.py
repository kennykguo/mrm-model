"""Block B3 Lock-In Demodulator for Thermal Locking Controller (TLC).

Implements sliding 4 ms Hann-windowed dual-harmonic I/Q demodulation extracting
fundamental amplitude (y1), second harmonic (y2), DC level (y0), and the
power-invariant harmonic ratio zeta = y2 / y1 for 8-channel G26 MRM transmitters.

Remediated Architecture:
- Complete removal of cross-channel borrowing (no reference amplitude copying)
  and inter-channel DC baseline subtraction. Every channel is demodulated 100%
  independently.
- Native support for intra-step / multi-sample ADC blocks (B, N, M) at 250 kS/s,
  completely resolving Nyquist and DC-aliasing degeneracies across all 8 channels.
- Configurable multi-rate scalar stream processing (B, N) for arbitrary sample rates.
- Vectorized execution supporting arbitrary batch sizes B and channel counts N.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = ["LockInConfig", "LockInDemodulator", "B3LockIn"]

DEFAULT_FREQUENCIES_HZ: list[float] = [
    1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0
]


@dataclass
class LockInConfig:
    """Configuration parameters for Block B3 Lock-In Demodulator.

    Attributes:
        sample_rate_hz: Base controller step rate / sample rate in Hz (default 10 kHz, Ts = 100 us).
        adc_rate_hz: Plant ADC / intra-step sampling rate in Hz (default 250 kS/s, Ts = 4 us).
        window_length_s: Sliding Hann window duration in seconds (default 4 ms = 0.004 s).
        harmonic_order: Number of harmonics to extract (default 2: fundamental and 2nd harmonic).
    """

    sample_rate_hz: float = 10000.0
    adc_rate_hz: float = 250000.0
    window_length_s: float = 0.004
    harmonic_order: int = 2


class LockInDemodulator:
    """Sliding Hann-windowed dual-harmonic I/Q demodulator.

    Extracts fundamental (y1) and second harmonic (y2) dither amplitudes and
    computes the power-invariant harmonic ratio zeta = y2 / y1 across (B, N) channels
    without any inter-channel borrowing or heuristics.
    """

    def __init__(
        self,
        n_rings: int = 8,
        dither_frequencies: list[float] | tuple[float, ...] | None = None,
        config: LockInConfig | None = None,
    ) -> None:
        self.n_rings = int(n_rings)
        self.config = config if config is not None else LockInConfig()

        if dither_frequencies is None:
            self.freqs = np.array(DEFAULT_FREQUENCIES_HZ[:self.n_rings], dtype=np.float64)
        else:
            self.freqs = np.array(dither_frequencies[:self.n_rings], dtype=np.float64)

        self.sample_rate_hz = float(self.config.sample_rate_hz)
        self.adc_rate_hz = float(getattr(self.config, "adc_rate_hz", 250000.0))
        self.window_length_s = float(self.config.window_length_s)

        # Base scalar window parameters
        self.ts = 1.0 / self.sample_rate_hz
        self.n_window = int(round(self.window_length_s * self.sample_rate_hz))

        # Effective sampling rate and sub-sample multiplier (initialized for scalar M=1)
        self.m_subsamples = 1
        self.effective_fs = self.sample_rate_hz
        self.effective_ts = self.ts
        self.effective_n_window = self.n_window

        # Precompute window and basis tables for current effective window length
        self._init_window_and_bases(self.effective_n_window, self.effective_fs)

        # Internal state initialized for batch_size = 1
        self.batch_size = 1
        self.reset(batch_size=1)

    def _init_window_and_bases(self, n_win: int, fs: float) -> None:
        """Initialize normalized Hann window and basis functions for window length n_win."""
        self.effective_n_window = int(n_win)
        self.effective_fs = float(fs)
        self.effective_ts = 1.0 / self.effective_fs

        # Periodic (DFT-even) Hann window normalized to sum = 1
        # w_k = 0.5 * [1 - cos(2*pi*k / N_w)]
        w_raw = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(self.effective_n_window) / self.effective_n_window))
        w_norm = w_raw / np.sum(w_raw)
        self.w = w_norm[:, None, None]  # Shape: (N_w, 1, 1)

        # Precompute time offsets relative to window start: t_win in [0, window_length_s)
        t_win = (np.arange(self.effective_n_window) * self.effective_ts)[:, None]  # Shape: (N_w, 1)
        omega = 2.0 * np.pi * self.freqs[None, :]  # Shape: (1, N)

        # Static quadrature basis vectors
        self.c1 = np.cos(omega * t_win)        # Shape: (N_w, N)
        self.s1 = np.sin(omega * t_win)
        self.c2 = np.cos(2.0 * omega * t_win)
        self.s2 = np.sin(2.0 * omega * t_win)

        w_2d = self.w[:, :, 0]  # Shape: (N_w, 1)
        self.norm_c1 = np.sum(2.0 * w_2d * (self.c1 ** 2), axis=0)  # Shape: (N,)
        self.norm_s1 = np.sum(2.0 * w_2d * (self.s1 ** 2), axis=0)
        self.norm_c2 = np.sum(2.0 * w_2d * (self.c2 ** 2), axis=0)
        self.norm_s2 = np.sum(2.0 * w_2d * (self.s2 ** 2), axis=0)

    def reset(self, batch_size: int = 1) -> None:
        """Reset internal sliding buffer and accumulators."""
        self.batch_size = int(batch_size)
        self.buffer = np.zeros(
            (self.effective_n_window, self.batch_size, self.n_rings), dtype=np.float64
        )
        self.ptr = 0
        self.total_samples_pushed = 0
        self.step_count = 0
        self._ready = False

        self._y0 = np.zeros((self.batch_size, self.n_rings), dtype=np.float64)
        self._y1 = np.zeros((self.batch_size, self.n_rings), dtype=np.float64)
        self._y2 = np.zeros((self.batch_size, self.n_rings), dtype=np.float64)
        self._zeta = np.zeros((self.batch_size, self.n_rings), dtype=np.float64)

        self._I1 = np.zeros((self.batch_size, self.n_rings), dtype=np.float64)
        self._Q1 = np.zeros((self.batch_size, self.n_rings), dtype=np.float64)
        self._I2 = np.zeros((self.batch_size, self.n_rings), dtype=np.float64)
        self._Q2 = np.zeros((self.batch_size, self.n_rings), dtype=np.float64)

    def step(
        self,
        monitor_dc: np.ndarray,
        dither_detuning: np.ndarray | None = None,
    ) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        """Process incoming monitor sample block and update sliding demodulator outputs.

        Supports:
        - Intra-step ADC sample blocks of shape (B, N, M) or (N, M) at high sampling rate.
        - Scalar per-step samples of shape (B, N) or (N,).

        Args:
            monitor_dc: Conditioned monitor photodiode power / ADC codes.
                Shape (B, N, M), (N, M), (B, N), or (N,).
            dither_detuning: Optional applied dither detuning in pm.

        Returns:
            ready: bool indicating whether sliding 4 ms window has filled.
            y1: Fundamental harmonic amplitude, shape (B, N).
            y2: Second harmonic amplitude, shape (B, N).
            zeta: Harmonic ratio y2 / y1, shape (B, N).
        """
        arr = np.asarray(monitor_dc, dtype=np.float64)

        # Normalize dimensions to (M, B, N) where M is samples per step
        if arr.ndim == 1:
            # Shape (N,) -> B=1, M=1
            B, N, M = 1, arr.shape[0], 1
            x_step = arr[None, :, None]  # (1, N, 1)
        elif arr.ndim == 2:
            # Shape (B, N) or (N, M)
            if arr.shape[0] == self.n_rings and arr.shape[1] != self.n_rings:
                # Single batch of intra-step samples: (N, M) -> (1, N, M)
                B, N, M = 1, arr.shape[0], arr.shape[1]
                x_step = arr[None, :, :]  # (1, N, M)
            elif arr.shape[1] == self.n_rings:
                # Standard batch of scalar samples: (B, N) -> (B, N, 1)
                B, N, M = arr.shape[0], arr.shape[1], 1
                x_step = arr[:, :, None]  # (B, N, 1)
            else:
                B, N, M = arr.shape[0], arr.shape[1], 1
                x_step = arr[:, :, None]
        elif arr.ndim == 3:
            # Shape (B, N, M)
            B, N, M = arr.shape[0], arr.shape[1], arr.shape[2]
            x_step = arr
        else:
            raise ValueError(f"Expected array of ndim 1, 2, or 3, got shape {arr.shape}")

        # Adapt internal buffer if batch size, ring count, or intra-step sample count changed
        if B != self.batch_size or N != self.n_rings or M != self.m_subsamples:
            self.batch_size = B
            self.n_rings = N
            self.m_subsamples = M
            if M > 1:
                eff_fs = self.sample_rate_hz * M
            else:
                eff_fs = self.sample_rate_hz
            n_win = int(round(self.window_length_s * eff_fs))
            self._init_window_and_bases(n_win, eff_fs)
            self.reset(batch_size=B)

        # Transpose x_step from (B, N, M) to (M, B, N) for sequential buffer insertion
        x_seq = np.transpose(x_step, (2, 0, 1))  # Shape: (M, B, N)

        # Append M samples sequentially into circular buffer
        for m in range(M):
            self.buffer[self.ptr] = x_seq[m]
            self.ptr = (self.ptr + 1) % self.effective_n_window
            self.total_samples_pushed += 1

        self.step_count += 1
        self._ready = bool(self.total_samples_pushed >= self.effective_n_window)

        if not self._ready:
            zeros = np.zeros((B, N), dtype=np.float64)
            return False, zeros, zeros, zeros

        # Extract ordered window (oldest at index 0, newest at index N_w - 1)
        ordered_idx = (np.arange(self.effective_n_window) + self.ptr) % self.effective_n_window
        ordered = self.buffer[ordered_idx]  # Shape: (N_w, B, N)

        # Apply normalized Hann window
        xw = ordered * self.w               # Shape: (N_w, B, N)
        self._y0 = np.sum(xw, axis=0)       # Shape: (B, N)

        # Quadrature basis vectors: (N_w, 1, N)
        c1 = self.c1[:, None, :]
        s1 = self.s1[:, None, :]
        c2 = self.c2[:, None, :]
        s2 = self.s2[:, None, :]

        # Normalized quadrature correlations
        I1_raw = np.sum(2.0 * xw * c1, axis=0)  # Shape: (B, N)
        Q1_raw = np.sum(2.0 * xw * s1, axis=0)
        I2_raw = np.sum(2.0 * xw * c2, axis=0)
        Q2_raw = np.sum(2.0 * xw * s2, axis=0)

        norm_c1 = self.norm_c1[None, :]  # Shape: (1, N)
        norm_s1 = self.norm_s1[None, :]
        norm_c2 = self.norm_c2[None, :]
        norm_s2 = self.norm_s2[None, :]

        self._I1 = np.where(norm_c1 > 1e-6, I1_raw / norm_c1, 0.0)
        self._Q1 = np.where(norm_s1 > 1e-6, Q1_raw / norm_s1, 0.0)
        self._I2 = np.where(norm_c2 > 1e-6, I2_raw / norm_c2, 0.0)
        self._Q2 = np.where(norm_s2 > 1e-6, Q2_raw / norm_s2, 0.0)

        # Amplitude calculation (100% independent per channel)
        self._y1 = np.sqrt(self._I1 ** 2 + self._Q1 ** 2)
        self._y2 = np.sqrt(self._I2 ** 2 + self._Q2 ** 2)

        # Safe harmonic ratio calculation guarded against division by zero
        denom = np.where(self._y1 > 1e-6, self._y1, 1.0)
        self._zeta = np.where(self._y1 > 1e-6, self._y2 / denom, 0.0)

        return True, self._y1, self._y2, self._zeta

    # --- Property Accessors ---
    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def y0(self) -> np.ndarray:
        return self._y0

    @property
    def y1(self) -> np.ndarray:
        return self._y1

    @property
    def y2(self) -> np.ndarray:
        return self._y2

    @property
    def zeta(self) -> np.ndarray:
        return self._zeta

    @property
    def I1(self) -> np.ndarray:
        return self._I1

    @property
    def Q1(self) -> np.ndarray:
        return self._Q1

    @property
    def I2(self) -> np.ndarray:
        return self._I2

    @property
    def Q2(self) -> np.ndarray:
        return self._Q2

    # --- ECE496 Compatibility Interface ---
    def push(self, y: np.ndarray, t: float | np.ndarray, invalid: np.ndarray | None = None) -> None:
        """Push sample block into buffer (ECE496 compatibility wrapper)."""
        self.step(y)

    def demod(
        self,
        h_complex: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """Demodulate window (ECE496 compatibility wrapper)."""
        return self._y0, self._y1, self._y2, self._zeta, self._ready


# Top-level alias
B3LockIn = LockInDemodulator
