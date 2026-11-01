"""Block B4 Dual-Rate Unscented Kalman Filter (UKF) for TLC.

Ports and formulates state estimation for 8-channel Lightmatter G26 microring modulators,
fusing:
  1. Fast 100 us (10 kHz) DC monitor photodiode power levels (y0)
  2. Slow 4 ms (250 Hz) Hann-windowed harmonic ratios (zeta = y2 / y1)

State Vector Formulation:
  The state vector x in R^24 across N=8 rings is organized into three 8-state blocks:
    - delta (x[0:8]): Detuning of each ring from its laser line in picometers (pm).
    - r (x[8:16]): Drift rate of unmeasured ambient/die thermal drift in pm/s.
                   Random walk covariance shaped as common mode + gradient modes (linear,
                   quadratic) + local independent noise.
    - p (x[16:24]): Photodiode DC scale factor (laser power * coupling * PD gain * ADC scale)
                    in ADC full-scale units (~35000 - 45000 codes). Slow random walk.

Dual-Rate Fusion Mechanics:
  - Predict (Ts = 100 us):
      delta[k+1] = delta[k] + Ts * r[k] + dh_thermal(u_heater)
      r[k+1]     = r[k] + w_r
      p[k+1]     = p[k] + w_p (frozen if squelched)
  - Fast Update (Ts = 100 us):
      Measures y0_dc = p * L_avg(delta, hwhm, mod_shift)
      Tracks fast detuning dynamics and unmeasured thermal drift.
      Huber innovation soft gating (NIS > 25.0) inflates measurement noise to bound outliers.
      Cross-ring decoupling isolates sensor glitches on ring i from corrupting neighbor ring j.
  - Slow Update (Ts = 4 ms):
      Measures zeta = y2 / y1 (power-invariant harmonic ratio).
      Anchors absolute detuning delta without optical power bias (zeta ~ 0 at inflection).
      Squelch rejection drops invalid/collapsed modulation channels.
      DC photodiode scale p remains frozen during squelch.

Numerical Health:
  - Scaled Unscented Transform with alpha=0.5, beta=2.0, kappa=0.0 (c = alpha^2 * (nx + kappa)).
  - Joseph-form covariance update guarantees positive semi-definiteness under arbitrary gains.
  - Symmetrization and conditioning P = 0.5 * (P + P^T) + eps * I.
  - Full vectorization supporting arbitrary batch sizes B >= 1 and ring counts N=8.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["DualRateUKF", "B4UKF", "UKFConfig"]

# Default G26 Plant & Controller Constants
DEFAULT_N_RINGS: int = 8
DEFAULT_DT_FAST: float = 1e-4  # 100 us (10 kHz)
DEFAULT_DT_SLOW: float = 4e-3  # 4 ms (250 Hz)
DEFAULT_HWHM_PM: float = 170.13  # Loaded Q ~ 3850 at 1310 nm -> FWHM ~ 340.26 pm, HWHM ~ 170.13 pm
DEFAULT_MOD_SHIFT_PM: float = 50.0  # NRZ modulation resonance shift in pm
DEFAULT_DITHER_AMP_PM: float = 25.0  # Dither detuning peak amplitude in pm
DEFAULT_P_NOMINAL: float = 42116.0  # Nominal PD scale yielding y0 ~ 28000 codes at inflection


@dataclass
class UKFConfig:
    """Configuration parameters for Block B4 Dual-Rate UKF.

    Attributes:
        n_rings: Number of microring channels (default 8).
        dt_fast: Fast sampling period in seconds (default 100 us = 1e-4 s).
        dt_slow: Slow sampling period in seconds (default 4 ms = 4e-3 s).
        hwhm_pm: Optical resonance half-width at half-maximum in pm (default 170.13 pm).
        mod_shift_pm: NRZ modulation resonance shift in pm (default 50.0 pm).
        dither_amp_pm: Nominal dither detuning amplitude in pm (default 25.0 pm).
        p_nominal: Nominal photodiode DC scale factor (default 42116.0 codes).
        q_delta: Process noise standard deviation on detuning per step in pm (default 0.1 pm).
        q_rate_common: Common-mode drift rate noise standard deviation in pm/s (default 3400.0).
        q_rate_grad: Gradient-mode drift rate noise standard deviation in pm/s (default 1700.0).
        q_rate_local: Independent local drift rate noise standard deviation in pm/s (default 1400.0).
        q_p: Relative random walk noise standard deviation on DC scale per step (default 1e-5).
        r0_dc: Nominal measurement noise variance for fast DC photodiode levels (default 100.0^2).
        rz_zeta: Nominal measurement noise variance for slow harmonic ratios (default 0.05^2).
        gate: Huber innovation gating threshold on Normalized Innovation Squared (default 25.0).
        alpha: Scaled unscented transform primary scaling parameter (default 0.5).
        beta: Scaled unscented transform secondary parameter for Gaussian priors (default 2.0).
        kappa: Scaled unscented transform tertiary parameter (default 0.0).
    """

    n_rings: int = DEFAULT_N_RINGS
    dt_fast: float = DEFAULT_DT_FAST
    dt_slow: float = DEFAULT_DT_SLOW
    hwhm_pm: float = DEFAULT_HWHM_PM
    mod_shift_pm: float = DEFAULT_MOD_SHIFT_PM
    dither_amp_pm: DEFAULT_DITHER_AMP_PM = DEFAULT_DITHER_AMP_PM
    p_nominal: float = DEFAULT_P_NOMINAL
    q_delta: float = 0.1
    q_rate_common: float = 3400.0
    q_rate_grad: float = 1700.0
    q_rate_local: float = 1400.0
    q_p: float = 1e-5
    r0_dc: float = 10000.0  # (100.0)^2
    rz_zeta: float = 0.0025  # (0.05)^2
    gate: float = 25.0
    alpha: float = 0.5
    beta: float = 2.0
    kappa: float = 0.0


class DualRateUKF:
    """Dual-Rate 24-State Unscented Kalman Filter for 8-Ring G26 Modulators.

    State x = [delta (8), r (8), p (8)]:
      - delta: detuning from laser in pm
      - r: drift rate of unmeasured thermal drift in pm/s
      - p: photodiode DC scale factor (laser power * coupling * PD gain * ADC scale)
    """

    def __init__(
        self,
        n_rings: int = DEFAULT_N_RINGS,
        dt_fast: float = DEFAULT_DT_FAST,
        dt_slow: float = DEFAULT_DT_SLOW,
        ts: float | None = None,
        config: UKFConfig | None = None,
        thermal_model: Any = None,
        **kwargs: Any,
    ) -> None:
        self.n_rings = int(n_rings)
        self.dt_fast = float(ts if ts is not None else dt_fast)
        self.dt_slow = float(dt_slow)
        self.config = config if config is not None else UKFConfig(n_rings=self.n_rings, dt_fast=self.dt_fast, dt_slow=self.dt_slow)

        # Allow kwargs override of config values
        self.hwhm = float(kwargs.get("hwhm_pm", self.config.hwhm_pm))
        self.mod_shift = float(kwargs.get("mod_shift_pm", self.config.mod_shift_pm))
        self.dither_amp = float(kwargs.get("dither_amp_pm", self.config.dither_amp_pm))
        self.p_nominal = float(kwargs.get("p_nominal", self.config.p_nominal))

        self.q_delta = float(kwargs.get("q_delta", self.config.q_delta))
        self.q_rate_common = float(kwargs.get("q_rate_common", self.config.q_rate_common))
        self.q_rate_grad = float(kwargs.get("q_rate_grad", self.config.q_rate_grad))
        self.q_rate_local = float(kwargs.get("q_rate_local", self.config.q_rate_local))
        self.q_p = float(kwargs.get("q_p", self.config.q_p))
        self.ukf_gate = float(kwargs.get("gate", self.config.gate))
        self.r0_dc = float(kwargs.get("r0_dc", self.config.r0_dc))
        self.rz_zeta = float(kwargs.get("rz_zeta", self.config.rz_zeta))

        # 24 States across N rings: delta (N), r (N), p (N)
        self.nx = 3 * self.n_rings
        self.n = self.nx

        # Scaled Unscented Transform Parameters
        self.alpha = float(kwargs.get("alpha", self.config.alpha))
        self.beta = float(kwargs.get("beta", self.config.beta))
        self.kappa = float(kwargs.get("kappa", self.config.kappa))
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n
        self.c = self.n + self.lam

        # Sigma Point Weights
        self.wm = np.full(2 * self.n + 1, 1.0 / (2.0 * self.c), dtype=np.float64)
        self.wc = self.wm.copy()
        self.wm[0] = self.lam / self.c
        self.wc[0] = self.lam / self.c + (1.0 - self.alpha**2 + self.beta)

        # Base Drift Rate Covariance Matrix (Qr_base) across N rings
        N = self.n_rings
        e = np.ones(N, dtype=np.float64) / np.sqrt(N)
        g1 = np.linspace(-1.0, 1.0, N, dtype=np.float64)
        g1 /= np.linalg.norm(g1)
        g2 = g1**2
        g2 -= float(g2.mean())
        g2 /= np.linalg.norm(g2)

        self.Qr_base = (
            (self.q_rate_common**2) * np.outer(e, e)
            + (self.q_rate_grad**2) * (np.outer(g1, g1) + np.outer(g2, g2))
            + (self.q_rate_local**2) * np.eye(N, dtype=np.float64)
        )

        # Quadrature angles for lock-in ratio integration (24-point circle)
        self._NQ = 24
        self._theta = 2.0 * np.pi * (np.arange(self._NQ, dtype=np.float64) + 0.5) / self._NQ
        self._cos1 = np.cos(self._theta)
        self._cos2 = np.cos(2.0 * self._theta)

        # Attached or optional ReducedThermalModel
        self.thermal_model = thermal_model
        if self.thermal_model is None:
            try:
                from mrm.control.tlc.b4_thermal_model import ReducedThermalModel
                self.thermal_model = ReducedThermalModel(n_rings=self.n_rings, dt=self.dt_fast)
            except Exception:
                self.thermal_model = None

        # Internal history buffer for lock-in lag alignment
        self.history_length = int(round(self.dt_slow / self.dt_fast))
        self.hist: deque[np.ndarray] = deque(maxlen=self.history_length)

        # Initialize filter state
        self.batch_size = 1
        self.reset(batch_size=1)

    def reset(self, batch_size: int = 1) -> None:
        """Reset UKF states, covariance, and adaptive trackers for given batch size."""
        self.batch_size = int(batch_size)
        B, N = self.batch_size, self.n_rings

        # State vector x: (B, 24)
        self.x = np.zeros((B, self.nx), dtype=np.float64)
        # Initial detuning delta ~ +124 pm (nominal inflection setpoint)
        self.x[:, :N] = 124.0
        # Initial drift rate r = 0 pm/s
        self.x[:, N:2 * N] = 0.0
        # Initial photodiode scale p ~ 42116 codes
        self.x[:, 2 * N:] = self.p_nominal

        # Initial Covariance Matrix P: (B, 24, 24)
        # P_delta = (5.0 pm)^2 = 25.0 pm^2
        # P_r = (100.0 pm/s)^2 = 1e4 (pm/s)^2
        # P_p = (0.03 * p_nominal)^2 codes^2
        P0_diag = np.r_[
            np.full(N, 25.0, dtype=np.float64),
            np.full(N, 1e4, dtype=np.float64),
            ((0.03 * self.p_nominal) ** 2) * np.ones(N, dtype=np.float64),
        ]
        P0 = np.diag(P0_diag)
        self.P = np.repeat(P0[None, :, :], B, axis=0)

        # Adaptive rate noise multipliers and innovation monitors
        self.qm = np.ones((B, N), dtype=np.float64)
        self.gated = np.zeros((B, N), dtype=int)
        self.nis = np.zeros((B, N), dtype=np.float64)
        self.nis_dc = np.zeros((B, N), dtype=np.float64)
        self.squelch_mask = np.zeros((B, N), dtype=bool)
        self.latest_y0 = np.full((B, N), 28000.0, dtype=np.float64)
        self.step_idx = 0

        # Thermal state scratch
        self._thermal_x = np.zeros((B, 2 * N), dtype=np.float64)
        self._thermal_h_prev = np.zeros((B, N), dtype=np.float64)

    def predict(
        self,
        u_heater_power_w: np.ndarray | float | None = None,
        dh: np.ndarray | None = None,
    ) -> None:
        """Execute UKF time update step advancing state and covariance by dt_fast.

        Integrates thermal model detuning shift dh from known heater power u,
        or directly provided detuning shift increment dh, and random walk drift on
        detuning, drift rate, and DC scale.

        Args:
            u_heater_power_w: Applied heater electrical power in Watts.
                Shape (B, N), (N,), or scalar.
            dh: Direct thermal detuning shift increment in picometers.
                Shape (B, N), (N,), or None.
        """
        B, N = self.batch_size, self.n_rings

        if dh is not None:
            dh_val = np.asarray(dh, dtype=np.float64)
            if dh_val.ndim == 0:
                dh_val = np.full((B, N), float(dh_val))
            elif dh_val.ndim == 1:
                if dh_val.shape[0] == N:
                    dh_val = np.repeat(dh_val[None, :], B, axis=0)
                elif dh_val.shape[0] == B:
                    dh_val = np.repeat(dh_val[:, None], N, axis=1)
            elif dh_val.shape[0] != B:
                dh_val = np.broadcast_to(dh_val, (B, N))
        elif u_heater_power_w is not None:
            u = np.asarray(u_heater_power_w, dtype=np.float64)
            if u.ndim == 0:
                u = np.full((B, N), float(u))
            elif u.ndim == 1:
                if u.shape[0] == N:
                    u = np.repeat(u[None, :], B, axis=0)
                elif u.shape[0] == B:
                    u = np.repeat(u[:, None], N, axis=1)
            elif u.shape[0] != B:
                u = np.broadcast_to(u, (B, N))

            if not hasattr(self, "_thermal_h_prev") or self._thermal_h_prev.shape != (B, N):
                self._thermal_h_prev = np.zeros((B, N), dtype=np.float64)

            dh_val = np.zeros((B, N), dtype=np.float64)
            if self.thermal_model is not None and np.any(u != 0):
                try:
                    res = self.thermal_model.step(self._thermal_x, u)
                    if isinstance(res, tuple):
                        self._thermal_x, h_now = res[0], res[1]
                    else:
                        h_now = res
                    h_now = np.asarray(h_now, dtype=np.float64).reshape(B, N)
                    dh_val = h_now - self._thermal_h_prev
                    self._thermal_h_prev = h_now.copy()
                except Exception:
                    g_steady = 5.25e5
                    a_th = np.exp(-self.dt_fast / 13.0e-6)
                    x_next = a_th * self._thermal_x[:, :N] + (1.0 - a_th) * g_steady * u
                    dh_val = x_next - self._thermal_x[:, :N]
                    self._thermal_x[:, :N] = x_next
            elif self.thermal_model is not None and np.all(u == 0):
                if np.any(self._thermal_x != 0) or np.any(self._thermal_h_prev != 0):
                    res = self.thermal_model.step(self._thermal_x, u)
                    if isinstance(res, tuple):
                        self._thermal_x, h_now = res[0], res[1]
                    else:
                        h_now = res
                    h_now = np.asarray(h_now, dtype=np.float64).reshape(B, N)
                    dh_val = h_now - self._thermal_h_prev
                    self._thermal_h_prev = h_now.copy()
        else:
            dh_val = np.zeros((B, N), dtype=np.float64)

        # Linear state transition matrix F:
        # delta[k+1] = delta[k] + dt_fast * r[k] + dh
        # r[k+1]     = r[k]
        # p[k+1]     = p[k]
        F = np.eye(self.nx, dtype=np.float64)
        F[:N, N:2 * N] = self.dt_fast * np.eye(N, dtype=np.float64)

        for b in range(B):
            self.x[b] = F @ self.x[b]
            self.x[b, :N] += dh_val[b]

            # Process noise matrix Q
            Q = np.zeros((self.nx, self.nx), dtype=np.float64)
            Q[:N, :N] = np.eye(N, dtype=np.float64) * (self.q_delta**2)

            dq = np.sqrt(np.maximum(self.qm[b], 1e-6))
            Q[N:2 * N, N:2 * N] = self.Qr_base * np.outer(dq, dq)

            # Squelch handling: freeze DC photodiode scale process noise
            p_q = np.where(self.squelch_mask[b], 0.0, (self.q_p * self.x[b, 2 * N:]) ** 2)
            Q[2 * N:, 2 * N:] = np.diag(p_q)

            # Covariance time update with positive-definite conditioning
            self.P[b] = F @ self.P[b] @ F.T + Q
            self.P[b] = 0.5 * (self.P[b] + self.P[b].T) + 1e-10 * np.eye(self.nx, dtype=np.float64)

        # Record detuning state in sliding history
        self.hist.append(self.x[:, :N].copy())
        self.step_idx += 1

    def _sigma(self, b: int) -> np.ndarray:
        """Compute (2*nx + 1, nx) sigma points via scaled unscented transform for batch b."""
        n = self.nx
        Pb = 0.5 * (self.P[b] + self.P[b].T)
        try:
            S = np.linalg.cholesky(self.c * Pb)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(Pb)
            Pb = (v * np.maximum(w, 1e-9)) @ v.T
            S = np.linalg.cholesky(self.c * Pb + np.eye(n, dtype=np.float64) * 1e-9)
        return np.vstack([self.x[b], self.x[b] + S.T, self.x[b] - S.T])

    def _drop_avg(self, delta: np.ndarray, hwhm: float | np.ndarray, mod: np.ndarray) -> np.ndarray:
        """Data-averaged Lorentzian drop transmission under NRZ modulation."""
        h = np.asarray(hwhm, dtype=np.float64)
        l1 = 1.0 / (1.0 + ((delta + mod) / h) ** 2)
        l2 = 1.0 / (1.0 + ((delta - mod) / h) ** 2)
        return 0.5 * (l1 + l2)

    def _lockin_zeta(
        self, delta: np.ndarray, a: float | np.ndarray, hwhm: float | np.ndarray, mod: np.ndarray
    ) -> np.ndarray:
        """Compute exact power-invariant harmonic ratio zeta = y2 / y1 for detuning sigma points."""
        d = delta[:, :, None] + a * self._cos1[None, None, :]
        m = mod[None, :, None]
        h = np.asarray(hwhm, dtype=np.float64)[None, :, None] if np.ndim(hwhm) else hwhm
        l1 = 1.0 / (1.0 + ((d + m) / h) ** 2)
        l2 = 1.0 / (1.0 + ((d - m) / h) ** 2)
        lv = 0.5 * (l1 + l2)
        y1 = 2.0 * np.mean(lv * self._cos1, axis=-1)
        y2 = 2.0 * np.mean(lv * self._cos2, axis=-1)
        denom = np.where(np.abs(y1) > 1e-6, y1, 1.0)
        return np.where(np.abs(y1) > 1e-6, y2 / denom, 0.0)

    def update_fast(self, y0_dc: np.ndarray, squelch_mask: np.ndarray | None = None) -> None:
        """Fast 100 us measurement update using DC monitor photodiode levels.

        Fuses y0 = p * L_avg(delta) to track fast thermal detuning and unmeasured drift.
        Applies Huber innovation soft gating (NIS > 25.0) and cross-ring outlier decoupling.

        Args:
            y0_dc: Monitor photodiode DC level in ADC codes, shape (B, N) or (N,).
            squelch_mask: Optional boolean mask indicating squelch condition, shape (B, N) or (N,).
        """
        y0 = np.asarray(y0_dc, dtype=np.float64)
        if y0.ndim == 1:
            y0 = np.broadcast_to(y0, (self.batch_size, self.n_rings)).copy()
        B, N = self.batch_size, self.n_rings

        if squelch_mask is not None:
            sq = np.asarray(squelch_mask, dtype=bool)
            if sq.ndim == 1:
                sq = np.broadcast_to(sq, (self.batch_size, self.n_rings)).copy()
            self.squelch_mask = sq.copy()

        self.latest_y0 = y0.copy()

        for b in range(B):
            X = self._sigma(b)
            D = X[:, :N]
            PS = X[:, 2 * N:]
            mods = np.where(self.squelch_mask[b], 0.0, self.mod_shift)

            # Measurement sigma points: y0 = p * L_avg(delta)
            Z = PS * self._drop_avg(D, self.hwhm, mods)
            zm = self.wm @ Z
            dZ = Z - zm
            dX = X - self.x[b]

            R = np.full(N, self.r0_dc, dtype=np.float64)
            var = np.sum(self.wc[:, None] * (dZ**2), axis=0) + R
            innov = y0[b] - zm
            nis = (innov**2) / np.maximum(var, 1e-12)

            # Huber soft gating
            big = nis > self.ukf_gate
            self.gated[b] = np.where(big, self.gated[b] + 1, 0)
            self.nis_dc[b] = 0.7 * self.nis_dc[b] + 0.3 * np.minimum(nis, 4.0 * self.ukf_gate)

            # Adaptive drift-rate multiplier based on persistent large DC innovations
            target = np.clip(self.nis_dc[b] / 3.0, 1.0, 30.0)
            self.qm[b] = np.where(target > self.qm[b], target, 0.98 * self.qm[b] + 0.02 * target)

            # Huber noise inflation
            R_eff = R * np.maximum(1.0, nis / self.ukf_gate)
            Pzz = (self.wc[:, None] * dZ).T @ dZ + np.diag(R_eff)
            Pxz = (self.wc[:, None] * dX).T @ dZ

            K = np.linalg.solve(Pzz, Pxz.T).T  # Shape: (24, 8)

            # Cross-ring outlier decoupling: outlying ring i may only correct its own states
            state_ring = np.tile(np.arange(N), 3)
            for j in np.where(big)[0]:
                K[state_ring != j, j] = 0.0

            # Squelch handling: freeze DC photodiode scale p
            for i in range(N):
                if self.squelch_mask[b, i]:
                    K[2 * N + i, :] = 0.0

            # State and Joseph-form covariance update
            self.x[b] = self.x[b] + K @ innov
            self.P[b] = self.P[b] - K @ Pxz.T - Pxz @ K.T + K @ Pzz @ K.T
            self.P[b] = 0.5 * (self.P[b] + self.P[b].T) + 1e-10 * np.eye(self.nx, dtype=np.float64)

    def update_slow(self, zeta: np.ndarray, squelch_mask: np.ndarray | None = None) -> None:
        """Slow 4 ms measurement update using lock-in harmonic ratio zeta = y2 / y1.

        Anchors absolute detuning delta without optical power or photodiode gain bias.
        Rejects squelched channels and isolates cross-ring outliers.

        Args:
            zeta: Harmonic ratio y2 / y1, shape (B, N) or (N,).
            squelch_mask: Optional boolean mask indicating squelch condition, shape (B, N) or (N,).
        """
        z = np.asarray(zeta, dtype=np.float64)
        if z.ndim == 1:
            z = np.broadcast_to(z, (self.batch_size, self.n_rings)).copy()
        B, N = self.batch_size, self.n_rings

        if squelch_mask is not None:
            sq = np.asarray(squelch_mask, dtype=bool)
            if sq.ndim == 1:
                sq = np.broadcast_to(sq, (self.batch_size, self.n_rings)).copy()
            self.squelch_mask = sq.copy()

        for b in range(B):
            X = self._sigma(b)
            D = X[:, :N]
            mods = np.where(self.squelch_mask[b], 0.0, self.mod_shift)

            # Measurement sigma points for harmonic ratio zeta
            Z = self._lockin_zeta(D, self.dither_amp, self.hwhm, mods)
            zm = self.wm @ Z
            dZ = Z - zm
            dX = X - self.x[b]

            R = np.full(N, self.rz_zeta, dtype=np.float64)
            var = np.sum(self.wc[:, None] * (dZ**2), axis=0) + R
            innov = z[b] - zm
            nis = (innov**2) / np.maximum(var, 1e-12)

            self.nis[b] = 0.9 * self.nis[b] + 0.1 * np.minimum(nis, 50.0)
            big = nis > self.ukf_gate

            # Huber noise inflation
            R_eff = R * np.maximum(1.0, nis / self.ukf_gate)
            Pzz = (self.wc[:, None] * dZ).T @ dZ + np.diag(R_eff)
            Pxz = (self.wc[:, None] * dX).T @ dZ

            K = np.linalg.solve(Pzz, Pxz.T).T

            # Cross-ring outlier decoupling
            state_ring = np.tile(np.arange(N), 3)
            for j in np.where(big)[0]:
                K[state_ring != j, j] = 0.0

            # Squelch rejection: drop collapsed/invalid channels completely
            for j in range(N):
                if self.squelch_mask[b, j]:
                    K[:, j] = 0.0

            # Squelch handling: freeze DC scale p
            for i in range(N):
                if self.squelch_mask[b, i]:
                    K[2 * N + i, :] = 0.0

            # State and Joseph-form covariance update
            self.x[b] = self.x[b] + K @ innov
            self.P[b] = self.P[b] - K @ Pxz.T - Pxz @ K.T + K @ Pzz @ K.T
            self.P[b] = 0.5 * (self.P[b] + self.P[b].T) + 1e-10 * np.eye(self.nx, dtype=np.float64)

    def update(
        self,
        zeta: np.ndarray,
        zeta_valid: np.ndarray | None = None,
        y0_block: np.ndarray | None = None,
        dc_valid: np.ndarray | None = None,
        squelch: np.ndarray | None = None,
    ) -> None:
        """Unified dual-rate update interface compatible with ECE496 B4 controller dispatch."""
        if squelch is not None:
            sq = np.asarray(squelch, dtype=bool)
            if sq.ndim == 1:
                sq = np.broadcast_to(sq, (self.batch_size, self.n_rings)).copy()
            self.squelch_mask = sq.copy()
        if y0_block is not None and (dc_valid is None or np.any(dc_valid)):
            self.update_fast(y0_block, self.squelch_mask)
        if zeta_valid is not None and np.any(zeta_valid):
            self.update_slow(zeta, self.squelch_mask)

    def get_estimates(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns optimal detuning estimates and 1-sigma uncertainties.

        Returns:
            delta_hat: Estimated detuning in pm, shape (B, N).
            sigma_delta: 1-sigma detuning estimation uncertainty in pm, shape (B, N).
        """
        B, N = self.batch_size, self.n_rings
        delta_hat = self.x[:, :N].copy()
        sigma_delta = np.zeros((B, N), dtype=np.float64)
        for b in range(B):
            sigma_delta[b] = np.sqrt(np.maximum(np.diag(self.P[b])[:N], 1e-12))
        return delta_hat, sigma_delta

    # --- Property Accessors ---
    @property
    def delta(self) -> np.ndarray:
        """Estimated detuning in pm, shape (B, N)."""
        return self.x[:, :self.n_rings]

    @property
    def rate(self) -> np.ndarray:
        """Estimated drift rate in pm/s, shape (B, N)."""
        return self.x[:, self.n_rings:2 * self.n_rings]

    @property
    def pscale(self) -> np.ndarray:
        """Estimated photodiode DC scale factor, shape (B, N)."""
        return self.x[:, 2 * self.n_rings:]

    @property
    def sigma(self) -> np.ndarray:
        """1-sigma detuning estimation uncertainty in pm, shape (B, N)."""
        _, sig = self.get_estimates()
        return sig

    @property
    def cov(self) -> np.ndarray:
        """State covariance matrix, shape (B, 24, 24)."""
        return self.P

    @property
    def q_mult(self) -> float:
        """Maximum adaptive rate noise multiplier across all rings."""
        return float(np.max(self.qm))


# Top-level class alias
B4UKF = DualRateUKF
