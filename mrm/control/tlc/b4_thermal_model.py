"""Block B4 Reduced Thermal Model for Lightmatter G26 8-Ring Modulator.

Implements the discrete-time state-space thermal dynamics (Ad, Bd, Cd, Dd)
at Ts = 100 us for the 8-ring microring modulator array.

Physical Dynamics and Modeling:
- Fast heater pole (tau_fast = 3.18 us): Settles completely within Ts = 100 us
  (exp(-100 / 3.18) ~ 0). Captured as an algebraic feedthrough matrix Dd.
- Slow heater pole (tau_slow = 13.01 us) & Package tail pole (tau_tail = 3.0 ms):
  Dynamic self-heating modes modeled via discrete state z_hs.
  At Ts = 100 us, the package tail pole gives a_hs = exp(-Ts / tau_tail) = 0.9672,
  while tau_slow gives a_slow = exp(-Ts / tau_slow) = 0.000459.
- Substrate thermal crosstalk (tau_x = 1.52 ms at 50 um pitch):
  Couples adjacent and distant rings via symmetric Toeplitz crosstalk matrix K_ij:
    K_ii = 0 (self-coupling captured in self-heating gains)
    K_{i, i+-1} = 0.030 (3.0% nearest-neighbor coupling)
    K_{i, i+-2} = 0.015 (1.5% next-nearest-neighbor coupling)
    K_{i, i+-k} = 0.015 (1.5% floor for distant rings)
  The discrete crosstalk pole is a_x = exp(-Ts / tau_x) = 0.9363.

State Space Formulation (16 states = 2 * N_RINGS):
  x[k] = [z_hs[k]; z_x[k]] in R^(2N)
  x[k+1] = Ad @ x[k] + Bd @ u[k]
  y[k]   = Cd @ x[k+1] + Dd @ u[k]   (or Cd @ x[k] + Dd @ u[k])

Batched Execution:
  Supports arbitrary batch size B >= 1 with inputs of shape (B, N) or (N,).
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np


# Default physical constants calibrated for Lightmatter G26 8-ring modulator
DEFAULT_N_RINGS: int = 8
DEFAULT_TS: float = 100e-6  # 100 us (10 kHz update rate)
DEFAULT_TAU_FAST: float = 3.18e-6  # 3.18 us (imec FE fit, CO22-TCPMT)
DEFAULT_TAU_SLOW: float = 13.01e-6  # 13.01 us (imec FE fit, CO22-TCPMT)
DEFAULT_TAU_TAIL: float = 3.0e-3  # 3.0 ms (package/substrate thermal tail)
DEFAULT_TAU_X: float = 1.52e-3  # 1.52 ms (crosstalk time constant at 50 um pitch)
DEFAULT_ETA_PM_PER_W: float = 404700.0  # ~404.7 pm/mW = 404,700 pm/W (P_fsr = 25.45 mW, FSR = 10.3 nm)
DEFAULT_SHARE_FAST: float = 0.4057  # 40.57% fast heater share
DEFAULT_SHARE_SLOW: float = 0.5443  # 54.43% slow heater share
DEFAULT_SHARE_TAIL: float = 0.05  # 5.0% package tail share
DEFAULT_K_NEAREST: float = 0.03  # 3.0% nearest-neighbor coupling
DEFAULT_K_NEXT: float = 0.015  # 1.5% next-nearest-neighbor coupling
DEFAULT_K_FLOOR: float = 0.015  # 1.5% distant coupling floor


class ReducedThermalModel:
    """Discrete-time reduced-order state-space thermal model for the 8-ring G26 array.

    Attributes:
        n_rings (int): Number of rings in modulator bank (N=8).
        n_states (int): Total state space dimension (2*N = 16).
        dt (float): Discrete sampling period Ts (default 100 us).
        ts (float): Alias for dt.
        Ad (np.ndarray): Discrete state transition matrix (2N, 2N).
        Bd (np.ndarray): Discrete input coupling matrix (2N, N).
        Cd (np.ndarray): Output observation matrix (N, 2N).
        Dd (np.ndarray): Algebraic feedthrough matrix (N, N).
        K (np.ndarray): Normalized inter-ring thermal crosstalk matrix (N, N).
        g_cross (np.ndarray): Alias for K.
        crosstalk_matrix (np.ndarray): Alias for K.
        gx (np.ndarray): Scaled crosstalk gain matrix in pm/W (N, N).
        gf (np.ndarray): Fast algebraic self-heating gain vector in pm/W (N,).
        ghs (np.ndarray): Slow dynamic self-heating gain vector in pm/W (N,).
        x (np.ndarray): Current internal state array of shape (B, 2N).
    """

    def __init__(
        self,
        n_rings: int = DEFAULT_N_RINGS,
        dt: float = DEFAULT_TS,
        ts: float | None = None,
        tau_fast: float = DEFAULT_TAU_FAST,
        tau_slow: float = DEFAULT_TAU_SLOW,
        tau_tail: float = DEFAULT_TAU_TAIL,
        tau_x: float = DEFAULT_TAU_X,
        eta_pm_per_w: float = DEFAULT_ETA_PM_PER_W,
        share_fast: float = DEFAULT_SHARE_FAST,
        share_slow: float = DEFAULT_SHARE_SLOW,
        share_tail: float = DEFAULT_SHARE_TAIL,
        k_nearest: float = DEFAULT_K_NEAREST,
        k_next: float = DEFAULT_K_NEXT,
        k_floor: float = DEFAULT_K_FLOOR,
        slow_pole_mode: str = "tail",
        preset_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the reduced thermal model with physical parameters.

        Args:
            n_rings: Number of microrings (default 8).
            dt: Sample period in seconds (default 100 us).
            ts: Optional alias for dt.
            tau_fast: Fast self-heating time constant in seconds (default 3.18 us).
            tau_slow: Slow heater time constant in seconds (default 13.01 us).
            tau_tail: Substrate/package tail time constant in seconds (default 3.0 ms).
            tau_x: Crosstalk time constant in seconds (default 1.52 ms).
            eta_pm_per_w: Total self-heating tuning efficiency in pm/W (default 404,700 pm/W).
            share_fast: Fractional share of fast pole (default 0.4057).
            share_slow: Fractional share of slow heater pole (default 0.5443).
            share_tail: Fractional share of package tail (default 0.05).
            k_nearest: Crosstalk fraction to nearest neighbor (default 0.03).
            k_next: Crosstalk fraction to second neighbor (default 0.015).
            k_floor: Crosstalk fraction floor beyond second neighbor (default 0.015).
            slow_pole_mode: 'tail' to use tau_tail as dynamic slow pole (poles > 0.9),
                or 'slow' to use tau_slow directly.
            preset_params: Optional configuration dictionary overriding defaults.
            **kwargs: Extra parameters (e.g. tau_hs for ECE496 compatibility).
        """
        if ts is not None:
            dt = ts
        if "tau_hs" in kwargs:
            tau_slow = float(kwargs["tau_hs"])
            slow_pole_mode = "custom"

        # Apply preset overrides if provided
        if preset_params is not None:
            tau_fast = preset_params.get("tau_fast_s", tau_fast)
            tau_slow = preset_params.get("tau_slow_s", tau_slow)
            tau_tail = preset_params.get("tail_tau_s", tau_tail)
            tau_x = preset_params.get("xt_tau_ref_s", tau_x)
            share_fast = preset_params.get("share_fast_frac", share_fast)
            share_tail = preset_params.get("tail_share_frac", share_tail)
            share_slow = 1.0 - share_fast - share_tail
            k_nearest = preset_params.get("xt_nearest_frac", k_nearest)
            k_next = preset_params.get("xt_next_frac", k_next)
            k_floor = preset_params.get("xt_floor_frac", k_floor)

        self.n_rings: int = int(n_rings)
        self.dt: float = float(dt)
        self.ts: float = self.dt
        self.n_states: int = 2 * self.n_rings

        self.tau_fast: float = float(tau_fast)
        self.tau_slow: float = float(tau_slow)
        self.tau_tail: float = float(tau_tail)
        self.tau_x: float = float(tau_x)
        self.eta_pm_per_w: float = float(eta_pm_per_w)
        self.share_fast: float = float(share_fast)
        self.share_slow: float = float(share_slow)
        self.share_tail: float = float(share_tail)

        # Configure dynamic slow pole vs algebraic feedthrough
        if slow_pole_mode == "tail":
            self.tau_hs: float = self.tau_tail
            # Fast + slow heater poles settle within Ts=100us -> feedthrough
            self.f_feedthrough: float = self.share_fast + self.share_slow
            self.f_slow_dynamic: float = self.share_tail
        elif slow_pole_mode == "custom":
            self.tau_hs = tau_slow
            self.f_feedthrough = self.share_fast
            self.f_slow_dynamic = self.share_slow + self.share_tail
        else:
            self.tau_hs = self.tau_slow
            self.f_feedthrough = self.share_fast
            self.f_slow_dynamic = self.share_slow + self.share_tail

        # Individual ring gains (pm/W)
        self.gf: np.ndarray = np.full(self.n_rings, self.f_feedthrough * self.eta_pm_per_w, dtype=np.float64)
        self.ghs: np.ndarray = np.full(self.n_rings, self.f_slow_dynamic * self.eta_pm_per_w, dtype=np.float64)

        # Crosstalk coupling matrix K (N, N)
        k_mat = np.zeros((self.n_rings, self.n_rings), dtype=np.float64)
        for i in range(self.n_rings):
            for j in range(self.n_rings):
                dist = abs(i - j)
                if dist == 0:
                    k_mat[i, j] = 0.0
                elif dist == 1:
                    k_mat[i, j] = k_nearest
                elif dist == 2:
                    k_mat[i, j] = k_next
                else:
                    k_mat[i, j] = k_floor

        self.K: np.ndarray = k_mat
        self.g_cross: np.ndarray = k_mat
        self.crosstalk_matrix: np.ndarray = k_mat

        # Physical crosstalk gain matrix in pm/W: Gx = K * eta
        self.gx: np.ndarray = k_mat * self.eta_pm_per_w
        self.Gx: np.ndarray = self.gx

        # Discrete poles: a = exp(-Ts / tau)
        self.a_hs: float = float(np.exp(-self.dt / self.tau_hs))
        self.a_x: float = float(np.exp(-self.dt / self.tau_x))

        # Build discrete state space matrices
        self._build_matrices()

        # State storage (B, 2N)
        self.x: np.ndarray = np.zeros((1, self.n_states), dtype=np.float64)

    def _build_matrices(self) -> None:
        """Construct discrete state-space matrices Ad, Bd, Cd, Dd."""
        N = self.n_rings

        # Ad: (2N, 2N) diagonal state transition
        # Upper block: slow self-heating poles a_hs * I_N
        # Lower block: thermal crosstalk poles a_x * I_N
        diag_poles = [self.a_hs] * N + [self.a_x] * N
        self.Ad: np.ndarray = np.diag(diag_poles).astype(np.float64)
        self.A: np.ndarray = self.Ad

        # Bd: (2N, N) input matrix
        # Upper block: (1 - a_hs) * diag(ghs)
        # Lower block: (1 - a_x) * gx
        self.Bd: np.ndarray = np.vstack([
            (1.0 - self.a_hs) * np.diag(self.ghs),
            (1.0 - self.a_x) * self.gx
        ]).astype(np.float64)
        self.B: np.ndarray = self.Bd

        # Cd: (N, 2N) state observation matrix
        # Observation is sum of slow self-heating and crosstalk states: [I_N, I_N]
        self.Cd: np.ndarray = np.hstack([np.eye(N), np.eye(N)]).astype(np.float64)
        self.C: np.ndarray = self.Cd

        # Dd: (N, N) direct feedthrough matrix
        # Fast self-heating algebraic feedthrough: diag(gf)
        self.Dd: np.ndarray = np.diag(self.gf).astype(np.float64)
        self.D: np.ndarray = self.Dd

    def set_gains(
        self,
        gf: np.ndarray | Sequence[float] | float,
        ghs: np.ndarray | Sequence[float] | float,
        gx: np.ndarray | Sequence[Sequence[float]],
    ) -> None:
        """Update self-heating and crosstalk gains dynamically.

        Args:
            gf: Fast algebraic self-heating gain (N,) or scalar in pm/W.
            ghs: Slow dynamic self-heating gain (N,) or scalar in pm/W.
            gx: Crosstalk gain matrix (N, N) in pm/W.
        """
        self.gf = np.broadcast_to(np.asarray(gf, dtype=np.float64), (self.n_rings,)).copy()
        self.ghs = np.broadcast_to(np.asarray(ghs, dtype=np.float64), (self.n_rings,)).copy()
        self.gx = np.asarray(gx, dtype=np.float64).copy()
        self.Gx = self.gx
        self._build_matrices()

    def reset(self, batch_size: int = 1) -> None:
        """Reset internal state to zero for a given batch size.

        Args:
            batch_size: Number of independent batch instances (B >= 1).
        """
        self.x = np.zeros((batch_size, self.n_states), dtype=np.float64)

    def settle(self, u: np.ndarray) -> np.ndarray:
        """Compute asymptotic steady-state thermal state for constant input u.

        Sets internal state self.x to the analytical steady state:
          z_hs_ss = ghs * u
          z_x_ss = gx @ u

        Args:
            u: Constant heater power vector in Watts, shape (B, N) or (N,).

        Returns:
            Steady state array matching input batch shape: (B, 2N) or (2N,).
        """
        u_arr = np.asarray(u, dtype=np.float64)
        is_1d = (u_arr.ndim == 1)
        if is_1d:
            u_arr = u_arr[None, :]

        # Steady state: (I - Ad)^(-1) @ Bd @ u
        z_hs_ss = u_arr * self.ghs[None, :]
        z_x_ss = u_arr @ self.gx.T

        self.x = np.hstack([z_hs_ss, z_x_ss])
        if is_1d:
            return self.x[0]
        return self.x

    def h(self, u_end: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
        """Compute instantaneous thermal output from heater command and state.

        y = Cd @ x + Dd @ u_end = gf * u_end + z_hs + z_x

        Args:
            u_end: Heater command at evaluation instant, shape (B, N) or (N,).
            x: State vector, shape (B, 2N) or (2N,). If None, uses internal self.x.

        Returns:
            Thermal detuning shift in picometers, shape (B, N) or (N,).
        """
        u_arr = np.asarray(u_end, dtype=np.float64)
        u_is_1d = (u_arr.ndim == 1)
        if u_is_1d:
            u_arr = u_arr[None, :]

        if x is None:
            x_arr = self.x
            x_is_1d = u_is_1d
        else:
            x_arr = np.asarray(x, dtype=np.float64)
            x_is_1d = (x_arr.ndim == 1)
            if x_is_1d:
                x_arr = x_arr[None, :]

        # Observation equation: y = x @ Cd.T + u @ Dd.T
        y = x_arr @ self.Cd.T + u_arr @ self.Dd.T

        if u_is_1d and x_is_1d:
            return y[0]
        return y

    def step(
        self,
        x: np.ndarray,
        u_heater_w: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Execute discrete-time state-space update step.

        Supports two execution modes:
        1. Explicit State-Space Mode (PROJECT.md / test_b4_ukf.py contract):
           step(x, u_heater_w) where x has shape (B, 2N) or (2N,):
             x[k+1] = x[k] @ Ad.T + u[k] @ Bd.T
             y[k]   = x[k+1] @ Cd.T + u[k] @ Dd.T
           Returns tuple: (x_next, delta_lambda_thermal_pm)

        2. Stateful Model Mode (ECE496 legacy contract):
           step(u_mean, u_end) where first arg has shape (B, N) or (N,):
           Updates internal state self.x and returns delta_lambda_thermal_pm.

        Args:
            x: State array of shape (B, 2N) or (2N,), OR u_mean of shape (B, N) or (N,).
            u_heater_w: Heater power in Watts, shape (B, N) or (N,).

        Returns:
            If explicit state provided: (x_next, y)
            If stateful (u_mean, u_end): y
        """
        x_arr = np.asarray(x, dtype=np.float64)

        # Mode 2: ECE496 stateful call where x is actually u_mean
        if x_arr.shape[-1] == self.n_rings:
            u_mean = x_arr
            u_end = x_arr if u_heater_w is None else np.asarray(u_heater_w, dtype=np.float64)

            mean_is_1d = (u_mean.ndim == 1)
            end_is_1d = (u_end.ndim == 1)
            if mean_is_1d:
                u_mean = u_mean[None, :]
            if end_is_1d:
                u_end = u_end[None, :]

            # Ensure internal state matches batch dimension
            B = u_mean.shape[0]
            if self.x.shape[0] != B:
                self.reset(batch_size=B)

            # Update internal state using u_mean
            self.x = self.x @ self.Ad.T + u_mean @ self.Bd.T
            # Output evaluated at u_end
            y = self.x @ self.Cd.T + u_end @ self.Dd.T

            if mean_is_1d and end_is_1d:
                return y[0]
            return y

        # Mode 1: Standard state-space step(x, u)
        if u_heater_w is None:
            raise ValueError("u_heater_w must be provided when stepping with explicit state x")

        u_arr = np.asarray(u_heater_w, dtype=np.float64)
        x_is_1d = (x_arr.ndim == 1)
        u_is_1d = (u_arr.ndim == 1)

        if x_is_1d:
            x_arr = x_arr[None, :]
        if u_is_1d:
            u_arr = u_arr[None, :]

        # Batched matrix multiplication:
        # x_next[b] = Ad @ x[b] + Bd @ u[b]
        # y[b]      = Cd @ x_next[b] + Dd @ u[b]
        x_next = x_arr @ self.Ad.T + u_arr @ self.Bd.T
        y = x_next @ self.Cd.T + u_arr @ self.Dd.T

        # Update internal state tracking
        self.x = x_next

        if x_is_1d:
            x_next = x_next[0]
        if x_is_1d and u_is_1d:
            y = y[0]

        return x_next, y

    def steady_state_gain_matrix(self) -> np.ndarray:
        """Return the (N, N) steady-state gain matrix in pm/W.

        Y_ss = G_ss @ u
        where G_ss = diag(gf + ghs) + gx = diag(g_self) + K * eta.
        """
        return np.diag(self.gf + self.ghs) + self.gx
