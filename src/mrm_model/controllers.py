"""Controllers share measured ADC data and nominal calibration, never plant truth.

The plant's internal state is intentionally absent from Observation. Nominal calibration
is permissible controller knowledge, just as a measured lookup table is on real hardware.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class Observation:
    time_s: float
    sample_time_s: float
    adc_codes: np.ndarray
    valid: bool = True
    clipped: np.ndarray | None = None


@dataclass(frozen=True)
class ControllerContext:
    sample_period_s: float
    initial_codes: np.ndarray
    max_code: int
    target_fraction: np.ndarray
    # Measured monitor fraction sensitivity / DAC code at the chosen flank.
    jacobian_per_code: np.ndarray
    adc_max_code: int
    # Local identified plant A,B,C: state deviations driven by DAC code deviations.
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray


@dataclass(frozen=True)
class ControllerParams:
    kind: str = "pi"
    kp: float = 0.35
    ki_per_s: float = 2000.0
    dither_amplitude_codes: float = 10.0
    dither_frequency_hz: float = 2000.0
    dither_gain_codes_per_s: float = 1e5
    dither_target_gradient_per_code: float = 0.0
    dither_objective: str = "flank"
    dither_phase_rad: float = 0.0
    mpc_horizon: int = 8
    mpc_move_penalty: float = 0.1
    acquisition: bool = False
    scan_min_code: int = 0
    scan_max_code: int = 65535
    scan_points: int = 100
    scan_dwell_samples: int = 40

    def __post_init__(self):
        if self.kind not in {"open_loop", "pi", "dither", "mpc"}:
            raise ValueError(f"Unknown controller: {self.kind}")
        if self.dither_objective not in {"flank", "peak", "slope"}:
            raise ValueError("dither_objective must be flank, peak or slope")
        for name in (
            "kp",
            "ki_per_s",
            "dither_amplitude_codes",
            "dither_gain_codes_per_s",
            "mpc_move_penalty",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.dither_frequency_hz) or self.dither_frequency_hz <= 0:
            raise ValueError("dither_frequency_hz must be positive")
        if not np.isfinite(self.dither_target_gradient_per_code) or not np.isfinite(self.dither_phase_rad):
            raise ValueError("dither target must be finite")
        if not 1 <= self.mpc_horizon <= 100:
            raise ValueError("MPC horizon must lie in [1,100]")
        if self.scan_points < 2 or self.scan_dwell_samples < 1 or self.scan_min_code > self.scan_max_code:
            raise ValueError("invalid acquisition scan")


class Controller(Protocol):
    def step(self, observation: Observation) -> np.ndarray: ...


class OpenLoop:
    def __init__(self, params, context):
        self.codes = context.initial_codes.copy()

    def step(self, observation):
        return self.codes.copy()


class SideOfFringePI:
    """Local PI in sensor space with conditional-integration anti-windup.

    The flank's measured slope sets feedback sign. This controller cannot acquire a
    resonance from an arbitrary initial condition or disambiguate equal readings on
    opposite flanks. Acquisition is a separate optional measured scan.
    """

    def __init__(self, params, context):
        self.p, self.ctx = params, context
        slope = np.diag(context.jacobian_per_code)
        if np.any(np.abs(slope) < 1e-12):
            raise ValueError("PI needs an observable nonzero slope; choose a flank target")
        self.inverse_slope = 1 / slope
        self.integral = np.zeros_like(slope)
        self.last_codes = context.initial_codes.copy()
        self.last_timestamp = -np.inf

    def step(self, observation):
        if not observation.valid or observation.sample_time_s <= self.last_timestamp:
            return self.last_codes.copy()
        dt = (
            self.ctx.sample_period_s
            if not np.isfinite(self.last_timestamp)
            else (observation.sample_time_s - self.last_timestamp)
        )
        self.last_timestamp = observation.sample_time_s
        error = self.ctx.target_fraction - observation.adc_codes / self.ctx.adc_max_code
        correction = error * self.inverse_slope
        candidate = self.integral + self.p.ki_per_s * dt * correction
        raw = self.ctx.initial_codes + self.p.kp * correction + candidate
        # Integrate only if unsaturated, or if error drives back toward the valid range.
        allow = (
            ((raw >= 0) & (raw <= self.ctx.max_code))
            | ((raw < 0) & (correction > 0))
            | ((raw > self.ctx.max_code) & (correction < 0))
        )
        self.integral = np.where(allow, candidate, self.integral)
        raw = self.ctx.initial_codes + self.p.kp * correction + self.integral
        self.last_codes = np.clip(np.rint(raw), 0, self.ctx.max_code)
        return self.last_codes.copy()


class DitherLock:
    """Lock-in adaptive flank tracker, or optional monitor peak/slope objective.

    This optimizes a monitor objective, not BER or OMA. Quadrature phase needs tuning
    when the plant/filter delay is appreciable. Distinct frequencies reduce lane
    interference; no demodulated data reaches other controllers.
    """

    def __init__(self, params, context):
        self.p, self.ctx = params, context
        self.dc = context.initial_codes.copy()
        self.integral = np.zeros_like(self.dc)
        self.mean = context.target_fraction.copy()
        self.gradient = np.zeros_like(self.dc)
        self.last_timestamp = -np.inf
        self.freq = params.dither_frequency_hz * (1 + 0.17 * np.arange(len(self.dc)))
        if np.max(self.freq) >= 0.25 / context.sample_period_s:
            raise ValueError("dither frequency must be below one quarter of controller sample rate")

    def step(self, observation):
        if observation.valid and observation.sample_time_s > self.last_timestamp:
            dt = (
                self.ctx.sample_period_s
                if not np.isfinite(self.last_timestamp)
                else (observation.sample_time_s - self.last_timestamp)
            )
            self.last_timestamp = observation.sample_time_s
            y = observation.adc_codes / self.ctx.adc_max_code
            phase = np.sin(2 * np.pi * self.freq * observation.sample_time_s + self.p.dither_phase_rad)
            weight = -np.expm1(-dt * 2 * np.pi * self.freq / 8)
            self.mean += weight * (y - self.mean)
            scale = max(self.p.dither_amplitude_codes, 1)
            self.gradient += weight * (2 * (y - self.mean) * phase / scale - self.gradient)
            # Scale to a dimensionless gradient based on local sensitivity to keep gain interpretable.
            scale_slope = np.maximum(np.abs(np.diag(self.ctx.jacobian_per_code)), 1e-9)
            if self.p.dither_objective == "flank":
                nominal = np.diag(self.ctx.jacobian_per_code)
                # Keep the configured flank until a measured scan establishes another branch.
                # Clamp a weak/noisy gradient to avoid division near an unobservable peak.
                effective = np.sign(nominal) * np.clip(abs(self.gradient), 0.7 * scale_slope, 2 * scale_slope)
                # Remove the observed first-harmonic probe before tracking the DC setpoint.
                dc_measurement = y - effective * self.p.dither_amplitude_codes * phase
                correction = (self.ctx.target_fraction - dc_measurement) / effective
                candidate = self.integral + dt * self.p.ki_per_s * correction
                raw = self.ctx.initial_codes + self.p.kp * correction + candidate
                allow = (
                    ((raw >= 0) & (raw <= self.ctx.max_code))
                    | ((raw < 0) & (correction > 0))
                    | ((raw > self.ctx.max_code) & (correction < 0))
                )
                self.integral = np.where(allow, candidate, self.integral)
                update = self.ctx.initial_codes + self.p.kp * correction + self.integral - self.dc
            else:
                target = 0 if self.p.dither_objective == "peak" else self.p.dither_target_gradient_per_code
                update = dt * self.p.dither_gain_codes_per_s * (self.gradient - target) / scale_slope
            self.dc = np.clip(
                self.dc + update,
                self.p.dither_amplitude_codes,
                self.ctx.max_code - self.p.dither_amplitude_codes,
            )
        return np.clip(
            np.rint(
                self.dc + self.p.dither_amplitude_codes * np.sin(2 * np.pi * self.freq * observation.time_s)
            ),
            0,
            self.ctx.max_code,
        )


class LinearMPC:
    """Constrained local thermal MPC with a sensor-space Kalman observer.

    A random-walk output disturbance provides offset rejection. A nominal A/B/C
    model comes from the configured ROM and a local optical/heater linearization.
    This is useful near an identified flank; it is not global nonlinear acquisition.
    """

    def __init__(self, params, context):
        self.p, self.ctx = params, context
        a, b, c = context.a, context.b, context.c
        self.nx, self.n = a.shape[0], b.shape[1]
        self.a = np.block([[a, np.zeros((self.nx, self.n))], [np.zeros((self.n, self.nx)), np.eye(self.n)]])
        self.b = np.vstack([b, np.zeros((self.n, self.n))])
        self.c = np.hstack([c, np.eye(self.n)])
        self.x = np.zeros(self.nx + self.n)
        self.cov = np.eye(self.nx + self.n) * 0.01
        self.q = np.diag([1e-7] * self.nx + [1e-8] * self.n)
        self.r = np.eye(self.n) * 2e-7
        self.u = np.zeros(self.n)
        self.last_timestamp = -np.inf
        horizon = params.mpc_horizon
        self.pred = np.vstack([self.c @ np.linalg.matrix_power(self.a, k + 1) for k in range(horizon)])
        self.influence = np.zeros((horizon * self.n, horizon * self.n))
        for k in range(horizon):
            for j in range(k + 1):
                self.influence[k * self.n : (k + 1) * self.n, j * self.n : (j + 1) * self.n] = (
                    self.c @ np.linalg.matrix_power(self.a, k - j) @ self.b
                )
        # Scale inputs by DAC full range; scale outputs in ADC fractions.
        self.design = self.influence * context.max_code
        moves = np.eye(horizon * self.n) - np.eye(horizon * self.n, k=-self.n)
        self.moves = np.sqrt(params.mpc_move_penalty) * moves
        self.matrix = np.vstack([self.design, self.moves])

    def step(self, observation):
        self.x = self.a @ self.x + self.b @ self.u
        self.cov = self.a @ self.cov @ self.a.T + self.q
        # The nominal model includes command/output histories for transport delay.
        # A repeated delivered sample must not be counted as a fresh observation.
        if observation.valid and observation.sample_time_s > self.last_timestamp:
            self.last_timestamp = observation.sample_time_s
            residual = observation.adc_codes / self.ctx.adc_max_code - self.ctx.target_fraction
            gain = np.linalg.solve(self.c @ self.cov @ self.c.T + self.r, self.c @ self.cov).T
            self.x += gain @ (residual - self.c @ self.x)
            eye = np.eye(len(self.x)) - gain @ self.c
            self.cov = eye @ self.cov @ eye.T + gain @ self.r @ gain.T
        previous = np.zeros(self.p.mpc_horizon * self.n)
        previous[: self.n] = self.u / self.ctx.max_code
        rhs = np.concatenate([-self.pred @ self.x, np.sqrt(self.p.mpc_move_penalty) * previous])
        lower = np.tile(-self.ctx.initial_codes / self.ctx.max_code, self.p.mpc_horizon)
        upper = np.tile(1 - self.ctx.initial_codes / self.ctx.max_code, self.p.mpc_horizon)
        solution = lsq_linear(self.matrix, rhs, bounds=(lower, upper), tol=1e-6, max_iter=50)
        self.u = solution.x[: self.n] * self.ctx.max_code
        return np.clip(np.rint(self.ctx.initial_codes + self.u), 0, self.ctx.max_code)


class ScanAcquisition:
    """Sequential lane scan, using only monitor readings, then local controller.

    For a flank target it chooses the rising-code crossing (matching the nominal
    slope sign), not merely the closest value on either side. The finite range may
    find a wrong WDM alias: the simulator reports alias occupancy independently.
    """

    def __init__(self, params, context, inner):
        self.p, self.ctx, self.inner = params, context, inner
        self.codes = context.initial_codes.copy()
        self.lane, self.index, self.dwell = 0, 0, 0
        self.grid = np.linspace(
            max(0, params.scan_min_code), min(context.max_code, params.scan_max_code), params.scan_points
        )
        self.previous = None
        self.best = (np.inf, self.codes[0])
        self.done = False

    def step(self, observation):
        if self.done:
            return self.inner.step(observation)
        self.codes[self.lane] = self.grid[self.index]
        self.dwell += 1
        if self.dwell < self.p.scan_dwell_samples or not observation.valid:
            return self.codes.copy()
        self.dwell = 0
        y = observation.adc_codes[self.lane] / self.ctx.adc_max_code
        target = self.ctx.target_fraction[self.lane]
        if abs(y - target) < self.best[0]:
            self.best = (abs(y - target), self.grid[self.index])
        crossed = False
        if self.previous is not None:
            yp, cp = self.previous
            sign = np.sign(self.ctx.jacobian_per_code[self.lane, self.lane])
            crossed = ((yp - target) * (y - target) <= 0) and np.sign(y - yp) == sign
            if crossed and y != yp:
                self.best = (0, cp + (self.grid[self.index] - cp) * (target - yp) / (y - yp))
        self.previous = y, self.grid[self.index]
        self.index += 1
        if crossed or self.index == len(self.grid):
            self.codes[self.lane] = self.best[1]
            self.lane += 1
            self.index = 0
            self.previous = None
            if self.lane == len(self.codes):
                # Rebuild the same local controller around the acquired commands.
                from dataclasses import replace

                context = replace(self.ctx, initial_codes=self.codes.copy())
                self.inner = make_controller(replace(self.p, acquisition=False), context)
                self.done = True
            else:
                self.best = (np.inf, self.codes[self.lane])
        return np.rint(self.codes.copy())


def make_controller(params: ControllerParams, context: ControllerContext) -> Controller:
    classes = {"open_loop": OpenLoop, "pi": SideOfFringePI, "dither": DitherLock, "mpc": LinearMPC}
    inner = classes[params.kind](params, context)
    return ScanAcquisition(params, context, inner) if params.acquisition else inner
