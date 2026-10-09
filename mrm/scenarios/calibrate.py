"""Aggressor-power calibration: regularized, non-negative deconvolution of a target trajectory (07 P1-4, D6).

07 D6 decides that the fast on-die aggressor enters the plant as POWER into the thermal ROM's aggressor column, with
the waveform calibrated to a target stated at the victim ring. This module solves for that power waveform. It does
not import the thermal module: the plant's response from the aggressor site to the victim ring is a function
argument (a :class:`FosterResponse`, anything with ``taus_s`` and ``res_k_per_w``, or a callable step response).

Target and the single application of dlambda/dT
----------------------------------------------
:class:`AggressorTarget` holds a PRINTED temperature trajectory T_p(t) (degC on the slide, stored in K) and a
referencing hypothesis (07 D6, 4.8 ``aggressor.referencing``):

- ``resonance`` (default): the plotted degC is read as Lightmatter's conversion of a resonance shift with its own
  ~15 GHz/degC (85.86 pm/K at 1310 nm, printed aside in LM26-P section 1). The target is the resonance trajectory
  dlam(t) = s_ref * (T_p(t) - T_base). The model's own dlambda/dT, s(T) = s0 (1 + beta (T - T_ref)) (07 P0-3), is
  applied ONCE, here, to turn dlam(t) into the ring-temperature rise x(t) that produces it:
  dlam = s0 [(1 + beta (T_base - T_ref)) x + beta x^2 / 2]. The plant then applies the same s(T) on the way back,
  so the plant's resonance shift equals the target for any dlambda/dT hypothesis (07 P1-4: a degC target under
  H_solver would understate the disturbance by 1.2 to 1.3 times).
- ``temperature``: the plotted degC is a thermometer reading that sees a fraction f of the ring's rise
  (``thermometer_fraction``; 1 = the ring itself, 0.77-0.86 = the H86_sensor reading of 07 D2). Then
  x(t) = (T_p(t) - T_base) / f, and no dlambda/dT is applied to the target.

Deconvolution
-------------
The power is piecewise linear on n uniform knots (exact first-order hold in the ROM, 07 P0-10). With hat basis
functions phi_j and the ramp response R(t) = int_0^t g (g = step response, K/W), the response to phi_j is
H(t - t_j) = [R(t - t_j + D) - 2 R(t - t_j) + R(t - t_j - D)] / D (D = knot spacing). For a periodic target the
response is the periodic steady state, sum over all past periods; for a Foster response
g = sum_k r_k (1 - exp(-t/tau_k)) the sum is a closed-form geometric series (exact). The power solves

    min_p || A p - x ||^2 + (lam ||A||_2)^2 || D_k p ||^2     subject to 0 <= p <= p_max

(Tikhonov with a periodic difference operator of order ``reg_order``; scipy.optimize.lsq_linear, BVLS). p >= 0
because a heater can only heat (PAT-1 [0050], verbatim in 01). The residual on the collocation grid is reported in
K of ring rise and, for resonance targets, in m of resonance. The result is tagged ``calibrated`` and records the
assumed thermal response (07 P1-4).
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from mrm.scenarios import generators as _g

__all__ = [
    "LM_DLAMBDA_DT_M_PER_K", "LAMBDA_REF_M", "FosterResponse", "foster_from_system", "AggressorTarget",
    "resonance_to_rise_k", "rise_to_resonance_m", "CalibratedWaveform", "periodic_response_matrix",
    "transient_response_matrix", "calibrate_aggressor_power", "calibrate_scenario", "simulate_foster_foh",
]

_C0 = 299_792_458.0          # speed of light, m/s (exact)
#: Reference wavelength of the G26 O-band grid, m: 1310 nm (channel_plan centre, c/1310 nm = 228.849 THz; LM26-P).
LAMBDA_REF_M = 1310e-9
#: Lightmatter's printed ~15 GHz/degC (LM26-P section 1, an aside; 01 section 2.5 verified) as a wavelength
#: sensitivity at 1310 nm: 15e9 * lambda^2 / c = 85.86 pm/K. Used only to express printed degC as resonance.
LM_DLAMBDA_DT_M_PER_K = 15.0e9 * LAMBDA_REF_M ** 2 / _C0


# =====================================================================================================================
# Responses
# =====================================================================================================================

@dataclass(frozen=True, eq=False)
class FosterResponse:
    """Step response g(t) = sum_k r_k (1 - exp(-t/tau_k)) for t > 0 (K/W), the Foster form of the thermal ROM
    (ARCHITECTURE 5.3; mrm.plant.thermal). Mixed-sign residues are allowed (non-collocated pairs)."""
    taus_s: np.ndarray
    res_k_per_w: np.ndarray
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        t = np.atleast_1d(np.asarray(self.taus_s, dtype=np.float64))
        r = np.atleast_1d(np.asarray(self.res_k_per_w, dtype=np.float64))
        if t.shape != r.shape or t.ndim != 1:
            raise ValueError("taus_s and res_k_per_w must be 1-D arrays of the same length")
        if np.any(~(t > 0)) or not np.all(np.isfinite(r)):
            raise ValueError("taus must be > 0 and residues finite")
        object.__setattr__(self, "taus_s", t)
        object.__setattr__(self, "res_k_per_w", r)

    def step(self, t_s) -> np.ndarray:
        """g(t) in K/W (0 for t <= 0)."""
        t = np.asarray(t_s, dtype=np.float64)[..., None]
        v = np.sum(self.res_k_per_w * -np.expm1(-np.clip(t, 0, None) / self.taus_s), axis=-1)
        return np.where(t[..., 0] > 0, v, 0.0)

    def ramp(self, t_s) -> np.ndarray:
        """R(t) = int_0^t g in K s/W (0 for t <= 0), with a series for t << tau (no cancellation)."""
        t = np.clip(np.asarray(t_s, dtype=np.float64), 0.0, None)[..., None]
        y = t / self.taus_s
        series = t * y * (0.5 - y / 6.0 + y * y / 24.0)
        exact = t + self.taus_s * np.expm1(-y)
        v = np.sum(self.res_k_per_w * np.where(y < 1e-3, series, exact), axis=-1)
        return v

    def dc(self) -> float:
        return float(np.sum(self.res_k_per_w))

    def describe(self) -> dict:
        return {"form": "foster", "taus_s": self.taus_s.tolist(), "res_k_per_w": self.res_k_per_w.tolist(),
                **dict(self.info)}


def _as_response(r) -> FosterResponse | Callable:
    if isinstance(r, FosterResponse):
        return r
    if hasattr(r, "taus_s") and hasattr(r, "res_k_per_w"):
        return FosterResponse(r.taus_s, r.res_k_per_w)
    if callable(r):
        return r
    raise TypeError("a response must be a FosterResponse, an object with taus_s/res_k_per_w, or a callable step "
                    "response g(t) in K/W")


def foster_from_system(system, *, victim_ring: int, site: str, observer: str = "ring_mode_weighted",
                       min_abs_res_k_per_w: float = 0.0) -> list[FosterResponse]:
    """Per-world Foster responses (aggressor ``site`` -> ``observer`` of ``victim_ring``) from a thermal system.

    Duck-typed on mrm.plant.thermal.ThermalSystem (no import): uses ``observer_index(kind, ring=)``,
    ``source_index('aggressor', name=)``, ``lag_obs``, ``lag_src``, ``taus_s`` (B, M) and ``res_k_per_w`` (B, M).
    """
    o = system.observer_index(observer, ring=victim_ring)
    s = system.source_index("aggressor", name=site)
    m = np.flatnonzero((np.asarray(system.lag_obs) == o) & (np.asarray(system.lag_src) == s))
    if m.size == 0:
        raise ValueError(f"no thermal path from aggressor {site!r} to {observer} of ring {victim_ring}")
    taus = np.asarray(system.taus_s)[:, m]
    res = np.asarray(system.res_k_per_w)[:, m]
    out = []
    for b in range(taus.shape[0]):
        keep = np.abs(res[b]) > min_abs_res_k_per_w
        out.append(FosterResponse(taus[b, keep], res[b, keep],
                                  info={"source": f"thermal system: aggressor {site} -> {observer}[{victim_ring}]",
                                        "world": b}))
    return out


# =====================================================================================================================
# Target
# =====================================================================================================================

def resonance_to_rise_k(shift_m, *, s0_m_per_k, beta_per_k: float = 0.0, t_ref_k: float | None = None,
                        t_base_k: float) -> np.ndarray:
    """Ring-temperature rise x (K) above ``t_base_k`` (the ring temperature before the rise) that shifts the
    resonance by ``shift_m`` (m) under the model's s(T) = s0 (1 + beta (T - T_ref)) (07 P0-3, thermo_optic_order 2;
    beta = 0 is order 1; PlantParams thermal.dlambda_dt_s0_m_per_k, _beta_per_k, _t0_k):
    shift = int_{T_base}^{T_base + x} s dT = s0 [(1 + beta (T_base - T_ref)) x + beta x^2 / 2], solved in the
    cancellation-free form x = 2 shift / (b + sqrt(b^2 + 4 a shift)), a = s0 beta / 2, b = s0 (1 + beta (T_base -
    T_ref))."""
    d = np.asarray(shift_m, dtype=np.float64)
    tr = t_base_k if t_ref_k is None else t_ref_k
    b = s0_m_per_k * (1.0 + beta_per_k * (t_base_k - tr))
    a = 0.5 * s0_m_per_k * beta_per_k
    disc = b * b + 4.0 * a * d
    if np.any(disc < 0) or np.any(b <= 0):
        raise ValueError("the resonance target is outside the range the model's s(T) can reach")
    return 2.0 * d / (b + np.sqrt(disc))


def rise_to_resonance_m(rise_k, *, s0_m_per_k, beta_per_k: float = 0.0, t_ref_k: float | None = None,
                        t_base_k: float) -> np.ndarray:
    """Forward map of :func:`resonance_to_rise_k`: shift (m) of a ring-temperature rise (K) above t_base_k."""
    x = np.asarray(rise_k, dtype=np.float64)
    tr = t_base_k if t_ref_k is None else t_ref_k
    return s0_m_per_k * ((1.0 + beta_per_k * (t_base_k - tr)) * x + 0.5 * beta_per_k * x * x)


@dataclass(frozen=True, eq=False)
class AggressorTarget:
    """A printed aggressor trajectory and its referencing hypothesis (07 D6, P1-4; see the module docstring).

    form: ``rectified_sine`` T_p = t_max_k - delta_k |sin(pi (t + phase_s) / period_s)|;
          ``trapezoid`` T_p = t_low_k + delta_k * trap(t + phase_s) with rise_s, high_s, fall_s (triangles and
          one-sided sawtooths are trapezoids);
          ``table`` T_p by linear interpolation in (table_t_s, table_k) (periodic when period_s is set).
    period_s: None for a one-shot (transient) target over [0, horizon_s].
    t_base_k: base temperature of the test (aggressor off); the excursion is T_p - t_base_k.
    units: ``resonance`` | ``temperature``; s_ref_m_per_k: the conversion behind the printed degC (resonance only);
    thermometer_fraction: fraction of the ring rise the printed thermometer sees (temperature only).
    """
    units: str = "resonance"
    form: str = "rectified_sine"
    period_s: float | None = 0.1307
    t_base_k: float = 298.15
    t_max_k: float = 386.75
    t_low_k: float = 298.15
    delta_k: float = 88.5
    rise_s: float = 0.0
    high_s: float = 0.0
    fall_s: float = 0.0
    phase_s: float = 0.0
    table_t_s: Any = None
    table_k: Any = None
    horizon_s: float | None = None
    s_ref_m_per_k: float = LM_DLAMBDA_DT_M_PER_K
    thermometer_fraction: float = 1.0
    name: str = ""
    provenance: str = "read-off"
    source: str = ""
    note: str = ""

    def __post_init__(self):
        if self.units not in ("resonance", "temperature"):
            raise ValueError(f"target units must be resonance or temperature, got {self.units!r} (07 D6)")
        if self.form not in ("rectified_sine", "trapezoid", "table"):
            raise ValueError(f"unknown target form {self.form!r}")
        if self.period_s is None and not (self.horizon_s and self.horizon_s > 0):
            raise ValueError("a one-shot target (period_s None) needs horizon_s > 0")
        if self.period_s is not None and not self.period_s > 0:
            raise ValueError("period_s must be > 0")
        if self.form == "trapezoid":
            per = self.period_s if self.period_s is not None else math.inf
            if min(self.rise_s, self.high_s, self.fall_s) < 0 or self.rise_s + self.high_s + self.fall_s > per:
                raise ValueError("trapezoid rise/high/fall must be >= 0 and fit in the period")
        if self.form == "table":
            t = np.asarray(self.table_t_s, dtype=np.float64)
            k = np.asarray(self.table_k, dtype=np.float64)
            if t.ndim != 1 or t.shape != k.shape or t.size < 2 or np.any(np.diff(t) <= 0):
                raise ValueError("table target needs increasing table_t_s and table_k of equal length")
        if not 0 < self.thermometer_fraction <= 1.0:
            raise ValueError("thermometer_fraction must be in (0, 1]")
        if not self.s_ref_m_per_k > 0:
            raise ValueError("s_ref_m_per_k must be > 0")

    # ---- the printed trajectory
    def printed_k(self, t_s) -> np.ndarray:
        """Printed temperature T_p(t) in K."""
        x = np.asarray(t_s, dtype=np.float64) + self.phase_s
        if self.form == "rectified_sine":
            return self.t_max_k - self.delta_k * _g.np_rect_sine(x, self.period_s)
        if self.form == "trapezoid":
            per = self.period_s if self.period_s is not None else 1e300
            return self.t_low_k + self.delta_k * _g.np_trapezoid(x, per, self.rise_s, self.high_s, self.fall_s)
        t = np.asarray(self.table_t_s, dtype=np.float64)
        k = np.asarray(self.table_k, dtype=np.float64)
        if self.period_s is not None:
            x = t[0] + np.mod(x - t[0], self.period_s)
            return np.interp(x, np.append(t, t[0] + self.period_s), np.append(k, k[0]))
        return np.interp(x, t, k)

    def excursion_k(self, t_s) -> np.ndarray:
        """T_p(t) - t_base_k (K)."""
        return self.printed_k(t_s) - self.t_base_k

    def resonance_m(self, t_s) -> np.ndarray:
        """Resonance target s_ref (T_p - T_base), m (``resonance`` referencing)."""
        return self.s_ref_m_per_k * self.excursion_k(t_s)

    def ring_rise_k(self, t_s, *, s0_m_per_k=None, beta_per_k: float = 0.0, t_ref_k: float | None = None,
                    ring_t0_k: float | None = None):
        """Aggressor-induced rise of the victim ring (K) that the plant must produce. dlambda/dT of the MODEL is
        applied here exactly once (resonance referencing): the rise starts from ``ring_t0_k``, the ring temperature
        with the aggressor off (default t_base_k: heater off; pass the lock temperature, PlantParams
        thermal.t_ring_lock_k, for the shift seen by a locked ring -- with beta ~1.4e-3/K the two differ by about
        beta x / 2, ~7 % at a 100 K rise). Temperature referencing divides by the thermometer fraction instead."""
        if self.units == "temperature":
            return self.excursion_k(t_s) / self.thermometer_fraction
        if s0_m_per_k is None:
            raise ValueError("a resonance target needs the model's dlambda/dT (s0_m_per_k, beta_per_k, t_ref_k)")
        return resonance_to_rise_k(self.resonance_m(t_s), s0_m_per_k=s0_m_per_k, beta_per_k=beta_per_k,
                                   t_ref_k=t_ref_k, t_base_k=self.t_base_k if ring_t0_k is None else ring_t0_k)

    def peak_rate_k_per_s(self) -> float:
        """Peak |dT_p/dt| of the printed trajectory, K/s (rectified sine: pi delta / P)."""
        if self.form == "rectified_sine":
            return math.pi * self.delta_k / self.period_s
        if self.form == "trapezoid":
            r = [self.delta_k / x for x in (self.rise_s, self.fall_s) if x > 0]
            return max(r) if r else math.inf
        t = np.asarray(self.table_t_s, dtype=np.float64)
        k = np.asarray(self.table_k, dtype=np.float64)
        return float(np.max(np.abs(np.diff(k) / np.diff(t))))

    def mean_rate_k_per_s(self) -> float:
        """Mean |dT_p/dt| over one period (rectified sine: 2 delta / P)."""
        if self.period_s is None:
            raise ValueError("mean rate needs a periodic target")
        t = np.linspace(0, self.period_s, 20001)
        return float(np.mean(np.abs(np.gradient(self.printed_k(t), t))))

    def describe(self) -> dict:
        d = {k: getattr(self, k) for k in ("units", "form", "period_s", "t_base_k", "t_max_k", "t_low_k",
                                           "delta_k", "rise_s", "high_s", "fall_s", "phase_s", "horizon_s",
                                           "s_ref_m_per_k", "thermometer_fraction", "name", "provenance",
                                           "source", "note")}
        if self.form == "table":
            d["table_t_s"] = np.asarray(self.table_t_s, dtype=np.float64).tolist()
            d["table_k"] = np.asarray(self.table_k, dtype=np.float64).tolist()
        return d


# =====================================================================================================================
# Response matrices
# =====================================================================================================================

def _gl_nodes(n: int = 8):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def _ramp_grid(step: Callable, delta: float, m_max: int) -> np.ndarray:
    """R(m*delta) for m = 0..m_max from a callable step response, by 8-point Gauss-Legendre on each interval; the
    first interval is graded towards t = 0 (fast poles make g steep there)."""
    xn, wn = _gl_nodes(8)
    a = np.arange(m_max, dtype=np.float64)[:, None] * delta
    t = a + xn[None, :] * delta
    vals = np.asarray(step(t.ravel()), dtype=np.float64).reshape(t.shape)
    integ = delta * (vals @ wn)
    # graded first interval [0, delta]: [delta 2^-(k+1), delta 2^-k], k = 0..39, plus [0, delta 2^-40]
    edges = delta * np.concatenate([[0.0], 2.0 ** -np.arange(40, -1, -1)])
    first = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        tt = lo + xn * (hi - lo)
        first += (hi - lo) * float(np.asarray(step(tt), dtype=np.float64) @ wn)
    if m_max > 0:
        integ[0] = first
    return np.concatenate([[0.0], np.cumsum(integ)])


def _foster_hat(resp: FosterResponse, x: np.ndarray, D: float) -> np.ndarray:
    """H(x) = [R(x + D) - 2 R(x) + R(x - D)] / D for a Foster response (closed form for x >= D)."""
    x = np.asarray(x, dtype=np.float64)
    tau = resp.taus_s
    r = resp.res_k_per_w
    out = np.zeros_like(x)
    big = x >= D
    if np.any(big):
        xb = x[big][:, None]
        q = -np.expm1(-D / tau)                                   # 1 - exp(-D/tau)
        out[big] = np.sum(r * tau * np.exp(-(xb - D) / tau) * q * q, axis=1) / D
    small = ~big
    if np.any(small):
        xs = x[small]
        out[small] = (resp.ramp(xs + D) - 2.0 * resp.ramp(xs) + resp.ramp(xs - D)) / D
    return out


def periodic_response_matrix(resp, period_s: float, n_knots: int, oversample: int = 4, *,
                             settle_s: float | None = None) -> tuple[np.ndarray, dict]:
    """A (n_knots*oversample, n_knots): periodic-steady-state response at s_i = i*D/os to the periodic hat basis on
    knots t_j = j*D, D = period/n_knots (K/W). Exact for a Foster response; for a callable step response the sum
    over past periods is truncated after the response has settled (``settle_s``, estimated when None)."""
    resp = _as_response(resp)
    n = int(n_knots)
    os_ = int(oversample)
    P = float(period_s)
    D = P / n
    m = n * os_
    u = np.arange(m) * (D / os_)                                  # grid of (s - t_j) mod P
    info: dict[str, Any] = {}
    if isinstance(resp, FosterResponse):
        tau = resp.taus_s
        r = resp.res_k_per_w
        q = -np.expm1(-D / tau)
        geo = np.sum(r * tau * q * q * np.exp(-(u[:, None] + P - D) / tau) / (-np.expm1(-P / tau)), axis=1) / D
        phi = _foster_hat(resp, u, D) + geo
        wrap = u > P - D
        phi[wrap] += resp.ramp(u[wrap] - P + D) / D
        info["truncation"] = "exact (closed-form geometric sum)"
    else:
        step = resp
        if settle_s is None:
            settle_s = _estimate_settle(step, P)
        Q = int(math.ceil(settle_s / P)) + 1
        if Q > 5000:
            warnings.warn(f"response settles after {settle_s:.3g} s = {Q} periods; truncated at 5000",
                          RuntimeWarning, stacklevel=2)
            Q = 5000
        Rg = _ramp_grid(step, D / os_, (Q + 2) * m + 2 * os_)          # one extra period for the tail estimate

        def R(idx):
            idx = np.asarray(idx)
            return np.where(idx > 0, Rg[np.clip(idx, 0, Rg.size - 1)], 0.0)
        qi = np.arange(m)
        phi = np.zeros(m)
        for p in range(Q + 1):
            k = qi + p * m
            phi += (R(k + os_) - 2.0 * R(k) + R(k - os_)) / D
        wrap = qi > m - os_
        k = qi[wrap] - m + os_
        phi[wrap] += R(k) / D
        tail = (R(qi + (Q + 1) * m + os_) - 2 * R(qi + (Q + 1) * m) + R(qi + (Q + 1) * m - os_)) / D
        info["truncation"] = f"{Q + 1} periods (settle {settle_s:.3g} s); last term max {np.max(np.abs(tail)):.3g} K/W"
    rows = np.arange(m)[:, None]
    cols = np.arange(n)[None, :]
    A = phi[(rows - cols * os_) % m]
    return A, info


def transient_response_matrix(resp, horizon_s: float, n_knots: int, oversample: int = 4) -> np.ndarray:
    """A (n_knots*os, n_knots): response at s_i = i*D/os (i = 1..n*os) from rest to hats on t_j = j*D (j = 1..n),
    D = horizon/n; the power is 0 at t = 0 (it starts from rest)."""
    resp = _as_response(resp)
    n = int(n_knots)
    os_ = int(oversample)
    D = float(horizon_s) / n
    m = n * os_
    lag = np.arange(-n * os_, m + 1)                               # (i - j*os) in grid steps
    if isinstance(resp, FosterResponse):
        H = _foster_hat(resp, lag * (D / os_), D)
    else:
        Rg = _ramp_grid(resp, D / os_, m + 2 * os_)
        k = lag
        def R(idx):
            return np.where(idx > 0, Rg[np.clip(idx, 0, Rg.size - 1)], 0.0)
        H = (R(k + os_) - 2.0 * R(k) + R(k - os_)) / D
    rows = np.arange(1, m + 1)[:, None]
    cols = np.arange(1, n + 1)[None, :]
    return H[rows - cols * os_ + n * os_]


def _estimate_settle(step: Callable, period: float) -> float:
    t = period * 2.0 ** np.arange(0, 40)
    g = np.asarray(step(t), dtype=np.float64)
    scale = max(float(np.max(np.abs(g))), 1e-300)
    for i in range(len(t) - 1):
        if abs(g[i + 1] - g[i]) <= 1e-9 * scale:
            return float(t[i])
    warnings.warn("step response did not settle within 2**39 periods", RuntimeWarning, stacklevel=3)
    return float(t[-1])


def _diff_operator(n: int, order: int, periodic: bool) -> np.ndarray:
    if order == 0:
        return np.eye(n)
    D = np.eye(n)
    for _ in range(order):
        if periodic:
            D = np.roll(np.eye(n), 1, axis=1) @ D - D
        else:
            D = (np.eye(D.shape[0])[1:] - np.eye(D.shape[0])[:-1]) @ D if D.shape[0] > 1 else D
    return D


# =====================================================================================================================
# Calibration
# =====================================================================================================================

@dataclass(frozen=True, eq=False)
class CalibratedWaveform:
    """The calibrated aggressor power (07 P1-4): piecewise linear on ``t_knots_s`` (uniform, from 0), one row per
    world. Residuals are on the collocation grid: K of victim-ring rise, and m of resonance for resonance targets.
    ``record()`` is what Scenario.with_aggressor_power stores (provenance ``calibrated``)."""
    t_knots_s: np.ndarray
    power_w: np.ndarray
    period_s: float | None
    residual_rms_k: np.ndarray
    residual_max_k: np.ndarray
    target_rms_k: np.ndarray
    residual_rms_m: np.ndarray | None
    residual_max_m: np.ndarray | None
    offset_k: np.ndarray | None
    lower_bound_fraction: np.ndarray
    units: str
    reg: float
    reg_order: int
    oversample: int
    response_info: tuple
    target_info: Mapping[str, Any]
    model_info: Mapping[str, Any]
    provenance: str = "calibrated"

    @property
    def n_knots(self) -> int:
        return int(self.t_knots_s.size)

    @property
    def peak_power_w(self) -> np.ndarray:
        return np.max(self.power_w, axis=1)

    @property
    def mean_power_w(self) -> np.ndarray:
        return np.mean(self.power_w, axis=1) if self.period_s is not None else np.mean(self.power_w, axis=1)

    def power_at(self, t_s, b: int = 0) -> np.ndarray:
        """Power (W) of world b at times t_s (linear between knots; periodic or held after the last knot)."""
        t = np.asarray(t_s, dtype=np.float64)
        tk = self.t_knots_s
        p = self.power_w[b]
        if self.period_s is not None:
            return np.interp(np.mod(t, self.period_s), np.append(tk, self.period_s), np.append(p, p[0]))
        return np.interp(t, np.concatenate([[0.0], tk]), np.concatenate([[0.0], p]))

    def record(self) -> dict:
        return {
            "provenance": self.provenance, "source": "mrm.scenarios.calibrate.calibrate_aggressor_power",
            "note": f"calibrated to the {self.units} target (07 P1-4); residual rms "
                    f"{np.array2string(self.residual_rms_k, precision=3)} K",
            "residual_rms_k": self.residual_rms_k.tolist(), "residual_max_k": self.residual_max_k.tolist(),
            "residual_rms_m": None if self.residual_rms_m is None else self.residual_rms_m.tolist(),
            "residual_max_m": None if self.residual_max_m is None else self.residual_max_m.tolist(),
            "offset_k": None if self.offset_k is None else self.offset_k.tolist(),
            "lower_bound_fraction": self.lower_bound_fraction.tolist(),
            "peak_power_w": self.peak_power_w.tolist(), "mean_power_w": self.mean_power_w.tolist(),
            "reg": self.reg, "reg_order": self.reg_order, "n_knots": self.n_knots, "oversample": self.oversample,
            "assumed_thermal_response": list(self.response_info), "target": dict(self.target_info),
            "model": dict(self.model_info),
        }


def _per_world(v, nb: int, name: str) -> np.ndarray:
    a = np.atleast_1d(np.asarray(v, dtype=np.float64))
    if a.size == 1:
        return np.full(nb, float(a[0]))
    if a.size != nb:
        raise ValueError(f"{name} has {a.size} values but there are {nb} worlds")
    return a


def calibrate_aggressor_power(response, target: AggressorTarget, *, s0_m_per_k=None, beta_per_k=0.0,
                              t_ref_k=None, ring_t0_k=None, n_knots: int = 256, oversample: int = 4,
                              reg: float = 1e-4, reg_order: int = 1, p_max_w: float | None = None,
                              fit_offset: bool = False, settle_s: float | None = None,
                              response_info: Sequence[Mapping] | Mapping | None = None) -> CalibratedWaveform:
    """Solve for the aggressor power waveform that drives the victim ring along ``target`` (07 P1-4, D6).

    response: the aggressor-site -> victim-ring response (FosterResponse, an object with taus_s/res_k_per_w, a
        callable step response g(t) in K/W), or a sequence of them, one per uncertainty world (B worlds).
    target: AggressorTarget (periodic -> periodic steady state; one-shot -> from rest over its horizon).
    s0_m_per_k, beta_per_k, t_ref_k: the MODEL's dlambda/dT, s(T) = s0 (1 + beta (T - T_ref)) (scalar or (B,);
        PlantParams thermal.dlambda_dt_s0_m_per_k[:, victim], _beta_per_k[:, 0], _t0_k[:, 0]); needed for resonance
        targets (applied once, module docstring). ring_t0_k: ring temperature with the aggressor off (scalar or (B,);
        default the target's base temperature; see AggressorTarget.ring_rise_k).
    n_knots, oversample: knots of the piecewise-linear power and collocation points per knot.
    reg, reg_order: Tikhonov weight relative to ||A||_2 and the order of the (periodic) difference operator.
    p_max_w: optional upper bound (the aggressor heater's limit); fit_offset: also fit a constant offset of the rise
        (unbounded), reported in ``offset_k`` (it would have to come from the base temperature).
    Returns a CalibratedWaveform tagged ``calibrated`` with residuals and the assumed response.
    """
    resps = list(response) if isinstance(response, (list, tuple)) else [response]
    resps = [_as_response(r) for r in resps]
    nb = len(resps)
    if response_info is None:
        rinfo = tuple(r.describe() if isinstance(r, FosterResponse) else {"form": "callable step response",
                                                                          "repr": repr(r)} for r in resps)
    else:
        rinfo = tuple(response_info) if not isinstance(response_info, Mapping) else (dict(response_info),) * nb
    periodic = target.period_s is not None
    n = int(n_knots)
    os_ = int(oversample)
    if n < 4 or os_ < 1:
        raise ValueError("n_knots must be >= 4 and oversample >= 1")
    if periodic:
        P = float(target.period_s)
        D = P / n
        s = np.arange(n * os_) * (D / os_)
        tk = np.arange(n) * D
    else:
        Hz = float(target.horizon_s)
        D = Hz / n
        s = np.arange(1, n * os_ + 1) * (D / os_)
        tk = np.arange(1, n + 1) * D
    s0 = None if s0_m_per_k is None else _per_world(s0_m_per_k, nb, "s0_m_per_k")
    beta = _per_world(beta_per_k, nb, "beta_per_k")
    tref = None if t_ref_k is None else _per_world(t_ref_k, nb, "t_ref_k")
    rt0 = _per_world(target.t_base_k if ring_t0_k is None else ring_t0_k, nb, "ring_t0_k")
    if target.units == "resonance" and s0 is None:
        raise ValueError("a resonance target needs the model's s0_m_per_k (07 D6: dlambda/dT applied once)")
    Dop = _diff_operator(n, int(reg_order), periodic)
    powers = np.zeros((nb, n))
    rms = np.zeros(nb)
    rmax = np.zeros(nb)
    trms = np.zeros(nb)
    rms_m = np.zeros(nb) if target.units == "resonance" else None
    max_m = np.zeros(nb) if target.units == "resonance" else None
    offs = np.zeros(nb) if fit_offset else None
    lbf = np.zeros(nb)
    cache: dict[int, np.ndarray] = {}
    for b, rsp in enumerate(resps):
        key = id(rsp)
        if key not in cache:
            if periodic:
                A, _info = periodic_response_matrix(rsp, P, n, os_, settle_s=settle_s)
            else:
                A = transient_response_matrix(rsp, Hz, n, os_)
            cache[key] = A
        A = cache[key]
        y = target.ring_rise_k(s, s0_m_per_k=None if s0 is None else s0[b], beta_per_k=float(beta[b]),
                               t_ref_k=None if tref is None else float(tref[b]), ring_t0_k=float(rt0[b]))
        lam = float(reg) * float(np.linalg.norm(A, 2))
        Ab = A
        Db = Dop
        lb = np.zeros(n)
        ub = np.full(n, np.inf if p_max_w is None else float(p_max_w))
        if fit_offset:
            Ab = np.hstack([A, np.ones((A.shape[0], 1))])
            Db = np.hstack([Dop, np.zeros((Dop.shape[0], 1))])
            lb = np.append(lb, -np.inf)
            ub = np.append(ub, np.inf)
        M = np.vstack([Ab, lam * Db])
        rhs = np.concatenate([y, np.zeros(Db.shape[0])])
        sol = lsq_linear(M, rhs, bounds=(lb, ub), method="bvls", lsq_solver="exact", max_iter=50 * n)
        if not sol.success:
            warnings.warn(f"aggressor calibration, world {b}: BVLS stopped before convergence ({sol.message}); "
                          f"the residual reported below is that of the unconverged power", RuntimeWarning,
                          stacklevel=2)
        x = sol.x
        p = np.clip(x[:n], 0.0, None)
        powers[b] = p
        fit = A @ p + (x[n] if fit_offset else 0.0)
        res = fit - y
        rms[b] = float(np.sqrt(np.mean(res ** 2)))
        rmax[b] = float(np.max(np.abs(res)))
        trms[b] = float(np.sqrt(np.mean(y ** 2)))
        lbf[b] = float(np.mean(p <= 1e-12 * max(float(np.max(p)), 1e-300)))
        if fit_offset:
            offs[b] = float(x[n])
        if target.units == "resonance":
            got = rise_to_resonance_m(fit, s0_m_per_k=s0[b], beta_per_k=float(beta[b]),
                                      t_ref_k=None if tref is None else float(tref[b]), t_base_k=float(rt0[b]))
            want = target.resonance_m(s)
            rms_m[b] = float(np.sqrt(np.mean((got - want) ** 2)))
            max_m[b] = float(np.max(np.abs(got - want)))
    model = {"s0_m_per_k": None if s0 is None else s0.tolist(), "beta_per_k": beta.tolist(),
             "t_ref_k": None if tref is None else tref.tolist(), "ring_t0_k": rt0.tolist(),
             "dlambda_dt_applied": "once, in the target (resonance referencing)" if target.units == "resonance"
             else "not applied to the target (temperature referencing)"}
    return CalibratedWaveform(t_knots_s=tk if periodic else tk, power_w=powers, period_s=target.period_s,
                              residual_rms_k=rms, residual_max_k=rmax, target_rms_k=trms, residual_rms_m=rms_m,
                              residual_max_m=max_m, offset_k=offs, lower_bound_fraction=lbf, units=target.units,
                              reg=float(reg), reg_order=int(reg_order), oversample=os_, response_info=rinfo,
                              target_info=target.describe(), model_info=model)


def calibrate_scenario(scenario: "_g.Scenario", responses: Mapping[str, Any], *, model: Mapping[str, Any] | None = None,
                       **kw) -> tuple["_g.Scenario", dict[str, CalibratedWaveform]]:
    """Calibrate every pending aggressor of ``scenario`` and attach the power (07 P1-4).

    responses: {site: response or per-world sequence} (see calibrate_aggressor_power; build them with
    foster_from_system(system, victim_ring=, site=)). model: {s0_m_per_k, beta_per_k, t_ref_k[, ring_t0_k]} of the
    plant (needed for resonance targets; per world allowed), or {site: {...}} per aggressor site when the victims
    differ (s0 is per victim ring). Extra keyword arguments go to calibrate_aggressor_power. A fitted offset
    (``fit_offset=True``) is reported in CalibratedWaveform.offset_k but NOT applied to the scenario (it belongs to
    the base temperature; a warning says so). Returns the calibrated scenario and {site: CalibratedWaveform}.
    """
    model = dict(model or {})
    per_site = bool(model) and all(isinstance(v, Mapping) for v in model.values())
    out = {}
    sc = scenario
    for a in scenario.aggressors:
        if not a.pending:
            continue
        if a.site not in responses:
            raise _g.ScenarioError(f"no thermal response given for aggressor site {a.site!r}")
        if per_site and a.site not in model:
            raise _g.ScenarioError(f"model is given per site but has no entry for aggressor site {a.site!r}")
        m = dict(model[a.site]) if per_site else model
        cw = calibrate_aggressor_power(responses[a.site], a.target, **m, **kw)
        if cw.offset_k is not None and np.any(cw.offset_k != 0.0):
            warnings.warn(f"aggressor {a.site!r}: fitted offset {np.array2string(cw.offset_k, precision=3)} K is not "
                          f"applied to the scenario; move the base temperature by it if it is meant", RuntimeWarning,
                          stacklevel=2)
        if cw.period_s is not None:
            sc = sc.with_aggressor_power(a.site, cw.t_knots_s, cw.power_w, period_s=cw.period_s, record=cw.record())
        else:
            # one-shot: knots start at D; prepend the zero power at t = 0
            tk = np.concatenate([[0.0], cw.t_knots_s])
            pw = np.concatenate([np.zeros((cw.power_w.shape[0], 1)), cw.power_w], axis=1)
            sc = sc.with_aggressor_power(a.site, tk, pw, period_s=None, record=cw.record())
        out[a.site] = cw
    return sc, out


# =====================================================================================================================
# Independent check: exact first-order-hold simulation of a Foster response
# =====================================================================================================================

def simulate_foster_foh(resp: FosterResponse, u_w: np.ndarray, h_s: float, x0=None) -> tuple[np.ndarray, np.ndarray]:
    """Rise (K) at the sample instants of a Foster response driven by samples u_w (n+1,) linear between samples
    (exact FOH per pole: x <- a x + r (b0 u0 + b1 u1), b1 = 1 - (1 - a)/y, b0 = (1 - a)/y - a, y = h/tau; the same
    discretization as mrm.plant.thermal). Returns (rise (n+1,), final pole states)."""
    tau = resp.taus_s
    r = resp.res_k_per_w
    y = h_s / tau
    a = np.exp(-y)
    om = -np.expm1(-y)
    b1 = 1.0 - om / y
    b0 = om / y - a
    x = np.zeros_like(tau) if x0 is None else np.array(x0, dtype=np.float64)
    u = np.asarray(u_w, dtype=np.float64)
    out = np.zeros(u.size)
    out[0] = x.sum()
    for k in range(u.size - 1):
        x = a * x + r * (b0 * u[k] + b1 * u[k + 1])
        out[k + 1] = x.sum()
    return out, x
