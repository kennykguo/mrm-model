"""Passive thermal networks in SI units, with absolute Kelvin states.

The network equation is ``C dT/dt = -G T + B p + H T_boundary``.
G contains both internal and boundary conductances.  No assumption of a
constant ambient temperature is hidden in the state definition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _immutable(value: ArrayLike) -> NDArray[np.float64]:
    result = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("thermal arrays must contain finite values")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class ThermalNetwork:
    """A conservative thermal RC network, including optional insulated parts.

    Each source is a total heat input in watts: its B column must sum to one.
    Each observation is a weighted mean temperature: its W row sums to one.
    H can have multiple columns for independently varying thermal boundaries.
    Nodes with no path to a boundary conserve energy but have a zero pole;
    ``is_asymptotically_stable`` distinguishes those from anchored networks.
    """

    capacity_j_per_k: ArrayLike
    conductance_w_per_k: ArrayLike
    source_map: ArrayLike
    boundary_conductance_w_per_k: ArrayLike
    observer_matrix: ArrayLike
    node_names: tuple[str, ...] = ()
    source_names: tuple[str, ...] = ()
    observer_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("capacity_j_per_k", "conductance_w_per_k", "source_map",
                     "boundary_conductance_w_per_k", "observer_matrix"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))
        c, g, b, h, w = (self.capacity_j_per_k, self.conductance_w_per_k,
                         self.source_map, self.boundary_conductance_w_per_k,
                         self.observer_matrix)
        if c.ndim != 1 or not c.size or np.any(c <= 0):
            raise ValueError("capacity must be a nonempty positive vector")
        n = c.size
        if g.shape != (n, n) or b.ndim != 2 or b.shape[0] != n:
            raise ValueError("G must be n by n and B must be n by number of sources")
        if h.ndim != 2 or h.shape[0] != n or w.ndim != 2 or w.shape[1] != n:
            raise ValueError("H must be n by boundaries and W must be observations by n")
        tol = 1e-11 * max(float(np.max(np.abs(g))), np.finfo(float).tiny)
        if not np.allclose(g, g.T, rtol=1e-11, atol=tol):
            raise ValueError("G must be symmetric (reciprocal conduction)")
        off_diagonal = g - np.diag(np.diag(g))
        if np.any(off_diagonal > tol) or np.any(np.diag(g) < 0):
            raise ValueError("G must have nonpositive off-diagonals and nonnegative diagonal")
        if np.any(b < 0) or np.any(h < 0) or np.any(w < 0):
            raise ValueError("source, boundary, and observer weights must be nonnegative")
        if not np.allclose(g.sum(axis=1), h.sum(axis=1), rtol=1e-10, atol=tol * n):
            raise ValueError("G row sums must equal boundary conductance row sums")
        if not np.allclose(b.sum(axis=0), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("each source column must deposit one watt per input watt")
        if not np.allclose(w.sum(axis=1), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("each observer row must be a normalized temperature average")
        for name, count, prefix in (("node_names", n, "node"),
                                    ("source_names", b.shape[1], "source"),
                                    ("observer_names", w.shape[0], "observer")):
            names = tuple(getattr(self, name)) or tuple(f"{prefix}_{i}" for i in range(count))
            if len(names) != count or len(set(names)) != count:
                raise ValueError(f"{name} must contain {count} distinct names")
            object.__setattr__(self, name, names)

    @property
    def n_nodes(self) -> int:
        return self.capacity_j_per_k.size

    @property
    def n_sources(self) -> int:
        return self.source_map.shape[1]

    @property
    def n_observers(self) -> int:
        return self.observer_matrix.shape[0]

    @property
    def state_matrix(self) -> NDArray[np.float64]:
        return -self.conductance_w_per_k / self.capacity_j_per_k[:, None]

    @property
    def is_asymptotically_stable(self) -> bool:
        adjacency = self.conductance_w_per_k < 0
        _, labels = connected_components(csr_matrix(adjacency), directed=False)
        anchored = self.boundary_conductance_w_per_k.sum(axis=1) > 0
        return all(np.any(anchored[labels == label]) for label in np.unique(labels))

    def _temperature(self, temperature_k: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(temperature_k, dtype=float)
        if t.shape != (self.n_nodes,) or not np.all(np.isfinite(t)):
            raise ValueError("temperature_k must be a finite n-node vector")
        return t

    def _inputs(self, powers_w: ArrayLike, boundary_temperature_k: ArrayLike):
        p = np.atleast_1d(np.asarray(powers_w, dtype=float))
        h = self.boundary_conductance_w_per_k
        boundary = np.asarray(boundary_temperature_k, dtype=float)
        if boundary.ndim == 0:
            boundary = np.full(h.shape[1], float(boundary))
        if p.shape != (self.n_sources,) or not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError("powers_w must be a finite nonnegative vector of source watts")
        if boundary.shape != (h.shape[1],) or not np.all(np.isfinite(boundary)) or np.any(boundary < 0):
            raise ValueError("boundary_temperature_k must be nonnegative Kelvin per boundary")
        return p, boundary

    def derivative(self, temperature_k: ArrayLike, powers_w: ArrayLike,
                   boundary_temperature_k: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the RHS; recompute absorbed power outside at every ODE call."""
        temperature = self._temperature(temperature_k)
        powers, boundary = self._inputs(powers_w, boundary_temperature_k)
        return (-self.conductance_w_per_k @ temperature + self.source_map @ powers
                + self.boundary_conductance_w_per_k @ boundary) / self.capacity_j_per_k

    def observe(self, temperature_k: ArrayLike) -> NDArray[np.float64]:
        return self.observer_matrix @ self._temperature(temperature_k)

    def steady_state(self, powers_w: ArrayLike,
                     boundary_temperature_k: ArrayLike) -> NDArray[np.float64]:
        if not self.is_asymptotically_stable:
            raise ValueError("steady state is not unique for an unanchored thermal network")
        powers, boundary = self._inputs(powers_w, boundary_temperature_k)
        return np.linalg.solve(self.conductance_w_per_k,
                               self.source_map @ powers + self.boundary_conductance_w_per_k @ boundary)

    def step_zoh(self, temperature_k: ArrayLike, powers_w: ArrayLike,
                 boundary_temperature_k: ArrayLike, duration_s: float) -> NDArray[np.float64]:
        """Exact linear step for inputs and boundaries held constant over duration.

        Uses an augmented matrix exponential, so no inverse of G is needed and
        insulated networks are supported. Not valid for T-dependent absorption
        or changing ambient within the interval; use ``derivative`` with an ODE
        solver for those cases.
        """
        temperature = self._temperature(temperature_k)
        powers, boundary = self._inputs(powers_w, boundary_temperature_k)
        if not np.isfinite(duration_s) or duration_s < 0:
            raise ValueError("duration_s must be finite and nonnegative")
        if duration_s == 0:
            return temperature.copy()
        # Shift by a constant reference for numerical conditioning only. This
        # is not a moving ambient state and never produces a boundary jump.
        reference = float(np.mean(boundary)) if boundary.size else float(np.mean(temperature))
        forcing = (self.source_map @ powers + self.boundary_conductance_w_per_k @ boundary
                   - self.conductance_w_per_k @ np.full(self.n_nodes, reference))
        augmented = np.zeros((self.n_nodes + 1, self.n_nodes + 1))
        augmented[:-1, :-1] = self.state_matrix
        augmented[:-1, -1] = forcing / self.capacity_j_per_k
        next_state = expm(augmented * duration_s) @ np.r_[temperature - reference, 1.0]
        return next_state[:-1] + reference

    def energy_rate_w(self, temperature_k: ArrayLike, powers_w: ArrayLike,
                      boundary_temperature_k: ArrayLike) -> float:
        """Net energy rate in all finite-capacity nodes (boundary flux included)."""
        return float(self.capacity_j_per_k @ self.derivative(
            temperature_k, powers_w, boundary_temperature_k))


@dataclass(frozen=True)
class AssumedThermalParameters:
    """Illustrative constants only; none are measured Lightmatter properties."""

    waveguide_capacity_j_per_k: float = 1.2e-10
    heater_capacity_j_per_k: float = 3.0e-10
    package_capacity_j_per_k: float = 4.0e-8
    heater_waveguide_conductance_w_per_k: float = 3.0e-5
    waveguide_package_conductance_w_per_k: float = 2.0e-5
    heater_package_conductance_w_per_k: float = 1.0e-5
    neighbor_conductance_w_per_k: float = 1.0e-6
    package_boundary_conductance_w_per_k: float = 2.0e-3
    aggressor_package_fraction: float = 0.8

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "aggressor_package_fraction":
                if not 0 <= value <= 1:
                    raise ValueError("aggressor_package_fraction must lie in [0, 1]")
            elif name == "neighbor_conductance_w_per_k":
                if value < 0:
                    raise ValueError("neighbor conductance must be nonnegative")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")


def default_ring_network(n_rings: int = 1,
                         parameters: AssumedThermalParameters | None = None) -> ThermalNetwork:
    """Return an explicitly assumed two-local-node-per-ring + package network.

    Sources are ordered heaters, optical absorption, adjacent aggressors (each
    block has n_rings entries). Observations are waveguide temperatures.
    Package coupling and nearest-neighbor conduction create thermal crosstalk.
    The aggressor split is a surrogate for an unknown heat-source placement.
    """
    if not isinstance(n_rings, int) or isinstance(n_rings, bool) or n_rings < 1:
        raise ValueError("n_rings must be a positive integer")
    p = parameters or AssumedThermalParameters()
    n = 2 * n_rings + 1
    package = n - 1
    capacity = np.r_[np.full(n_rings, p.waveguide_capacity_j_per_k),
                     np.full(n_rings, p.heater_capacity_j_per_k), p.package_capacity_j_per_k]
    g = np.zeros((n, n))

    def connect(i, j, conductance):
        g[i, i] += conductance
        g[j, j] += conductance
        g[i, j] -= conductance
        g[j, i] -= conductance

    for i in range(n_rings):
        connect(i, n_rings + i, p.heater_waveguide_conductance_w_per_k)
        connect(i, package, p.waveguide_package_conductance_w_per_k)
        connect(n_rings + i, package, p.heater_package_conductance_w_per_k)
        if i:
            connect(i - 1, i, p.neighbor_conductance_w_per_k)
    h = np.zeros((n, 1))
    h[package, 0] = p.package_boundary_conductance_w_per_k
    g[package, package] += h[package, 0]
    b = np.zeros((n, 3 * n_rings))
    w = np.zeros((n_rings, n))
    for i in range(n_rings):
        b[n_rings + i, i] = 1.0
        b[i, n_rings + i] = 1.0
        b[package, 2 * n_rings + i] = p.aggressor_package_fraction
        b[n_rings + i, 2 * n_rings + i] = 1.0 - p.aggressor_package_fraction
        w[i, i] = 1.0
    return ThermalNetwork(
        capacity, g, b, h, w,
        tuple(f"waveguide_{i}" for i in range(n_rings)) +
        tuple(f"heater_{i}" for i in range(n_rings)) + ("package",),
        tuple(f"{kind}_{i}" for kind in ("heater", "absorption", "aggressor") for i in range(n_rings)),
        tuple(f"ring_{i}" for i in range(n_rings)),
    )
