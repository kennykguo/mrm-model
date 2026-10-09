"""Passive heat-flow networks and generalized thermal reduced-order models.

The state is a temperature rise relative to a *fixed* reference, not relative
to the instantaneous ambient.  Consequently a boundary-temperature step does
not instantaneously change stored temperatures.  The governing equation is

    C xdot = -G x + B p + E (T_boundary - T_reference) + q.

Symmetric positive-definite C and G give a passive, stable heat subsystem.
Full matrices permit a congruence-reduced FEM model as well as a nodal RC graph.
Illustrative defaults are deliberately not Lightmatter device calibration.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import cholesky, eigh, expm, solve


def _finite_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def _spd(matrix: np.ndarray, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-15):
        raise ValueError(f"{name} must be symmetric")
    try:
        cholesky(matrix, lower=True)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error


@dataclass
class ThermalParams:
    """Physical SI-unit coefficients; matrices are validated on construction.

    ``capacitance_j_per_k`` is a positive nodal vector or a full SPD matrix.
    ``heater_map`` distributes heater watts into state heat loads.  ``ring_map``
    maps states into ring-temperature rises.  Boundary columns have W/K units.
    Physical nodal networks must have nonpositive G off-diagonals, nonnegative
    deposition/observation maps, and boundary conductances equal to G row sums.

    For an arbitrary passive ROM, set ``physical_network=False`` and provide
    ``reference_state_map``: coordinates of a spatially uniform temperature.
    Signed modal maps are then allowed; passivity is still checked via C and G.
    Such models have a generalized energy state, not nodal temperature states.
    """

    capacitance_j_per_k: Any
    conductance_w_per_k: Any
    heater_map: Any
    ring_map: Any
    boundary_map_w_per_k: Any | None = None
    reference_temperature_k: float = 300.0
    heater_temperature_map: Any | None = None
    reference_state_map: Any | None = None
    physical_network: bool = True
    provenance: str = "user-specified; calibration status unspecified"

    def __post_init__(self) -> None:
        self.capacitance_j_per_k = _finite_array(self.capacitance_j_per_k, "capacitance_j_per_k")
        c = self.capacitance_j_per_k
        if c.ndim == 1:
            if np.any(c <= 0):
                raise ValueError("nodal heat capacities must be positive")
            c_matrix = np.diag(c)
        else:
            c_matrix = c
        _spd(c_matrix, "thermal capacitance")
        self.conductance_w_per_k = _finite_array(self.conductance_w_per_k, "conductance_w_per_k")
        g = self.conductance_w_per_k
        _spd(g, "thermal conductance")
        n = g.shape[0]
        if c_matrix.shape != (n, n):
            raise ValueError("C and G must have matching state dimensions")
        self.heater_map = _finite_array(self.heater_map, "heater_map")
        self.ring_map = _finite_array(self.ring_map, "ring_map")
        if self.heater_map.ndim != 2 or self.heater_map.shape[0] != n:
            raise ValueError("heater_map must have shape (states, heaters)")
        if self.ring_map.ndim != 2 or self.ring_map.shape[1] != n:
            raise ValueError("ring_map must have shape (rings, states)")
        if self.reference_state_map is None:
            self.reference_state_map = np.ones(n)
        else:
            self.reference_state_map = _finite_array(self.reference_state_map, "reference_state_map")
        s = self.reference_state_map
        if s.shape != (n,):
            raise ValueError("reference_state_map must have one value per state")
        if self.boundary_map_w_per_k is None:
            self.boundary_map_w_per_k = (g @ s)[:, None]
        else:
            self.boundary_map_w_per_k = _finite_array(self.boundary_map_w_per_k, "boundary_map_w_per_k")
        e = self.boundary_map_w_per_k
        if e.ndim == 1:
            e = self.boundary_map_w_per_k = e[:, None]
        if e.ndim != 2 or e.shape[0] != n or e.shape[1] < 1:
            raise ValueError("boundary map must have shape (states, boundaries)")
        scale = max(np.linalg.norm(g @ s), 1e-15)
        if np.linalg.norm(e.sum(axis=1) - g @ s) > 1e-9 * scale:
            raise ValueError("boundary map must preserve uniform-temperature equilibrium")
        if not np.allclose(self.ring_map @ s, 1.0, rtol=1e-9, atol=1e-10):
            raise ValueError("ring_map must preserve uniform temperatures")
        if self.heater_temperature_map is None:
            if self.ring_map.shape[0] != self.heater_map.shape[1]:
                raise ValueError("supply heater_temperature_map when heater/ring counts differ")
            self.heater_temperature_map = self.ring_map.copy()
        else:
            self.heater_temperature_map = _finite_array(self.heater_temperature_map, "heater_temperature_map")
        ht = self.heater_temperature_map
        if ht.shape != (self.heater_map.shape[1], n):
            raise ValueError("heater temperature map must have shape (heaters, states)")
        if not np.allclose(ht @ s, 1.0, rtol=1e-9, atol=1e-10):
            raise ValueError("heater temperature map must preserve uniform temperatures")
        if not np.isfinite(self.reference_temperature_k) or self.reference_temperature_k <= 0:
            raise ValueError("reference_temperature_k must be positive and finite")
        if self.physical_network:
            off_diagonal = g - np.diag(np.diag(g))
            tolerance = 1e-12 * max(float(np.max(np.abs(g))), 1e-15)
            if np.any(off_diagonal > tolerance) or np.any(e < -tolerance):
                raise ValueError("physical G must be a reciprocal passive conductance graph")
            if np.any(self.heater_map < 0) or np.any(self.ring_map < 0) or np.any(ht < 0):
                raise ValueError("physical heat and temperature maps must be nonnegative")
            if not np.allclose(s, 1):
                raise ValueError("physical nodal states require reference_state_map=ones")
            if np.any(self.heater_map.sum(axis=0) > 1 + 1e-10):
                raise ValueError("heater map may not create energy")

    @property
    def capacitance_matrix(self) -> np.ndarray:
        c = self.capacitance_j_per_k
        return np.diag(c) if c.ndim == 1 else c.copy()

    @property
    def n_states(self) -> int:
        return self.conductance_w_per_k.shape[0]

    @property
    def n_rings(self) -> int:
        return self.ring_map.shape[0]

    @property
    def n_heaters(self) -> int:
        return self.heater_map.shape[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacitance_j_per_k": self.capacitance_j_per_k.tolist(),
            "conductance_w_per_k": self.conductance_w_per_k.tolist(),
            "heater_map": self.heater_map.tolist(),
            "ring_map": self.ring_map.tolist(),
            "boundary_map_w_per_k": self.boundary_map_w_per_k.tolist(),
            "reference_temperature_k": self.reference_temperature_k,
            "heater_temperature_map": self.heater_temperature_map.tolist(),
            "reference_state_map": self.reference_state_map.tolist(),
            "physical_network": self.physical_network,
            "provenance": self.provenance,
        }


class ThermalNetwork:
    """Exact zero-order-hold dynamics of the *full* coupled heat network."""

    def __init__(self, params: ThermalParams, initial_temperature_k: Any | None = None):
        self.params = params
        self.p = params
        self.time_s = 0.0
        self._c = params.capacitance_matrix
        self._g = params.conductance_w_per_k
        self._a = -solve(self._c, self._g, assume_a="pos")
        self._transitions: OrderedDict[float, np.ndarray] = OrderedDict()
        self.state_rise_k = np.zeros(params.n_states)
        if initial_temperature_k is not None:
            initial = np.asarray(initial_temperature_k, dtype=float)
            if initial.ndim == 0:
                self.state_rise_k = params.reference_state_map * (
                    float(initial) - params.reference_temperature_k
                )
            elif initial.shape == (params.n_states,) and params.physical_network:
                self.state_rise_k = initial - params.reference_temperature_k
            else:
                raise ValueError("initial temperature must be scalar or nodal state vector")
            if np.any(initial <= 0) or not np.all(np.isfinite(initial)):
                raise ValueError("initial absolute temperature must be positive and finite")

    @property
    def state_temperature_k(self) -> np.ndarray:
        """Nodal temperatures, or generalized absolute coordinates for a ROM."""
        return self.state_rise_k + self.p.reference_state_map * self.p.reference_temperature_k

    @property
    def ring_temperatures_k(self) -> np.ndarray:
        return self.p.reference_temperature_k + self.p.ring_map @ self.state_rise_k

    @property
    def heater_temperatures_k(self) -> np.ndarray:
        return self.p.reference_temperature_k + self.p.heater_temperature_map @ self.state_rise_k

    def _forcing(self, heater_power_w: Any, ambient_temperature_k: Any, external_heat_w: Any) -> np.ndarray:
        power = np.asarray(heater_power_w, dtype=float)
        if power.ndim == 0 and self.p.n_heaters == 1:
            power = power.reshape(1)
        if power.shape != (self.p.n_heaters,) or np.any(power < 0) or not np.all(np.isfinite(power)):
            raise ValueError("heater power must be a finite nonnegative vector per heater")
        ambient = np.asarray(ambient_temperature_k, dtype=float)
        if ambient.ndim == 0:
            ambient = np.full(self.p.boundary_map_w_per_k.shape[1], float(ambient))
        if ambient.shape != (self.p.boundary_map_w_per_k.shape[1],):
            raise ValueError("ambient temperature must be scalar or one value per boundary")
        if np.any(ambient <= 0) or not np.all(np.isfinite(ambient)):
            raise ValueError("boundary absolute temperatures must be positive and finite")
        forcing = self.p.heater_map @ power + self.p.boundary_map_w_per_k @ (
            ambient - self.p.reference_temperature_k
        )
        if external_heat_w is not None:
            heat = np.asarray(external_heat_w, dtype=float)
            if heat.shape != (self.p.n_states,) or not np.all(np.isfinite(heat)):
                raise ValueError("external heat must have one finite watt value per state")
            forcing = forcing + heat
        return forcing

    def _transition(self, dt_s: float) -> np.ndarray:
        if dt_s not in self._transitions:
            if len(self._transitions) >= 8:
                self._transitions.popitem(last=False)
            self._transitions[dt_s] = expm(self._a * dt_s)
        self._transitions.move_to_end(dt_s)
        return self._transitions[dt_s]

    def step(
        self, dt_s: float, heater_power_w: Any, ambient_temperature_k: Any, external_heat_w: Any | None = None
    ) -> np.ndarray:
        if not np.isfinite(dt_s) or dt_s < 0:
            raise ValueError("dt_s must be finite and nonnegative")
        forcing = self._forcing(heater_power_w, ambient_temperature_k, external_heat_w)
        if dt_s:
            equilibrium_rise = solve(self._g, forcing, assume_a="pos")
            self.state_rise_k = equilibrium_rise + self._transition(float(dt_s)) @ (
                self.state_rise_k - equilibrium_rise
            )
            self.time_s += dt_s
        return self.ring_temperatures_k.copy()

    def equilibrium(
        self, heater_power_w: Any, ambient_temperature_k: Any, external_heat_w: Any | None = None
    ) -> np.ndarray:
        rise = self.equilibrium_state_rise(heater_power_w, ambient_temperature_k, external_heat_w)
        return self.p.reference_temperature_k + self.p.ring_map @ rise

    def equilibrium_state_rise(
        self, heater_power_w: Any, ambient_temperature_k: Any, external_heat_w: Any | None = None
    ) -> np.ndarray:
        """Return full fixed-reference equilibrium coordinates, including for ROMs."""
        return solve(
            self._g, self._forcing(heater_power_w, ambient_temperature_k, external_heat_w), assume_a="pos"
        )

    def equilibrium_state_temperature(
        self, heater_power_w: Any, ambient_temperature_k: Any, external_heat_w: Any | None = None
    ) -> np.ndarray:
        """Nodal equilibrium temperatures or generalized absolute coordinates."""
        return self.p.reference_temperature_k * self.p.reference_state_map + self.equilibrium_state_rise(
            heater_power_w, ambient_temperature_k, external_heat_w
        )

    def state_space(self) -> dict[str, np.ndarray | float | str]:
        """Continuous rise-state model: xdot=A x+B p+E(Tamb-Tref)+Q q."""
        return {
            "A": self._a.copy(),
            "B": solve(self._c, self.p.heater_map, assume_a="pos"),
            "E": solve(self._c, self.p.boundary_map_w_per_k, assume_a="pos"),
            "Q": solve(self._c, np.eye(self.p.n_states), assume_a="pos"),
            "H": self.p.ring_map.copy(),
            "reference_temperature_k": self.p.reference_temperature_k,
            "state_convention": "rise relative to fixed reference; output=Tref+H*x",
        }

    def modes(self) -> dict[str, np.ndarray]:
        rates, vectors = eigh(self._g, self._c)
        return {"decay_rates_per_s": rates, "time_constants_s": 1 / rates, "mode_vectors": vectors}

    def stored_energy_metric(self) -> float:
        """Lyapunov storage 0.5*x.T*C*x [J K], not thermodynamic heat [J]."""
        return float(0.5 * self.state_rise_k @ self._c @ self.state_rise_k)


def default_thermal_params(n_rings: int = 1) -> ThermalParams:
    """Illustrative reciprocal ring/heater/substrate network with multiple modes.

    Ring capacity 2 nJ/K, heater capacity 0.5 nJ/K, substrate capacity 2 uJ/K.
    Ring-heater conductance 0.2 mW/K, ring-substrate 0.05 mW/K,
    heater-substrate 0.02 mW/K, nearest ring-ring 0.005 mW/K,
    substrate-boundary 2 mW/K.  These values provide plausible microsecond
    local modes and a slower common mode; none is a measured device number.
    """
    if isinstance(n_rings, bool) or int(n_rings) != n_rings or n_rings < 1:
        raise ValueError("n_rings must be a positive integer")
    n_rings = int(n_rings)
    n_states = 2 * n_rings + 1
    substrate = n_states - 1
    capacity = np.r_[np.full(n_rings, 2e-9), np.full(n_rings, 0.5e-9), 2e-6]
    g = np.zeros((n_states, n_states))

    def connect(i: int, j: int, conductance: float) -> None:
        g[i, i] += conductance
        g[j, j] += conductance
        g[i, j] -= conductance
        g[j, i] -= conductance

    for ring in range(n_rings):
        connect(ring, n_rings + ring, 2e-4)
        connect(ring, substrate, 5e-5)
        connect(n_rings + ring, substrate, 2e-5)
        if ring:
            connect(ring - 1, ring, 5e-6)
    g[substrate, substrate] += 2e-3
    b = np.zeros((n_states, n_rings))
    b[n_rings : 2 * n_rings] = np.eye(n_rings)
    h = np.zeros((n_rings, n_states))
    h[:, :n_rings] = np.eye(n_rings)
    ht = np.zeros((n_rings, n_states))
    ht[:, n_rings : 2 * n_rings] = np.eye(n_rings)
    return ThermalParams(
        capacity,
        g,
        b,
        h,
        heater_temperature_map=ht,
        provenance="illustrative passive RC network; NOT calibrated to Lightmatter",
    )
