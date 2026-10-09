"""Sparse, conservative finite-volume extraction for an assumed layered slab.

This module assembles physical cell capacities and face conductances; SciPy
provides sparse factorization, integration, and linear algebra.  It is a 2-D
cross-section extruded through a specified depth, not a reconstructed foundry
MRM.  Region/cell overlap integrals avoid changing total power with mesh size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, qr
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh, splu, spsolve

from ringlab.thermal import ThermalNetwork

@dataclass(frozen=True)
class Layer:
    name: str
    thickness_m: float
    conductivity_w_per_m_k: float
    volumetric_heat_capacity_j_per_m3_k: float

    def __post_init__(self):
        values = (self.thickness_m, self.conductivity_w_per_m_k,
                  self.volumetric_heat_capacity_j_per_m3_k)
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError("layer thickness, conductivity, and heat capacity must be positive")


@dataclass(frozen=True)
class Region:
    name: str
    x_min_m: float
    x_max_m: float
    z_min_m: float
    z_max_m: float

    def __post_init__(self):
        if not np.all(np.isfinite((self.x_min_m, self.x_max_m, self.z_min_m, self.z_max_m))):
            raise ValueError("region coordinates must be finite")
        if self.x_max_m <= self.x_min_m or self.z_max_m <= self.z_min_m:
            raise ValueError("regions must have positive area")


@dataclass(frozen=True)
class SlabGeometry:
    width_m: float
    extrusion_depth_m: float
    layers: tuple[Layer, ...]  # Bottom to top.
    heat_sources: tuple[Region, ...]
    temperature_observers: tuple[Region, ...]
    provenance: str = "user supplied; geometry accuracy has not been established"

    def __post_init__(self):
        if (not np.isfinite(self.width_m) or not np.isfinite(self.extrusion_depth_m)
                or self.width_m <= 0 or self.extrusion_depth_m <= 0 or not self.layers):
            raise ValueError("positive slab dimensions and at least one layer required")
        height = sum(layer.thickness_m for layer in self.layers)
        for region in self.heat_sources + self.temperature_observers:
            if (region.x_min_m < -self.width_m / 2 or region.x_max_m > self.width_m / 2
                    or region.z_min_m < 0 or region.z_max_m > height * (1 + 1e-12)):
                raise ValueError(f"region {region.name} must be wholly inside the slab")
        for regions in (self.heat_sources, self.temperature_observers):
            if len({region.name for region in regions}) != len(regions):
                raise ValueError("source/observer names must be unique within each collection")


@dataclass
class ModalThermalModel:
    """Projected diffusion modes for a fixed uniform boundary temperature.

    Modal coordinates are not nodal Kelvin temperatures. All retained poles
    are stable and their quadratic thermal energy decays without heating, but
    truncation does not guarantee positive local reconstructed temperatures.
    There is no artificial instantaneous/DC feedthrough. Always quantify the
    omitted-mode error over the time/frequency band used by the controller.
    """

    decay_rates_per_s: NDArray[np.float64]
    input_matrix: NDArray[np.float64]
    observation_matrix: NDArray[np.float64]
    dc_gain_relative_error: float
    reduction_method: str = "slow diffusion eigenmodes"

    @property
    def n_modes(self):
        return self.decay_rates_per_s.size

    def dc_gain_k_per_w(self):
        return self.observation_matrix @ (self.input_matrix / self.decay_rates_per_s[:, None])

    def step_response(self, times_s, powers_w, boundary_temperature_k=300.0):
        times = np.asarray(times_s, dtype=float)
        powers = np.asarray(powers_w, dtype=float)
        if (times.ndim != 1 or np.any(times < 0) or not np.all(np.isfinite(times))
                or np.any(np.diff(times) < 0)):
            raise ValueError("times must be finite, nonnegative, and nondecreasing")
        if (powers.shape != (self.input_matrix.shape[1],) or np.any(powers < 0)
                or not np.all(np.isfinite(powers))):
            raise ValueError("one finite nonnegative power is needed per source")
        if not np.isfinite(boundary_temperature_k) or boundary_temperature_k < 0:
            raise ValueError("boundary temperature must be nonnegative Kelvin")
        modal_equilibrium = (self.input_matrix @ powers) / self.decay_rates_per_s
        modal_rise = -np.expm1(-times[:, None] * self.decay_rates_per_s[None, :]) * modal_equilibrium
        return modal_rise @ self.observation_matrix.T + boundary_temperature_k


@dataclass
class SpatialThermalModel:
    capacity_j_per_k: NDArray[np.float64]
    conductance_w_per_k: csr_matrix
    source_map: NDArray[np.float64]
    boundary_conductance_w_per_k: NDArray[np.float64]
    observer_matrix: NDArray[np.float64]
    x_edges_m: NDArray[np.float64]
    z_edges_m: NDArray[np.float64]
    z_layer_indices: NDArray[np.int64]
    geometry: SlabGeometry

    @property
    def n_nodes(self):
        return self.capacity_j_per_k.size

    @property
    def state_matrix(self):
        return -diags(1.0 / self.capacity_j_per_k) @ self.conductance_w_per_k

    def _powers(self, powers_w):
        powers = np.asarray(powers_w, dtype=float)
        if (powers.shape != (self.source_map.shape[1],) or np.any(powers < 0)
                or not np.all(np.isfinite(powers))):
            raise ValueError("one finite nonnegative watt value per heat source is required")
        return powers

    def steady_state(self, powers_w, boundary_temperature_k=300.0):
        """Bottom surface is isothermal; top and side surfaces are adiabatic."""
        powers = self._powers(powers_w)
        if not np.isfinite(boundary_temperature_k) or boundary_temperature_k < 0:
            raise ValueError("boundary temperature must be nonnegative Kelvin")
        return spsolve(self.conductance_w_per_k, self.source_map @ powers) + boundary_temperature_k

    def dc_gain_k_per_w(self):
        """Output temperature rise per source watt (includes thermal crosstalk)."""
        return self.observer_matrix @ spsolve(self.conductance_w_per_k, self.source_map)

    def to_network(self, max_nodes: int = 512) -> ThermalNetwork:
        """Export an unreduced, physical-temperature network for a small mesh.

        ThermalNetwork uses dense matrices for compact control integration.
        This explicit guard prevents accidentally densifying a fine mesh;
        use the sparse spatial methods for larger problems.
        """
        if self.n_nodes > max_nodes or max_nodes > 2048:
            raise ValueError("dense export exceeds max_nodes (hard limit 2048); retain sparse model")
        return ThermalNetwork(
            self.capacity_j_per_k, self.conductance_w_per_k.toarray(), self.source_map,
            self.boundary_conductance_w_per_k, self.observer_matrix,
            source_names=tuple(region.name for region in self.geometry.heat_sources),
            observer_names=tuple(region.name for region in self.geometry.temperature_observers),
        )

    def step_response(self, times_s, powers_w, boundary_temperature_k=300.0,
                      rtol=1e-7, atol_k=1e-8):
        """Integrate a power step from uniform boundary temperature using SciPy BDF.

        The sparse Jacobian avoids dense storage and is especially useful with
        thin film cells. Returned array has shape (time, observer).
        """
        times = np.asarray(times_s, dtype=float)
        powers = self._powers(powers_w)
        if times.ndim != 1 or times.size < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError("times_s must strictly increase from zero with at least two samples")
        if not np.isfinite(boundary_temperature_k) or boundary_temperature_k < 0:
            raise ValueError("boundary temperature must be nonnegative Kelvin")
        a = self.state_matrix
        drive = (self.source_map @ powers) / self.capacity_j_per_k
        solved = solve_ivp(lambda _t, rise: a @ rise + drive,
                           (0.0, times[-1]), np.zeros(self.n_nodes), method="BDF",
                           t_eval=times, jac=a, rtol=rtol, atol=atol_k)
        if not solved.success:
            raise RuntimeError(f"spatial heat integration failed: {solved.message}")
        return (self.observer_matrix @ solved.y).T + boundary_temperature_k

    def modal_reduce(self, n_modes: int = 32) -> ModalThermalModel:
        """Keep slow generalized diffusion modes using SciPy sparse shift-invert.

        Solves G V = C V Lambda with V.T C V = I. The returned dynamics are
        zdot=-Lambda z + V.T B p, y=W V z, for temperature rise above a fixed
        uniform boundary. It is not a physical-temperature ThermalNetwork and
        must not be substituted for one without validating truncation error.
        """
        if not isinstance(n_modes, int) or not 1 <= n_modes < self.n_nodes - 1:
            raise ValueError("n_modes must lie between 1 and n_nodes-2")
        if n_modes > 256 or self.n_nodes * n_modes > 20_000_000:
            raise ValueError("modal reduction exceeds the conservative memory guard")
        root_inverse_c = 1.0 / np.sqrt(self.capacity_j_per_k)
        scaling = diags(root_inverse_c)
        symmetric = scaling @ self.conductance_w_per_k @ scaling
        rates, vectors = eigsh(symmetric, k=n_modes, sigma=0.0, which="LM", tol=1e-10,
                               v0=np.ones(self.n_nodes))
        order = np.argsort(rates)
        rates = rates[order]
        if np.any(rates <= 0):
            raise RuntimeError("spatial diffusion produced a nonpositive decay rate")
        physical_modes = root_inverse_c[:, None] * vectors[:, order]
        inputs = physical_modes.T @ self.source_map
        observations = self.observer_matrix @ physical_modes
        full_gain = self.dc_gain_k_per_w()
        gain = observations @ (inputs / rates[:, None])
        relative_error = float(np.linalg.norm(gain - full_gain) / np.linalg.norm(full_gain))
        return ModalThermalModel(rates, inputs, observations, relative_error)

    def krylov_reduce(self, n_moments: int = 4) -> ModalThermalModel:
        """Match low-frequency input moments by a symmetric Krylov projection.

        The basis spans S^-1 F, S^-2 F, ... for S=C^-1/2 G C^-1/2 and
        F=C^-1/2 B. QR deflates dependent input directions. Symmetric Galerkin
        projection retains positive decay rates, and the first block captures
        every source's DC solution. SciPy supplies LU, QR, and eigensolvers.
        This is a compact RC specialization of projection/moment matching,
        not a full implementation of the RLC PRIMA algorithm.
        """
        if not isinstance(n_moments, int) or not 1 <= n_moments <= 32:
            raise ValueError("n_moments must lie between 1 and 32")
        max_rank = min(self.n_nodes, n_moments * self.source_map.shape[1])
        if self.n_nodes * max_rank > 20_000_000:
            raise ValueError("Krylov basis exceeds the conservative memory guard")
        root_inverse_c = 1.0 / np.sqrt(self.capacity_j_per_k)
        scaling = diags(root_inverse_c)
        symmetric = scaling @ self.conductance_w_per_k @ scaling
        factor = splu(symmetric.tocsc())
        block = factor.solve(root_inverse_c[:, None] * self.source_map)
        basis = np.empty((self.n_nodes, 0))
        for _ in range(n_moments):
            residual = block.copy()
            # Reorthogonalize to avoid duplicating slow directions over many
            # inverse-operator applications.
            for _pass in range(2):
                residual -= basis @ (basis.T @ residual)
            orthogonal, triangular, _ = qr(residual, mode="economic", pivoting=True)
            diagonal = np.abs(np.diag(triangular))
            if not diagonal.size or diagonal[0] <= np.finfo(float).tiny:
                break
            rank = int(np.count_nonzero(diagonal > diagonal[0] * 1e-10))
            rank = min(rank, self.n_nodes - basis.shape[1])
            if rank == 0:
                break
            new_block = orthogonal[:, :rank]
            basis = np.column_stack((basis, new_block))
            if basis.shape[1] == self.n_nodes:
                break
            block = factor.solve(new_block)
        projected = basis.T @ (symmetric @ basis)
        rates, rotations = eigh((projected + projected.T) / 2)
        if np.any(rates <= 0):
            raise RuntimeError("projected diffusion has nonpositive decay rates")
        physical_modes = root_inverse_c[:, None] * (basis @ rotations)
        inputs = physical_modes.T @ self.source_map
        observations = self.observer_matrix @ physical_modes
        full_gain = self.dc_gain_k_per_w()
        gain = observations @ (inputs / rates[:, None])
        error = float(np.linalg.norm(gain - full_gain) / np.linalg.norm(full_gain))
        return ModalThermalModel(rates, inputs, observations, error, "symmetric block Krylov moments at DC")


def assemble_slab(geometry: SlabGeometry, nx: int = 64,
                  cells_per_layer: tuple[int, ...] = (6, 4, 1, 3)) -> SpatialThermalModel:
    """Assemble a layer-conforming rectangular finite-volume mesh.

    Face conductance uses series thermal resistances from each cell center to
    its face, which is the harmonic face rule at material interfaces.
    Bottom Dirichlet temperature is connected across the bottom half-cell.
    """
    if not isinstance(nx, int) or nx < 1:
        raise ValueError("nx must be a positive integer")
    if len(cells_per_layer) != len(geometry.layers) or any(
            not isinstance(n, int) or n < 1 for n in cells_per_layer):
        raise ValueError("one positive cell count is needed for each layer")
    if nx * sum(cells_per_layer) > 100_000:
        raise ValueError("mesh exceeds the conservative 100,000-cell memory guard")
    x_edges = np.linspace(-geometry.width_m / 2, geometry.width_m / 2, nx + 1)
    z_edges = [0.0]
    layer_ids = []
    for layer_index, (layer, count) in enumerate(zip(geometry.layers, cells_per_layer)):
        z_edges.extend(np.linspace(z_edges[-1], z_edges[-1] + layer.thickness_m, count + 1)[1:])
        layer_ids.extend([layer_index] * count)
    z_edges = np.asarray(z_edges)
    layer_ids = np.asarray(layer_ids, dtype=int)
    dx = np.diff(x_edges)
    dz = np.diff(z_edges)
    nz = dz.size
    n = nx * nz
    depth = geometry.extrusion_depth_m
    conductivity = np.asarray([layer.conductivity_w_per_m_k for layer in geometry.layers])[layer_ids]
    cv = np.asarray([layer.volumetric_heat_capacity_j_per_m3_k for layer in geometry.layers])[layer_ids]
    capacity = (cv[:, None] * dz[:, None] * dx[None, :] * depth).ravel()
    rows, columns, values = [], [], []
    diagonal = np.zeros(n)

    def connect(i, j, conductance):
        rows.extend((i, j))
        columns.extend((j, i))
        values.extend((-conductance, -conductance))
        diagonal[i] += conductance
        diagonal[j] += conductance

    for z in range(nz):
        for x in range(nx):
            i = z * nx + x
            if x + 1 < nx:
                conductance = conductivity[z] * dz[z] * depth / ((dx[x] + dx[x + 1]) / 2)
                connect(i, i + 1, conductance)
            if z + 1 < nz:
                resistance = dz[z] / (2 * conductivity[z]) + dz[z + 1] / (2 * conductivity[z + 1])
                connect(i, i + nx, dx[x] * depth / resistance)
    h = np.zeros((n, 1))
    h[:nx, 0] = conductivity[0] * dx * depth / (dz[0] / 2)
    diagonal += h[:, 0]
    rows.extend(range(n))
    columns.extend(range(n))
    values.extend(diagonal)
    g = coo_matrix((values, (rows, columns)), shape=(n, n)).tocsr()

    def region_weights(region):
        x_overlap = np.maximum(0.0, np.minimum(x_edges[1:], region.x_max_m)
                               - np.maximum(x_edges[:-1], region.x_min_m))
        z_overlap = np.maximum(0.0, np.minimum(z_edges[1:], region.z_max_m)
                               - np.maximum(z_edges[:-1], region.z_min_m))
        weights = (z_overlap[:, None] * x_overlap[None, :]).ravel()
        if weights.sum() <= 0:
            raise ValueError(f"region {region.name} has no mesh overlap")
        return weights / weights.sum()

    b = np.column_stack([region_weights(region) for region in geometry.heat_sources])
    w = np.vstack([region_weights(region) for region in geometry.temperature_observers])
    return SpatialThermalModel(capacity, g, b, h, w, x_edges, z_edges, layer_ids, geometry)


def assumed_slab_geometry(n_rings: int = 2) -> SlabGeometry:
    """A declared surrogate cross-section, with assumed room-temperature values.

    The device layer is a continuous film, and the sources are rectangular;
    neither a full curved ring nor metal/via/package geometry is represented.
    Conductivities are assumptions, especially the reduced thin-film silicon
    value. These are not material data claimed for a Lightmatter process.
    """
    if not isinstance(n_rings, int) or isinstance(n_rings, bool) or not 1 <= n_rings <= 8:
        raise ValueError("n_rings must lie between 1 and 8 for this assumed geometry")
    layers = (Layer("silicon substrate", 20e-6, 130.0, 1.63e6),
              Layer("buried oxide", 2e-6, 1.4, 1.55e6),
              Layer("continuous silicon film surrogate", 0.22e-6, 80.0, 1.63e6),
              Layer("oxide overcladding", 1.5e-6, 1.4, 1.55e6))
    width = max(80e-6, (n_rings - 1) * 30e-6 + 50e-6)
    centers = (np.arange(n_rings) - (n_rings - 1) / 2) * 30e-6
    silicon_bottom = layers[0].thickness_m + layers[1].thickness_m
    silicon_top = silicon_bottom + layers[2].thickness_m
    top = silicon_top + layers[3].thickness_m
    optical = tuple(Region(f"absorption_{i}", x - 0.3e-6, x + 0.3e-6,
                           silicon_bottom, silicon_top) for i, x in enumerate(centers))
    heater = tuple(Region(f"heater_{i}", x - 2e-6, x + 2e-6,
                          top - 0.4e-6, top) for i, x in enumerate(centers))
    aggressor = tuple(Region(f"aggressor_{i}", x + 5e-6, x + 9e-6,
                             top - 0.4e-6, top) for i, x in enumerate(centers))
    observers = tuple(Region(f"ring_{i}", r.x_min_m, r.x_max_m, r.z_min_m, r.z_max_m)
                      for i, r in enumerate(optical))
    return SlabGeometry(width, 20e-6, layers, heater + optical + aggressor, observers,
                        "explicit assumed SOI-like slab; no foundry geometry or calibration")


def spatial_convergence(levels=(1, 2, 4), base_nx=32,
                         base_cells_per_layer=(3, 2, 1, 2), n_rings=2):
    """Return reproducible refinement evidence, with no empirical target data.

    Errors are differences to the finest requested grid, not a certificate of
    physical accuracy or an estimate of geometry/material uncertainty.
    """
    if len(levels) < 2 or any(not isinstance(v, int) or v < 1 for v in levels):
        raise ValueError("at least two positive integer refinement levels are needed")
    if any(b <= a for a, b in zip(levels, levels[1:])):
        raise ValueError("refinement levels must strictly increase")
    rows = []
    for level in levels:
        model = assemble_slab(assumed_slab_geometry(n_rings), base_nx * level,
                              tuple(n * level for n in base_cells_per_layer))
        gain = model.dc_gain_k_per_w()
        rows.append({"refinement": level, "cells": model.n_nodes,
                     "gain_k_per_w": gain.tolist(),
                     "power_balance_relative_error": float(np.max(np.abs(
                         model.boundary_conductance_w_per_k.T @
                         spsolve(model.conductance_w_per_k, model.source_map) - 1.0)))})
    reference = np.asarray(rows[-1]["gain_k_per_w"])
    for row in rows:
        row["relative_gain_difference_to_finest"] = float(
            np.linalg.norm(np.asarray(row["gain_k_per_w"]) - reference) / np.linalg.norm(reference))
    return {"geometry_status": "assumed, not a foundry device", "levels": rows,
            "reference_is_exact": False}
