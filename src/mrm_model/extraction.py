"""Optional, sparse finite-element thermal extraction for a generic SOI section.

This is an open-source reference workflow, NOT a reconstruction of Lightmatter
geometry.  P1 finite-element weak forms follow scikit-fem's heat-equation
examples: https://scikit-fem.readthedocs.io/en/stable/listofexamples.html .

The 2-D section is invariant over ``modeled_length_m`` in the third direction.
Power is total watts over that length, not watts/metre or a volumetric source.
All assembled mass/stiffness/load matrices include that length consistently.
The bottom has a finite Robin thermal contact to ambient; other boundaries are
insulated.  Keeping the boundary states avoids the missing boundary-temperature
derivative term that would arise from eliminating time-varying Dirichlet nodes
while retaining a consistent capacitance matrix.
Material constants are explicitly illustrative room-temperature bulk values;
thin-film/interface resistance, contacts, packaging, convection, 3-D curvature,
temperature dependence and optical-mode weighting are not resolved here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import qr
from scipy.sparse.linalg import eigsh, splu, spsolve

from .thermal import ThermalNetwork, ThermalParams


@dataclass
class _FEMSystem:
    capacitance: Any
    conductance: Any
    heater_load: np.ndarray
    ring_average: np.ndarray
    heater_average: np.ndarray
    boundary_load: np.ndarray
    free_dofs: np.ndarray
    mesh: Any
    n_total_dofs: int
    modeled_length_m: float
    boundary_contact_w_per_m2_k: float


def _coordinates(level: int) -> tuple[np.ndarray, np.ndarray]:
    # Layer/region interfaces are aligned with the mesh, including the heater.
    multiplier = 2**level

    def segments(edges: list[float], counts: list[int]) -> np.ndarray:
        return (
            np.concatenate(
                [
                    np.linspace(edges[i], edges[i + 1], counts[i] * multiplier + 1)[:-1]
                    for i in range(len(counts))
                ]
                + [np.array([edges[-1]])]
            )
            * 1e-6
        )

    x = segments([-8, -2, -1, -0.25, 0.25, 1, 2, 8], [9, 4, 5, 4, 5, 4, 9])
    z = segments([-8, -2, 0, 0.22, 1, 1.1, 1.8], [9, 10, 2, 5, 2, 4])
    return x, z


def _assemble(
    mesh_level: int, modeled_length_m: float, max_dofs: int = 25000, boundary_contact_w_per_m2_k: float = 1e7
) -> _FEMSystem:
    try:
        from skfem import Basis, BilinearForm, ElementTriP1, FacetBasis, LinearForm, MeshTri, asm
        from skfem.helpers import dot, grad
    except ImportError as error:
        raise ImportError("Install optional FEM dependencies with: pip install -e '.[physics]'") from error
    if isinstance(mesh_level, bool) or int(mesh_level) != mesh_level or not 0 <= mesh_level <= 3:
        raise ValueError("mesh_level must be an integer in [0,3]")
    if not np.isfinite(modeled_length_m) or modeled_length_m <= 0:
        raise ValueError("modeled_length_m must be positive and finite")
    if not np.isfinite(boundary_contact_w_per_m2_k) or boundary_contact_w_per_m2_k <= 0:
        raise ValueError("boundary_contact_w_per_m2_k must be positive and finite")
    x, z = _coordinates(int(mesh_level))
    requested_dofs = len(x) * len(z)
    if requested_dofs > max_dofs:
        raise ValueError(f"mesh would contain {requested_dofs} DOFs; memory guard limit is {max_dofs}")
    mesh = MeshTri.init_tensor(x, z)
    basis = Basis(mesh, ElementTriP1())
    centers = mesh.p[:, mesh.t].mean(axis=1)
    substrate = centers[1] < -2e-6
    ring = (np.abs(centers[0]) < 0.25e-6) & (centers[1] > 0) & (centers[1] < 0.22e-6)
    heater = (np.abs(centers[0]) < 1e-6) & (centers[1] > 1e-6) & (centers[1] < 1.1e-6)
    silicon = substrate | ring
    # k [W/(m K)], volumetric heat capacity rho*cp [J/(m^3 K)].
    k = np.full(mesh.nelements, 1.4)
    rho_cp = np.full(mesh.nelements, 1.55e6)
    k[silicon] = 130.0
    rho_cp[silicon] = 1.63e6
    k[heater] = 20.0
    rho_cp[heater] = 2.8e6

    @BilinearForm
    def conductivity(u, v, w):
        return w.k * dot(grad(u), grad(v)) * modeled_length_m

    @BilinearForm
    def heat_capacity(u, v, w):
        return w.rho_cp * u * v * modeled_length_m

    @LinearForm
    def volume_average(v, w):
        return v * modeled_length_m

    @BilinearForm
    def contact_conductance(u, v, w):
        return boundary_contact_w_per_m2_k * u * v * modeled_length_m

    @LinearForm
    def contact_load(v, w):
        return boundary_contact_w_per_m2_k * v * modeled_length_m

    conductivity_matrix = asm(conductivity, basis, k=k[:, None]).tocsr()
    capacity_matrix = asm(heat_capacity, basis, rho_cp=rho_cp[:, None]).tocsr()
    bottom_facets = mesh.facets_satisfying(lambda point: np.isclose(point[1], -8e-6, rtol=0, atol=1e-14))
    bottom_basis = FacetBasis(mesh, ElementTriP1(), facets=bottom_facets)
    conductivity_matrix = conductivity_matrix + asm(contact_conductance, bottom_basis)
    boundary_load = asm(contact_load, bottom_basis)
    heater_basis = Basis(mesh, ElementTriP1(), elements=np.flatnonzero(heater))
    ring_basis = Basis(mesh, ElementTriP1(), elements=np.flatnonzero(ring))
    heater_volume_weights = asm(volume_average, heater_basis)
    ring_volume_weights = asm(volume_average, ring_basis)
    heater_load = heater_volume_weights / heater_volume_weights.sum()
    ring_average = ring_volume_weights / ring_volume_weights.sum()
    active = np.arange(basis.N)
    return _FEMSystem(
        capacity_matrix.tocsc(),
        conductivity_matrix.tocsc(),
        heater_load,
        ring_average,
        heater_load,
        boundary_load,
        active,
        mesh,
        basis.N,
        modeled_length_m,
        boundary_contact_w_per_m2_k,
    )


def _full_transient(
    system: _FEMSystem,
    times: np.ndarray,
    power: float,
    subdivisions: int,
    boundary_rise_k: float | Callable[[float], float] = 0,
) -> np.ndarray:
    """Sparse backward Euler; refinement is reported rather than hidden."""
    state = np.zeros(len(system.free_dofs))
    response = np.zeros(len(times))
    previous_time = 0.0
    for i, time in enumerate(times):
        if time > previous_time:
            dt = (time - previous_time) / subdivisions
            factor = splu(system.capacitance + dt * system.conductance)
            for substep in range(subdivisions):
                end_time = previous_time + (substep + 1) * dt
                boundary = boundary_rise_k(end_time) if callable(boundary_rise_k) else boundary_rise_k
                state = factor.solve(
                    system.capacitance @ state
                    + dt * (system.heater_load * power + system.boundary_load * boundary)
                )
        response[i] = system.ring_average @ state
        previous_time = time
    return response


def _reduced_params(system: _FEMSystem, n_modes: int) -> tuple[ThermalParams, np.ndarray]:
    n = len(system.free_dofs)
    if isinstance(n_modes, bool) or int(n_modes) != n_modes or not 1 <= n_modes <= min(n - 2, 128):
        raise ValueError("n_modes must be an integer in [1,min(free_dofs-2,128)]")
    rates, modes = eigsh(
        system.conductance, k=int(n_modes), M=system.capacitance, sigma=0, which="LM", tol=1e-9, v0=np.ones(n)
    )
    order = np.argsort(rates)
    rates, modes = rates[order], modes[:, order]
    static_response = spsolve(system.conductance, system.heater_load)
    # Include uniform temperature and heater DC response explicitly.  This
    # preserves boundary equilibrium and DC gain instead of hiding truncation.
    candidates = np.column_stack([np.ones(n), static_response, modes])
    candidates /= np.linalg.norm(candidates, axis=0)
    vectors, r, _ = qr(candidates, mode="economic", pivoting=True)
    rank = int(np.sum(np.abs(np.diag(r)) > 1e-10))
    vectors = vectors[:, :rank]
    c = vectors.T @ (system.capacitance @ vectors)
    g = vectors.T @ (system.conductance @ vectors)
    c = 0.5 * (c + c.T)
    g = 0.5 * (g + g.T)
    uniform = vectors.T @ np.ones(n)
    params = ThermalParams(
        c,
        g,
        (vectors.T @ system.heater_load)[:, None],
        (system.ring_average @ vectors)[None, :],
        boundary_map_w_per_k=(vectors.T @ system.boundary_load)[:, None],
        heater_temperature_map=(system.heater_average @ vectors)[None, :],
        reference_state_map=uniform,
        physical_network=False,
        provenance="generic SOI scikit-fem congruence ROM; NOT calibrated to Lightmatter",
    )
    return params, rates


def thermal_fem_reference(
    mesh_level: int = 0,
    n_modes: int = 24,
    modeled_length_m: float = 100e-6,
    heater_power_w: float = 1e-3,
    transient_subdivisions: int = 8,
    boundary_contact_w_per_m2_k: float = 1e7,
) -> dict[str, Any]:
    """Return sparse-FEM reference, compatible passive ROM, and validation data.

    Use ``ThermalParams(**result['thermal_params'])`` to run this ROM through
    the same control interface.  A transient comparison is numerical validation
    against the same assumed cross-section, not experimental device validation.
    """
    if not np.isfinite(heater_power_w) or heater_power_w <= 0:
        raise ValueError("heater_power_w must be positive and finite")
    if int(transient_subdivisions) != transient_subdivisions or transient_subdivisions < 1:
        raise ValueError("transient_subdivisions must be a positive integer")
    system = _assemble(mesh_level, modeled_length_m, boundary_contact_w_per_m2_k=boundary_contact_w_per_m2_k)
    params, full_rates = _reduced_params(system, n_modes)
    steady = spsolve(system.conductance, system.heater_load)
    full_dc = float(system.ring_average @ steady)
    network = ThermalNetwork(params)
    times = np.r_[0.0, np.geomspace(1e-9, 2e-3, 72)]
    full_response = _full_transient(system, times, heater_power_w, transient_subdivisions)
    rom_response = []
    previous = 0.0
    for time in times:
        rom_response.append(float(network.step(time - previous, [heater_power_w], 300.0)[0] - 300.0))
        previous = time
    rom_response = np.asarray(rom_response)
    scale = max(float(np.max(np.abs(full_response))), 1e-15)
    maximum_rom_error = float(np.max(np.abs(rom_response - full_response)) / scale)
    csr_memory = sum(
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        for matrix in (system.capacitance, system.conductance)
    )
    return {
        "schema": "mrm-model.thermal-fem-reference.v1",
        "calibration_status": "illustrative_geometry_unvalidated",
        "source": "https://scikit-fem.readthedocs.io/en/stable/listofexamples.html",
        "assumptions": {
            "geometry": "16 um wide; 6 um silicon substrate, 2 um BOX; 0.5 x 0.22 um silicon core; 2 x 0.1 um heater in oxide at z=1 um",
            "modeled_length_m": modeled_length_m,
            "power_convention": "total heater watts over modeled_length_m; deposited load sums to 1",
            "boundary": "finite bottom Robin thermal contact to ambient; top and both sides adiabatic; all boundary states retain consistent capacitance",
            "boundary_contact_w_per_m2_k": boundary_contact_w_per_m2_k,
            "uniform_bottom_contact_resistance_k_per_w": float(
                1 / (boundary_contact_w_per_m2_k * 16e-6 * modeled_length_m)
            ),
            "materials": "constant illustrative bulk k/rhoCp: Si 130/1.63e6, SiO2 1.4/1.55e6, generic resistive heater 20/2.8e6 (SI units)",
            "ring_observation": "uniform volume-average silicon-core temperature; no optical mode weighting",
            "omissions": [
                "3-D ring curvature",
                "thermal boundary resistance",
                "actual metal contacts",
                "package",
                "temperature-dependent materials",
                "Lightmatter fabrication data",
            ],
        },
        "mesh": {
            "level": int(mesh_level),
            "total_dofs": int(system.n_total_dofs),
            "free_dofs": len(system.free_dofs),
            "triangles": int(system.mesh.nelements),
            "sparse_matrix_bytes": csr_memory,
            "maximum_allowed_dofs": 25000,
        },
        "steady_ring_temperature_rise_k_per_w": full_dc,
        "steady_heater_temperature_rise_k_per_w": float(system.heater_average @ steady),
        "slow_full_fem_time_constants_s": (1 / full_rates).tolist(),
        "thermal_params": params.to_dict(),
        "rom_state_count": params.n_states,
        "rom_dc_relative_error": float(abs(network.equilibrium([1.0], 300.0)[0] - 300.0 - full_dc) / full_dc),
        "rom_transient_max_relative_to_full_scale": maximum_rom_error,
        "transient_method": f"full sparse backward Euler, {transient_subdivisions} subdivisions per log-spaced interval; ROM exact ZOH",
        "step_response": {
            "time_s": times.tolist(),
            "heater_power_w": heater_power_w,
            "full_fem_ring_rise_k": full_response.tolist(),
            "rom_ring_rise_k": rom_response.tolist(),
        },
    }


def thermal_fem_convergence(
    n_modes: int = 24, modeled_length_m: float = 100e-6, boundary_contact_w_per_m2_k: float = 1e7
) -> dict[str, Any]:
    """Coarse/fine spatial, temporal, and passive-ROM validation, with raw errors."""
    coarse = thermal_fem_reference(
        0, n_modes, modeled_length_m, boundary_contact_w_per_m2_k=boundary_contact_w_per_m2_k
    )
    fine = thermal_fem_reference(
        1, n_modes, modeled_length_m, boundary_contact_w_per_m2_k=boundary_contact_w_per_m2_k
    )
    system = _assemble(1, modeled_length_m, boundary_contact_w_per_m2_k=boundary_contact_w_per_m2_k)
    times = np.asarray(fine["step_response"]["time_s"])
    temporal_fine = _full_transient(system, times, 1e-3, 16)
    temporal_coarse = np.asarray(fine["step_response"]["full_fem_ring_rise_k"])
    temporal_scale = max(float(np.max(np.abs(temporal_fine))), 1e-15)
    gain_coarse = coarse["steady_ring_temperature_rise_k_per_w"]
    gain_fine = fine["steady_ring_temperature_rise_k_per_w"]
    spatial_error = abs(gain_coarse - gain_fine) / gain_fine
    tau_coarse = coarse["slow_full_fem_time_constants_s"][0]
    tau_fine = fine["slow_full_fem_time_constants_s"][0]
    return {
        "schema": "mrm-model.thermal-fem-convergence.v1",
        "claim": "numerical checks for one explicitly assumed generic section; no Lightmatter device accuracy claim",
        "steady_gain_relative_mesh_change": float(spatial_error),
        "slowest_time_constant_relative_mesh_change": float(abs(tau_coarse - tau_fine) / tau_fine),
        "transient_relative_full_scale_temporal_change": float(
            np.max(np.abs(temporal_coarse - temporal_fine)) / temporal_scale
        ),
        "coarse": coarse,
        "fine": fine,
        "refined_time_response_k": temporal_fine.tolist(),
    }
