"""Optional full-vector mixed-FEM optical mode extraction for a generic strip.

Femwell computes transverse Nedelec and longitudinal Lagrange fields (order 2),
not a scalar effective-index approximation.  Our interface-aligned triangular
mesh uses scikit-fem directly; no CAD installation or proprietary solver is
needed.  The resulting coefficients describe the declared generic geometry,
not a Lightmatter G26 PDK or a measured microring.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

# CC0 tabulation from refractiveindex.info, Li at 293 K.  Use measured dispersion
# rather than holding silicon index constant during the group-index derivative.
# https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main/Si/nk/Li-293K.yml
_SI_WAVELENGTH_UM = np.array(
    [
        1.20,
        1.22,
        1.24,
        1.26,
        1.28,
        1.30,
        1.32,
        1.34,
        1.36,
        1.38,
        1.40,
        1.45,
        1.50,
        1.55,
        1.60,
        1.65,
        1.70,
        1.80,
        1.90,
        2.00,
        2.25,
        2.50,
        2.75,
        3.00,
        4.00,
        5.00,
        6.00,
        7.00,
        8.00,
        9.00,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
    ]
)
_SI_INDEX = np.array(
    [
        3.5167,
        3.5133,
        3.5102,
        3.5072,
        3.5043,
        3.5016,
        3.4990,
        3.4965,
        3.4941,
        3.4918,
        3.4896,
        3.4845,
        3.4799,
        3.4757,
        3.4719,
        3.4684,
        3.4653,
        3.4597,
        3.4550,
        3.4510,
        3.4431,
        3.4375,
        3.4334,
        3.4302,
        3.4229,
        3.4195,
        3.4177,
        3.4165,
        3.4158,
        3.4153,
        3.4150,
        3.4147,
        3.4145,
        3.4144,
        3.4142,
    ]
)
_SI_DISPERSION = PchipInterpolator(_SI_WAVELENGTH_UM, _SI_INDEX, extrapolate=False)


def material_indices(wavelength_m: float) -> tuple[float, float]:
    """Return bulk crystalline silicon and fused-silica indices at ~293 K.

    Silicon is PCHIP interpolation of Li's tabulated data; silica uses the
    Malitson Sellmeier formula.  Restrict the shared validated spectral range.
    Film/doping/strain/process dependence and material absorption are omitted.
    """
    wavelength_um = wavelength_m * 1e6
    if not np.isfinite(wavelength_um) or not 1.2 <= wavelength_um <= 3.71:
        raise ValueError("reference material data supports wavelengths in [1.2,3.71] um")
    square = wavelength_um**2
    silica_squared = (
        1
        + 0.6961663 * square / (square - 0.0684043**2)
        + 0.4079426 * square / (square - 0.1162414**2)
        + 0.8974794 * square / (square - 9.896161**2)
    )
    return float(_SI_DISPERSION(wavelength_um)), float(np.sqrt(silica_squared))


def _make_basis(width_m: float, height_m: float, mesh_scale: float, padding_m: float):
    try:
        from femwell.maxwell.waveguide import compute_modes
        from skfem import Basis, ElementTriN2, ElementTriP0, ElementTriP2, MeshTri
    except ImportError as error:
        raise ImportError(
            "Optical mode extraction requires femwell, scikit-fem and matplotlib; install the 'modes' optional extra"
        ) from error
    del compute_modes
    for name, value in (
        ("width_m", width_m),
        ("height_m", height_m),
        ("mesh_scale", mesh_scale),
        ("padding_m", padding_m),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not 0.2e-6 <= width_m <= 1e-6 or not 0.1e-6 <= height_m <= 0.5e-6:
        raise ValueError("memory-bounded reference supports width [0.2,1] um and height [0.1,0.5] um")
    if mesh_scale < 0.35:
        raise ValueError("mesh_scale below .35 exceeds the reference memory guard")
    width_um, height_um, padding_um = width_m * 1e6, height_m * 1e6, padding_m * 1e6

    def grid(core_size: float, core_count: int) -> np.ndarray:
        half = core_size / 2
        edges = [
            -half - padding_um,
            -half - 0.5,
            -half - 0.1,
            -half,
            half,
            half + 0.1,
            half + 0.5,
            half + padding_um,
        ]
        if padding_um <= 0.5:
            raise ValueError("padding_m must exceed 0.5 um")
        counts = np.ceil(np.array([4, 4, 3, core_count, 3, 4, 4]) / mesh_scale).astype(int)
        return np.concatenate(
            [np.linspace(edges[i], edges[i + 1], counts[i] + 1)[:-1] for i in range(len(counts))]
            + [np.array([edges[-1]])]
        )

    x, y = grid(width_um, 8), grid(height_um, 6)
    # Conservative upper bound on mixed order-2 DOFs for this tensor mesh.
    estimated_dofs = 14 * len(x) * len(y)
    if estimated_dofs > 42000:
        raise ValueError(f"estimated mixed FEM DOFs {estimated_dofs} exceed 42000 memory guard")
    mesh = MeshTri.init_tensor(x, y)
    mesh = mesh.with_subdomains(
        {
            "core": lambda point: (np.abs(point[0]) < width_um / 2) & (np.abs(point[1]) < height_um / 2),
        }
    )
    basis0 = Basis(mesh, ElementTriP0())
    mixed_dofs = Basis(mesh, ElementTriN2() * ElementTriP2()).N
    if mixed_dofs > 42000:
        raise ValueError("mixed FEM DOFs exceed 42000 memory guard")
    return basis0, {
        "triangles": int(mesh.nelements),
        "vertices": int(mesh.nvertices),
        "mixed_dofs": int(mixed_dofs),
        "mesh_scale": mesh_scale,
        "near_core_nominal_dx_m": width_m / int(np.ceil(8 / mesh_scale)),
        "near_core_nominal_dy_m": height_m / int(np.ceil(6 / mesh_scale)),
        "max_mixed_dofs": 42000,
    }


def _solve_mode(
    basis0,
    wavelength_m: float,
    reference=None,
    temperature_change_k: float = 0,
    silicon_dn_dt_per_k: float = 0,
    silica_dn_dt_per_k: float = 0,
):
    from femwell.maxwell.waveguide import compute_modes

    silicon, silica = material_indices(wavelength_m)
    silicon += silicon_dn_dt_per_k * temperature_change_k
    silica += silica_dn_dt_per_k * temperature_change_k
    if silicon <= silica or silica <= 0:
        raise ValueError("provided thermo-optic coefficients destroy the assumed guiding index contrast")
    epsilon = np.full(basis0.N, silica**2)
    epsilon[basis0.get_dofs(elements="core")] = silicon**2
    modes = compute_modes(
        basis0, epsilon, wavelength=wavelength_m * 1e6, num_modes=2, order=2, metallic_boundaries=True
    )
    guided = [mode for mode in modes if silica < mode.n_eff.real < silicon and abs(mode.n_eff.imag) < 1e-7]
    if not guided:
        raise ValueError("no bound core mode found; refine mesh or check geometry")
    if reference is None:
        transverse_electric = [mode for mode in guided if mode.te_fraction > 0.5]
        if not transverse_electric:
            raise ValueError("no quasi-TE core mode found among requested eigenmodes")
        mode = max(transverse_electric, key=lambda candidate: candidate.n_eff.real)
        overlap = 1.0
    else:
        overlap_values = [float(abs(reference.calculate_overlap(candidate)) ** 2) for candidate in guided]
        mode_index = int(np.argmax(overlap_values))
        mode, overlap = guided[mode_index], overlap_values[mode_index]
        if overlap < 0.9:
            raise ValueError(f"mode branch overlap {overlap:.5f} below 0.9; derivative is ambiguous")
    return mode, {
        "neff": float(mode.n_eff.real),
        "te_fraction": float(mode.te_fraction),
        "core_power_fraction": float(mode.calculate_power(elements="core").real),
        "overlap_with_reference": overlap,
    }


def waveguide_mode_reference(
    width_m: float = 400e-9,
    height_m: float = 200e-9,
    wavelength_m: float = 1310e-9,
    mesh_scale: float = 1.0,
    delta_wavelength_m: float = 2e-9,
    padding_m: float = 1.5e-6,
    silicon_dn_dt_per_k: float | None = None,
    silica_dn_dt_per_k: float | None = None,
    delta_temperature_k: float = 0.5,
) -> dict[str, Any]:
    """Compute quasi-TE neff/ng and optional *specified-material* dn_eff/dT.

    Group index uses five full-vector eigenmode solves, central differences
    with steps h and h/2, material dispersion and mode-overlap branch tracking.
    Returned wavelength-step discrepancy separates derivative truncation error
    from spatial mesh error.  Thermo-optic coefficients are required explicitly:
    an unknown silicon process coefficient is never inferred from the geometry.
    """
    if (
        not np.isfinite(delta_wavelength_m)
        or delta_wavelength_m <= 0
        or delta_wavelength_m > 0.01 * wavelength_m
    ):
        raise ValueError("delta_wavelength_m must be positive and <=1% of wavelength")
    material_indices(wavelength_m - delta_wavelength_m)
    material_indices(wavelength_m + delta_wavelength_m)
    basis0, mesh_report = _make_basis(width_m, height_m, mesh_scale, padding_m)
    center, central_report = _solve_mode(basis0, wavelength_m)
    samples = []
    derivatives = []
    for step in [delta_wavelength_m, delta_wavelength_m / 2]:
        _, lower = _solve_mode(basis0, wavelength_m - step, center)
        _, upper = _solve_mode(basis0, wavelength_m + step, center)
        derivative = (upper["neff"] - lower["neff"]) / (2 * step)
        derivatives.append(derivative)
        samples.extend(
            [dict(wavelength_m=wavelength_m - step, **lower), dict(wavelength_m=wavelength_m + step, **upper)]
        )
    groups = [central_report["neff"] - wavelength_m * derivative for derivative in derivatives]
    thermal = None
    if (silicon_dn_dt_per_k is None) != (silica_dn_dt_per_k is None):
        raise ValueError("supply both silicon_dn_dt_per_k and silica_dn_dt_per_k for thermal extraction")
    if silicon_dn_dt_per_k is not None:
        if (
            not all(
                np.isfinite(value) for value in [silicon_dn_dt_per_k, silica_dn_dt_per_k, delta_temperature_k]
            )
            or delta_temperature_k <= 0
        ):
            raise ValueError("thermal coefficients must be finite and temperature step positive")
        derivative_values = []
        for delta in [delta_temperature_k, delta_temperature_k / 2]:
            _, minus = _solve_mode(
                basis0, wavelength_m, center, -delta, silicon_dn_dt_per_k, silica_dn_dt_per_k
            )
            _, plus = _solve_mode(
                basis0, wavelength_m, center, delta, silicon_dn_dt_per_k, silica_dn_dt_per_k
            )
            derivative_values.append((plus["neff"] - minus["neff"]) / (2 * delta))
        thermal = {
            "dn_eff_per_k": derivative_values[1],
            "temperature_step_relative_change": abs(derivative_values[0] - derivative_values[1])
            / max(abs(derivative_values[1]), 1e-15),
            "silicon_dn_dt_per_k": silicon_dn_dt_per_k,
            "silica_dn_dt_per_k": silica_dn_dt_per_k,
            "temperature_step_k": delta_temperature_k,
            "claim": "full-vector response to explicitly supplied linear bulk material coefficients; geometry/thermal expansion held fixed",
        }
    silicon, silica = material_indices(wavelength_m)
    return {
        "schema": "mrm-model.full-vector-waveguide-reference.v1",
        "calibration_status": "generic_geometry_material_reference_NOT_Lightmatter_PDK",
        "solver": "Femwell full-vector order-2 mixed Nedelec/Lagrange Maxwell eigenproblem; sparse SciPy shift-invert",
        "versions": {"femwell": version("femwell"), "scikit-fem": version("scikit-fem")},
        "geometry": {
            "width_m": width_m,
            "height_m": height_m,
            "padding_m": padding_m,
            "topology": "straight rectangular silicon strip in infinite-oxide approximation to SOI BOX/cladding; no explicit handle substrate",
            "boundary": "PEC at outer rectangle; evanescent mode/domain convergence must be checked",
        },
        "materials": {
            "temperature_k": 293.0,
            "silicon_index": silicon,
            "silica_index": silica,
            "silicon_model": "Li 293K public CC0 tabulated index, shape-preserving cubic interpolation",
            "silica_model": "Malitson 1965 fused silica Sellmeier",
            "limitations": "real undoped bulk indices; 4-decimal Si source precision; no thin-film/doping/strain/absorption calibration",
        },
        "sources": [
            "https://helgegehring.github.io/femwell/photonics/examples/waveguide_modes.html",
            "https://github.com/HelgeGehring/femwell/blob/main/femwell/maxwell/waveguide.py",
            "https://doi.org/10.1063/1.555624",
            "https://doi.org/10.1364/JOSA.55.001205",
            "https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main/Si/nk/Li-293K.yml",
        ],
        "mesh": mesh_report,
        "wavelength_m": wavelength_m,
        "neff": central_report["neff"],
        "ng": float(groups[1]),
        "ng_coarse_wavelength_step": float(groups[0]),
        "wavelength_step_relative_ng_change": float(abs(groups[0] - groups[1]) / abs(groups[1])),
        "delta_wavelength_m": delta_wavelength_m,
        "mode": central_report,
        "wavelength_samples": samples,
        "thermal_sensitivity": thermal,
        "omissions": [
            "ring bend and radiation loss",
            "sidewall roughness",
            "bus-ring coupler",
            "junction/carriers",
            "actual BOX/handle substrate",
            "process-specific optical material data",
        ],
    }


def waveguide_mode_convergence(**kwargs) -> dict[str, Any]:
    """Run actual mesh refinement, derivative-step refinement and domain test."""
    coarse_kwargs = dict(kwargs, mesh_scale=1.0)
    fine_kwargs = dict(kwargs, mesh_scale=0.6)
    coarse = waveguide_mode_reference(**coarse_kwargs)
    fine = waveguide_mode_reference(**fine_kwargs)
    enlarged_kwargs = dict(fine_kwargs)
    enlarged_kwargs["padding_m"] = 2 * fine["geometry"]["padding_m"]
    basis, enlarged_mesh = _make_basis(
        enlarged_kwargs.get("width_m", 400e-9),
        enlarged_kwargs.get("height_m", 200e-9),
        0.6,
        enlarged_kwargs["padding_m"],
    )
    _, enlarged_mode = _solve_mode(basis, fine["wavelength_m"])
    return {
        "schema": "mrm-model.full-vector-waveguide-convergence.v1",
        "claim": "numerical convergence for declared generic geometry and materials; no measured Lightmatter accuracy claim",
        "neff_absolute_mesh_change": abs(coarse["neff"] - fine["neff"]),
        "neff_relative_mesh_change": abs(coarse["neff"] - fine["neff"]) / abs(fine["neff"]),
        "ng_relative_mesh_change": abs(coarse["ng"] - fine["ng"]) / abs(fine["ng"]),
        "neff_relative_boundary_domain_change": abs(enlarged_mode["neff"] - fine["neff"]) / abs(fine["neff"]),
        "enlarged_domain_mode": dict(
            enlarged_mode, padding_m=enlarged_kwargs["padding_m"], mesh=enlarged_mesh
        ),
        "coarse": coarse,
        "fine": fine,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generic full-vector Femwell waveguide mode extraction")
    parser.add_argument("--output", default="outputs/waveguide_modes.json")
    parser.add_argument("--width-nm", type=float, default=400)
    parser.add_argument("--height-nm", type=float, default=200)
    parser.add_argument("--wavelength-nm", type=float, default=1310)
    parser.add_argument("--delta-wavelength-nm", type=float, default=2)
    args = parser.parse_args(argv)
    report = waveguide_mode_convergence(
        width_m=args.width_nm * 1e-9,
        height_m=args.height_nm * 1e-9,
        wavelength_m=args.wavelength_nm * 1e-9,
        delta_wavelength_m=args.delta_wavelength_nm * 1e-9,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "neff": report["fine"]["neff"],
                "ng": report["fine"]["ng"],
                "neff_relative_mesh_change": report["neff_relative_mesh_change"],
                "ng_relative_mesh_change": report["ng_relative_mesh_change"],
                "neff_relative_boundary_domain_change": report["neff_relative_boundary_domain_change"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
