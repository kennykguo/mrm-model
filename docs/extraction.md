# Open-source physics extraction

The control simulator accepts measured coefficients, a physical heat-flow network, or a passive finite-element reduced-order model through the same configuration. Offline solvers help replace assumptions with independently calculated values. A converged calculation of an assumed structure is numerical verification; validation against Lightmatter hardware still requires device data.

## Thermal diffusion and a passive ROM

Run the sparse thermal reference and its mesh/time checks:

```sh
uv pip install --python .venv/bin/python 'scikit-fem>=11,<13'
.venv/bin/mrm extract-thermal --modes 32 --output outputs/thermal_fem
```

The generic 2-D SOI section contains a silicon substrate, buried oxide, a silicon waveguide, oxide cladding, and a generic resistive heater. The bottom exchanges heat with ambient through an explicitly specified Robin contact; top and sides are insulated. The default contact conductance is an illustrative `1e7 W/(m² K)`, giving `62.5 K/W` across the 16 um-wide bottom over a 100 um modeled length. It is a configurable boundary assumption, not a measured package property. The geometry and constant bulk thermal properties are recorded in the exported report. The heat equation uses consistent finite-element capacitance and conductivity matrices; matrices and heater deposition include the modeled third-direction length. Heater power means total watts over that length, so changing the length at fixed watts changes the temperature rise.

All finite-element boundary states retain their heat capacity. Ambient enters through the contact heat flux, preserving temperature memory when ambient changes. Eliminating time-varying Dirichlet boundary nodes with a consistent mass matrix would require an additional boundary-temperature derivative term; silently omitting that term would corrupt the exported ambient dynamics. The Robin formulation avoids this inconsistency. Tests compare both ambient steps and ramps against independently integrated full sparse meshes.

The extraction uses sparse P1 finite elements and a congruence projection. Slow thermal eigenvectors, a uniform-temperature vector, and the static heater response form the reduced basis. Congruence preserves positive-definite heat storage and conductance. The explicit uniform vector preserves ambient equilibrium and the static response preserves DC heater gain. Signed reduced coordinates are allowed; they represent modes and must not be interpreted as individual physical node temperatures.

The full mesh heater step is integrated independently with backward Euler; doubling its temporal subdivisions estimates time-discretization error. The ROM uses the compact model's exact matrix exponential. Comparing their time traces measures ROM truncation plus the remaining reference time error. Coarse/fine mesh changes measure spatial discretization error for this specified geometry. The shipped [thermal_fem.json](../examples/thermal_fem.json) records all checks and exported coefficients.

The included Robin-contact run uses 1,353 and 5,265 DOFs. Its coarse/fine heater gain changes by 0.5032%, its slowest time constant by 0.0448%, and its temporal refinement changes the heater-step response by 0.1343% of the final rise. The 34-state ROM differs from the fine full-mesh step by at most 0.5802% of that rise and preserves the static gain to numerical precision. The assumed section's fine-mesh ring thermal resistance is 2,358.17 K/W. These numbers verify this particular numerical reduction, not a target device's thermal parameters.

Use the exported ROM in an experiment:

```python
from dataclasses import replace
import json
from pathlib import Path

from mrm_model.config import default_config
from mrm_model.simulation import run_experiment
from mrm_model.thermal import ThermalParams

report = json.loads(Path("outputs/thermal_fem/reference.json").read_text())
thermal = ThermalParams(**report["thermal_params"])
config = replace(default_config(), thermal=thermal)
result = run_experiment(config)
```

The FEM geometry has a different heater efficiency from the compact defaults. Recalibrate the optical operating point and controller context for comparisons. The default numerical values are not interchangeable device calibrations. The full reference lacks measured interfaces, actual metal contacts, a package, 3-D ring curvature, and temperature-dependent thermal properties. Its ring observation is a volume average rather than an optical-mode-weighted temperature.

This implementation follows the [scikit-fem heat-equation examples](https://scikit-fem.readthedocs.io/en/stable/listofexamples.html), using their public sparse assembly interface.

## Full-vector optical waveguide modes

The optional mode reference solves the vector Maxwell eigenproblem for a generic straight, oxide-clad silicon strip. It uses Femwell's second-order transverse Nedelec edge elements and longitudinal Lagrange elements. The rectangular material interfaces align with a graded triangular mesh. Both transverse and longitudinal fields are solved; the calculation includes the high-contrast vector boundary behavior.

Femwell's general package dependency list includes CAD and visualization packages that this particular calculation does not import. A low-footprint install in the existing project environment is:

```sh
uv pip install --python .venv/bin/python 'scikit-fem>=11,<13' 'matplotlib>=3.8,<4'
uv pip install --python .venv/bin/python --no-deps femwell==0.1.12
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m mrm_model.mode_extraction \
  --width-nm 400 --height-nm 200 --wavelength-nm 1310 \
  --output outputs/waveguide_modes.json
```

The `--no-deps` command is specific to this implementation, whose imports are exercised by the tests. Other Femwell examples may require additional dependencies. This workflow needs neither Gmsh nor a CAD mesh because the strip's conforming mesh is generated directly by scikit-fem. Femwell is open-source GPL-3.0 software; review its license before distributing derived code or including it in another product.

The report contains actual coarse/fine eigenmode solves, a doubled-domain test, and wavelength derivative checks.

The shipped [waveguide_modes.json](../examples/waveguide_modes.json) contains the actual 400 x 200 nm, 1310 nm extraction. Its fine mesh has 35,345 mixed vector DOFs, `neff=2.48538163` and `ng=4.32846223`. Coarse/fine changes are 0.00218% and 0.00964%, respectively; halving the wavelength derivative step changes `ng` by 0.0000444%. Doubling the outer-boundary distance changes `neff` by 0.00000612%. The convergence run measured 757.3 MiB peak process resident memory with single-threaded BLAS. This measured value is specific to the recorded default problem; a DOF guard prevents substantially larger allocations.

The group index is

\[
n_g = n_{\rm eff} - \lambda\frac{\partial n_{\rm eff}}{\partial\lambda}.
\]

It uses central differences with wavelength steps of 2 nm and 1 nm. Every side sample solves the full vector eigenproblem with updated material dispersion. Field overlap tracks the same quasi-TE mode, rather than comparing two accidentally different eigenvalues. An overlap below 0.9 rejects an ambiguous derivative. The outer boundary is PEC, sufficiently far into the oxide for this bound mode; doubling its distance checks its residual effect. This calculation does not predict radiation loss.

Silicon dispersion uses shape-preserving interpolation of the public Li 293 K tabulation. Silica uses the Malitson Sellmeier equation. Bulk material data has its own uncertainty: the silicon table is rounded to four decimals, and a small finite-difference convergence error does not imply equivalently accurate material dispersion. Optical films, doped silicon, strain, roughness, waveguide bends, actual BOX/handle-substrate leakage, and the bus-ring coupler remain outside this straight-waveguide reference.

```python
from mrm_model.mode_extraction import waveguide_mode_reference

reference = waveguide_mode_reference()
print(reference["neff"], reference["ng"])

# Only if these material coefficients are supported by your chosen dataset:
with_temperature = waveguide_mode_reference(
    silicon_dn_dt_per_k=1.86e-4,
    silica_dn_dt_per_k=1e-5,
)
print(with_temperature["thermal_sensitivity"])
```

The two example temperature coefficients are user-selected illustrative inputs, not coefficients extracted from the Lightmatter paper. The default omits a temperature derivative because the room-temperature dispersion table alone does not identify one. If coefficients are supplied, separate full-vector solves perturb both materials and check two temperature-step sizes. Geometry is held fixed, so that result omits thermal expansion and stress. Coupling this derivative to a ring resonance additionally requires the ring's actual group index and circumference expansion.

Use `neff` and `ng` to inform the compact optical phase and FSR after choosing the appropriate geometry. Do not overwrite the paper's reported thermal sensitivity or claim calibrated Lightmatter coefficients from this reference. Ring loss and bus/drop coupling still need measured spectra or their own independently checked full-wave or coupled-mode extraction.

Primary references: [Femwell waveguide-mode example](https://helgegehring.github.io/femwell/photonics/examples/waveguide_modes.html), [Femwell solver source](https://github.com/HelgeGehring/femwell/blob/main/femwell/maxwell/waveguide.py), [Li silicon-index reference](https://doi.org/10.1063/1.555624), [public CC0 Li tabulation](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main/Si/nk/Li-293K.yml), and [Malitson fused-silica dispersion](https://doi.org/10.1364/JOSA.55.001205).
