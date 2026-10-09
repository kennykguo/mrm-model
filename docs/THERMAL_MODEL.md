# Thermal physics, extraction, and calibration

The default thermal circuit is an **assumed research plant**, not an extracted
Lightmatter device. Public reported drift rates and resonance sensitivity do
not determine heat capacities, heater efficiency, geometry, package impedance,
or a thermal transfer function. No measured thermal response is bundled.

## State and power conventions

`ThermalNetwork` implements

\[
\operatorname{diag}(C)\dot T=-G T+Bp+H T_b,\qquad y=W T.
\]

`T` and `T_b` are **absolute Kelvin**, `C` is J/K, `G,H` are W/K, and `p` is W.
The observation `y` is an optical-mode-weighted temperature, approximated by a
waveguide-node temperature in the default circuit. `B` distributes the actual
heat deposited by each source. Electrical heater power and optically absorbed
power belong in separate source columns; launched laser power is not absorbed
heat. Optical heating must be recomputed from the instantaneous optical state
at each coupled RHS evaluation.

The network validates positive capacities, reciprocal internal conduction,
nonnegative deposition/observation weights, and

\[
G_{ij}\le0\ (i\ne j),\quad G\mathbf1=H\mathbf1,\quad
\mathbf1^T B=\mathbf1^T,\quad W\mathbf1=\mathbf1.
\]

Consequently, internal conduction conserves energy and

\[
\frac{d}{dt}\sum_i C_iT_i
=\sum_jp_j+\sum_{ib}H_{ib}(T_{b,b}-T_i).
\]

Every connected component with a thermal boundary has strictly decaying modes.
An insulated component has a zero mode and conserves heat rather than having a
unique steady state. `is_asymptotically_stable` detects the distinction. The
off-diagonal entries of the state matrix are nonnegative, so the full physical
RC system preserves positive temperatures and nonnegative rises for positive
power at a fixed initial ambient.

`step_zoh(T,p,T_b,dt)` uses SciPy's matrix exponential on an augmented system:
it is exact up to floating-point error for the **linear network with constant
power and boundary over the interval**. It supports insulated networks without
inverting a singular conductance matrix. For continuously changing ambient or
temperature-dependent optical absorption, integrate `derivative` with the
coupled ODE. A boundary step changes heat flow, never the finite-capacity states
instantaneously. A moving-temperature-rise state would require an additional
`-dT_b/dt` term; the absolute-state formulation avoids that bookkeeping error.

## Default circuit and adjustable assumptions

`default_ring_network(N, parameters)` contains N waveguide nodes, N heater
nodes, and one package node. The package connects to a prescribed temperature
boundary; nearest-neighbor waveguides also exchange heat. The source ordering
is `[heater_0..N-1, absorption_0..N-1, aggressor_0..N-1]`; the N observations
follow ring order. Names are exposed through `source_names`, `node_names`, and
`observer_names`.

Every value below is an explicit assumption in `AssumedThermalParameters`, not
a fit to the supplied paper. Choosing these values does not establish target
device accuracy.

| Parameter | Default | Units |
|---|---:|---|
| Waveguide heat capacity | 1.2e-10 | J/K |
| Heater heat capacity | 3.0e-10 | J/K |
| Package heat capacity | 4.0e-8 | J/K |
| Heater–waveguide conductance | 3.0e-5 | W/K |
| Waveguide–package conductance | 2.0e-5 | W/K |
| Heater–package conductance | 1.0e-5 | W/K |
| Neighbor waveguide conductance | 1.0e-6 | W/K |
| Package–boundary conductance | 2.0e-3 | W/K |
| Aggressor fraction deposited at package | 0.8 | dimensionless |

The remaining aggressor power heats the corresponding heater node. This is a
surrogate for an undisclosed nearby-heater location. A realistic disturbance
must be supplied as aggressor power or a boundary-temperature profile using
identified transfer functions. A quoted 808 K/s ring drift must not silently be
assigned to ambient or converted to heater watts without that transfer data.

The linear conductances and capacities are temperature independent. Over an
80 K excursion, material and interface changes can matter; uncertainty studies
or new identification are needed. Convection, radiation, interface resistance,
metal vias, temperature-dependent conductivity, and true package structure
require additional nodes/elements or a richer spatial extraction.

## Free spatial extraction

`ringlab.extraction.thermal` builds a layer-conforming finite-volume thermal
system and uses SciPy sparse direct solves and the BDF integrator. It implements
cell/source geometry and heat-flow assembly, not a new sparse or ODE solver.
The face conductance is the inverse of the two half-cell series resistances,
so interfaces use harmonic conduction. This follows the cell flux balance
described in [NIST FiPy's finite-volume documentation](https://pages.nist.gov/fipy/en/Newton_iterations/numerical/discret.html).

The included two-ring example has an 80 µm width and a 20 µm uniform extrusion:

| Bottom-to-top layer | Thickness | Assumed k (W/m/K) | Assumed volumetric C (J/m³/K) |
|---|---:|---:|---:|
| Silicon substrate | 20 µm | 130 | 1.63e6 |
| Buried oxide | 2 µm | 1.4 | 1.55e6 |
| Continuous silicon film surrogate | 0.22 µm | 80 | 1.63e6 |
| Oxide overcladding | 1.5 µm | 1.4 | 1.55e6 |

Ring centers are separated by 30 µm. Optical sources/observers span 0.6 µm
laterally in the silicon film. Heaters span 4 µm laterally in the top 0.4 µm
of overcladding. Aggressors occupy similarly sized regions offset by 7 µm.
Power is integrated over cell–source intersections and normalized to total
watts, so source power does not change with mesh resolution. The bottom is
isothermal; top and sides are adiabatic. This is a rectangular **SOI-like slab**,
not a curved ring, metal heater simulation, or Lightmatter stack. Thin-film
thermal properties depend on fabrication and need measurement; see the
[NIST thin-film silicon-dioxide measurement report](https://www.nist.gov/publications/thermal-conductivity-measurements-thin-film-silicon-dioxide).

```python
from ringlab.extraction.thermal import assemble_slab, assumed_slab_geometry

spatial = assemble_slab(assumed_slab_geometry(2), nx=128,
                        cells_per_layer=(12, 8, 4, 8))
dc_gain = spatial.dc_gain_k_per_w()  # 2 observations × 6 heat sources
```

`to_network()` exports a small mesh unchanged to the physical-temperature
control-network API. The default dense-export guard is 512 cells (hard maximum
2048); fine extraction meshes remain sparse. Assembly is guarded at 100,000
cells, and reduced bases also have explicit memory limits.

The analytic verification case—uniform volumetric heating in a homogeneous
slab, insulated top and fixed bottom—has average thermal gain `L/(3 k A)`.
Refining from 8 to 16 to 32 cells reduces its error by a factor of four at each
step. This verifies the second-order spatial discretization for that smooth
case; it does not verify the assumed device geometry.

For the declared two-ring slab, this implementation produced:

| Cells | Self heater gain (K/W) | Neighbor heater gain (K/W) | Self optical-heat gain (K/W) |
|---:|---:|---:|---:|
| 256 | 5185.703 | 65.57793 | 5481.858 |
| 1024 | 5279.301 | 65.34641 | 6122.833 |
| 4096 | 5333.279 | 65.16094 | 6489.333 |
| 16384 | 5339.870 | 65.13276 | 6687.921 |
| 65536 | 5339.041 | 65.13429 | 6643.668 |

The finest pair differs by 0.016% for self-heater gain, 0.67% for narrow-source
optical heating, and 0.50% in the Frobenius norm of the entire gain matrix.
The finest mesh is a reference, **not an exact solution**. Source/interface
resolution can make convergence nonmonotonic. The worst full-mesh steady power
balance residual is 1.2e-11 relative. A 1 mW heater transient sampled through
1 ms differed from the 4096-cell reference by at most 0.1493 K on 256 cells and
0.0551 K on 1024 cells; time integration used `rtol=1e-7, atol=1e-8 K`.

## Reduced models and their measured limits

`modal_reduce(k)` keeps the k slowest generalized diffusion modes.
`krylov_reduce(n_moments)` uses symmetric projection onto inverse-diffusion
input moments. It captures each source's static solution while retaining a
positive definite projected diffusion operator. This is the RC specialization
of the projection/moment-matching idea described in the primary
[PRIMA paper](https://cseweb.ucsd.edu/classes/fa04/cse245/Reading/prima.pdf);
it is not a full PRIMA RLC implementation. SciPy supplies LU, QR, and eigenvalue
algorithms. The projected coordinates are not physical nodal temperatures.

Against the 4096-cell spatial model, a 1 mW heater step sampled at zero and
60 logarithmic times between 10 ns and 1 ms gave:

| Krylov state count | Maximum observed step error (K) | Full DC gain-matrix relative error |
|---:|---:|---:|
| 6 | 0.73685 | 2.9e-13 |
| 12 | 0.17432 | 2.9e-13 |
| 24 | 0.01882 | 2.9e-13 |
| 48 | 0.000919 | 2.9e-13 |

Good DC agreement alone therefore says little about transient accuracy.
Testing all six individual 1 mW source steps with 48 states gives a worst
observed error of **0.04797 K for an optical source**. Truncation also produces
small unphysical neighbor-temperature undershoots (as low as −0.005565 K).
They are reported, not clipped or interpreted as cooling. These ROMs preserve
stable dissipative dynamics but **do not preserve pointwise positivity** and
are therefore not the default control plant. The full finite-volume and
physical RC networks retain that property. Reduced-model step helpers assume
a constant uniform boundary and zero initial rise; they do not support an
arbitrary time-varying ambient through an implicit offset.

Reproduce the numerical evidence and save full DC gain matrices and metrics with:

```sh
uv run python examples/thermal_extraction.py --max-refinement 16
```

The default refinement cap is 8; 16 reaches 65,536 cells. Output is
`artifacts/thermal_extraction.json`, explicitly labeled numerical verification
of assumed geometry, with numerical-library versions and SHA256 hashes of the
thermal implementation files. It is not a measured-device validation record.

## Calibration using actual measurements

`ThermalStepTrace` requires increasing time samples, a measured temperature
rise, known source power, an acquisition identity, and provenance. The input
assumption is equilibrium at a constant uniform boundary followed by a
zero-to-constant power step at t=0. Temperature readout dynamics, source rise
time, unknown offsets, nonuniform initial conditions, or drifting ambient
must be resolved or separately modeled before using this simple fitter.

`load_step_csv` accepts `time_s,temperature_k` columns plus explicit baseline
temperature and source watts. It does not infer units or heater power from
control voltage. For example, with **user-provided** measurement files:

```python
from ringlab.calibration import load_step_csv, fit_single_pole

train = load_step_csv("measured_step_A.csv", power_step_w=[0.001],
                      baseline_temperature_k=300.0, dataset_id="acquisition_A",
                      provenance="replace with instrument, setup and date")
heldout = load_step_csv("measured_step_B.csv", power_step_w=[0.0004],
                        baseline_temperature_k=300.0, dataset_id="acquisition_B",
                        provenance="replace with independent acquisition details")
result = fit_single_pole(train, heldout)
print(result.metrics())
```

Those filenames and watt values illustrate the API; no such measured data are
supplied. Replace them with independently established values. Fitting uses
training only; the heldout acquisition is used only for RMSE and maximum-error
reporting. Reusing the same acquisition identity is rejected, though the caller
is responsible for honest identities and independent acquisition.

`fit_single_pole` returns positive `C=tau/R` and `G=1/R`. For a supplied
multi-node topology, `fit_network_scales` identifies positive global capacity
and conductance scale factors while preserving topology and reciprocity.
These two scale factors cannot recover unknown geometry or individual
couplings from a scalar trace. The Jacobian condition number is reported as an
identifiability diagnostic, not a confidence interval. A poor heldout match
should trigger richer input/temperature measurements or a revised topology.
The optimizer is [SciPy bounded nonlinear least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

For the integrated control plant, calibrate the **named multi-node topology**
rather than substituting the standalone one-pole example. For a single ring,
the measured heater step has source vector `[heater_watts, 0, 0]`:

```python
from ringlab.calibration import fit_network_scales, load_step_csv
from ringlab.config import load
from ringlab.plant import simulate
from ringlab.thermal import default_ring_network

training = load_step_csv("measured_step_A.csv", power_step_w=[0.001, 0, 0],
                         baseline_temperature_k=300.0, dataset_id="acquisition_A",
                         provenance="replace with actual acquisition details")
validation = load_step_csv("measured_step_B.csv", power_step_w=[0.0004, 0, 0],
                           baseline_temperature_k=300.0, dataset_id="acquisition_B",
                           provenance="replace with independent acquisition details")
config = load(overrides={"array.count": 1})
fit = fit_network_scales(default_ring_network(1, config.thermal), training, validation)
assert fit.solver_success
run = simulate(config, thermal_network=fit.network)
```

Source, heater-node, package-node, and ring-observer names are preserved by this
fit, so the calibrated network can enter the full nonlinear optical/heater/
sensor simulation. Assess heldout error before using the result for research;
optimizer success alone is not a validation criterion. The standalone
one-pole fit and raw spatial mesh have different node semantics and cannot
simply replace this named plant topology. A synthetic test generates traces
with an independent ODE solve, identifies the scales, injects the returned
network, and checks that the integrated plant agrees with the known network.

Synthetic unit tests recover known RC parameters and deliberately check that
bad heldout data do not influence a good training fit. They verify estimation
logic, not the accuracy of a Lightmatter thermal plant. To establish that
accuracy, collect multiple power amplitudes, time scales, initial/boundary
temperatures, and spatial source/observer combinations, then reserve separate
acquisitions for validation.
