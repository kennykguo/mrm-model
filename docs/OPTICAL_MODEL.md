# Optical, junction and data-rate models

These modules were derived independently for this implementation. They are a
transparent compact model, not a reconstruction of undisclosed Lightmatter
geometry or a claim of measured device accuracy. All quantities use SI units.

The supplied Anand et al., OFC 2026 M4B.3 paper constrains the operating band near
1310 nm, the approximate thermal frequency-sensitivity magnitude of 15 GHz/K,
50 Gb/s NRZ link operation, and an MRM drop-port monitor. Its 56 Gb/s eye uses
PRBS9. It does not identify the tested ring's linewidth, coupler strengths,
intrinsic loss, group index, junction response or absorbed/scattered loss split.
The negative thermal frequency slope used here follows the usual positive
silicon thermo-optic response; its sign is a physical inference, not a measured
signed coefficient in that paper. The default 10 um radius and other omitted
parameters are illustrative. A separate source-constrained preset can use the
approximate 15 um diameter in the public April 2026 Lightmatter presentation;
that still does not calibrate all properties of a particular fabricated device.

## Exact steady traveling-wave model

`optics.optical_response` uses two passive lossless point couplers, one forward
ring mode, uniform distributed intrinsic loss, and couplers separated by half a
roundtrip. The add port is unexcited. Set `drop_coupling_power=0` for a single-bus
all-pass ring. Excess coupler loss, different section losses, reflections,
backscatter-induced splitting, polarization conversion and nonlinear carrier
dynamics require an extended model. They are not silently represented by a
second free fit to the same notch.

For input/drop **power** coupling fractions K1/K2, field self-couplings
r1=sqrt(1-K1), r2=sqrt(1-K2), and intrinsic **power** loss alpha in Np/m, define
a=exp(-alpha L/2), the full-roundtrip field attenuation. Equivalently a is the
half-roundtrip power attenuation. With phase phi,

```
D = 1 - r1*r2*a*exp(-i*phi)
H_through = (r1 - r2*a*exp(-i*phi)) / D
H_drop = -sqrt(K1*K2*a)*exp(-i*phi/2) / D
P_circ_after_input / P_in = K1 / |D|^2
P_intrinsic / P_in = K1*(1-a)*(1+r2^2*a) / |D|^2
```

The complex transfer coefficients are input-field normalized; port powers are
their squared magnitudes times incident power. Coupler phase choices can change
the global drop-field phase without changing measured power. Each output power
is derived directly, and their sum satisfies

```
P_through + P_drop + P_intrinsic = P_in.
P_local_heat = absorption_fraction * P_intrinsic.
```

Unabsorbed intrinsic loss can represent scattering or radiation. The local
absorption fraction is independent of through-port extinction and cannot be
inferred from a transmission notch alone. A voltage-dependent loss delta is
added to alpha; their sum must stay nonnegative. The current implementation
applies one absorption fraction to that total loss. It therefore does not
separately assign a fully absorptive carrier-loss change on top of a partly
scattering baseline. If that distinction matters for optical self-heating,
separate measured/extracted absorptive and radiative loss channels are needed;
the present effective split must not be treated as a microscopic heat budget.

For local group delay tau_g, stored traveling-wave energy integrates the
attenuation along both half-rings:

```
U = P_circ_after_input * (1-a)/alpha * (1+r2^2*a) * tau_g/L.
```

The alpha=0 limit substitutes L/2 for (1-a)/alpha. This retains finite stored
energy in a lossless coupled ring, where estimating energy by dividing loss
power by zero would fail. For nondispersive propagation, intrinsic loss equals
alpha*(c/ng)*U.

These transfer relations agree with the all-pass/add-drop conventions in
[Bogaerts et al., Silicon microring resonators, 2012](https://photonics.intec.ugent.be/download/pub_3105.pdf).
The stored-energy integral above is evaluated explicitly from the assumed
distributed power rather than inferred from a Lorentzian fit.

## Resonance and dispersion

`resonance_frequency` uses a measured or assumed local frequency expansion:

```
f_res = c/lambda_ref + K_T*(T-T_ref) + K_T2*(T-T_ref)^2 + junction_shift.
tau_0 = ng*L/c
phi = tau_0*delta_omega + beta2*L*delta_omega^2/2
tau_g = tau_0 + beta2*L*delta_omega
```

The thermal slope already includes the observed sum of index and geometric
changes. Adding an expansion correction to the same measured slope would
double count that contribution. The curvature field is the coefficient of
delta_T squared, without an extra factor of one half. Constant coupling and
loss over wavelength are an assumption. Nonzero beta2 describes only a local
dispersion expansion and is rejected if it implies nonpositive group delay.

With beta2=0, FSR=c/(ng L). If rho=r1*r2*a, the exact intracavity/drop-power
full width at half maximum is

```
FWHM = (2*FSR/pi)*asin((1-rho)/(2*sqrt(rho))).
```

The function solves the dispersive phase equation on each side when beta2 is
nonzero. Some very heavily loaded rings do not have a resolved half-maximum
within an FSR; that case raises an error. Through-port half-depth and a fitted
Lorentzian width should not be substituted blindly in broad, distorted spectra.

## Junction interface

All voltages in `junction.py` are **positive reverse-bias magnitudes**. The
analytic baseline is an abrupt-depletion-shape model:

```
s(V) = sqrt(1+V/V_bi)
C(V) = Cj0/s(V) + Cparasitic
delta_f(V) = frequency_shift_scale * (s(V)-s(V_ref))
delta_alpha(V) = loss_change_scale * (s(V)-s(V_ref))
```

The square-root capacitance comes from the uniform abrupt PN depletion
approximation. The shift/loss coefficients summarize unknown mode overlap,
depletion geometry and carrier response. They are deliberately labeled
assumed fit parameters, not derived semiconductor solutions. Depleting carriers
usually increases the silicon refractive index and decreases free-carrier loss,
consistent with the default negative frequency/loss shifts. The coupling of
carrier density to optical properties is established by
[Soref and Bennett, Electrooptical Effects in Silicon, 1987](https://zenodo.org/records/1232209/files/article.pdf),
but its empirical density coefficients are not silently transferred between
1310 nm and 1550 nm or used without a modal-overlap calculation here.

`MeasuredJunctionTable` accepts voltage, resonance shift, additional loss,
capacitance, resistance and leakage-current columns, with mandatory provenance.
Interpolation is piecewise linear, preserving tabulated values without spline
overshoot. It never extrapolates. Arrays are copied and made read-only. Such a
table is only valid for its characterized wavelength and temperature; a 1D
voltage table does not establish temperature dependence or package impedance.
Replacing this baseline with DEVSIM or measured results is a normal calibration
step. The measured table interface does not imply measurements have been made.

## Fast optical dynamics and slow thermal averaging

`waveform.py` normalizes cavity amplitude q so that |q|^2 is energy in joules.
The input field s has |s|^2 in watts. Energy-decay rates match logarithmic
roundtrip losses:

```
k_i = alpha*c/ng
k_1 = -log(1-K1)/tau_0
k_2 = -log(1-K2)/tau_0
z = -(k_i+k_1+k_2)/2 - i*(omega_laser-omega_res)
dq/dt = z*q + sqrt(k_1)*s
s_through = s - sqrt(k_1)*q
P_drop = k_2*|q|^2
P_intrinsic = k_i*|q|^2
```

Thus energy decays at k_i+k_1+k_2 and amplitude at half that rate. The port
normalizations satisfy the instantaneous power balance including d|q|^2/dt.
This is the isolated-mode temporal coupled-mode framework of
[Fan, Suh and Joannopoulos, 2003](https://web.stanford.edu/group/fan/publication/Fan_JOSAA_20_569_2003.pdf).
It is not identical to the discrete roundtrip transfer function. Small
per-pass loss, small detuning relative to FSR and slow envelopes relative to
roundtrip time are needed. Tests compare the two models in that domain.
`tcmt_domain_metrics` exposes dimensionless loading and detuning diagnostics;
these are not guaranteed error bounds. TCMT does not include the beta2 term.

`waveform_response` includes voltage-dependent first-order RC dynamics and
complex cavity transients for a short user-specified waveform. A midpoint
exponential step handles the changing coefficients. Constant voltage and
constant RC are integrated exactly; varying coefficients require timestep or
substep refinement. Leakage heat is reported, but the tiny leakage current is
not included as a load in the RC voltage equation. Driver inductance,
transmission lines, frequency-dependent impedance, carrier lifetime and forward
injection are outside this baseline. Resistor heat is reported separately from
junction leakage heat because its location relative to the ring is unknown.
Temperature is held fixed over the short waveform; absorbed-power outputs are
available to a slower thermal solver.

The fast waveform is too expensive to repeat for every thermal timestep.
`nrz_average_response` therefore solves the stationary first and second moments
of independent identically distributed rectangular binary bits analytically.
For bit level j lasting Tb:

```
q_next = A_j*q + B_j,  A_j=exp(z_j*Tb), B_j=q_ss,j*(1-A_j)
mu = E[B] / (1-E[A])
M = (E[|B|^2] + 2*Re(E[A*conj(B)]*mu)) / (1-E[|A|^2])
```

Integrating the resulting field/energy moments over the bit yields average
through, drop and absorbed power with photon memory. The center-sample
conditional power difference is labeled an OMA proxy; it is **not** worst-case
eye opening, Q factor or BER. This averaging treats electrical edges as ideal
and bits as independent. A PRBS9 pattern is correlated and finite. Finite RC,
specified patterns and measurement bandwidth require waveform simulation and
receiver calibration.

For the illustrative default ring, 0.5/2.5 V levels and 50 Gb/s iid data at the
midpoint of their resonances, mean absorbed fraction is about 0.477 from the
dynamic moments, versus about 0.285 from quasistatic level averaging. Therefore
quasistatic averaging at high bit rate is not generally safe near resonance.
At a 400 GHz detuning, exact quasistatic absorption is about 1.031e-4 of input
power; single-mode TCMT predicts about 7.021e-5, while its calculated dynamic
correction is only 1.16e-8. This motivates using exact per-pass optics for far
WDM lines, with an explicit approximation that their modulation sidebands are
negligible. These numerical differences apply only to this assumed parameter
set, not to uncharacterized hardware. Wrapping a Lorentzian over a whole FSR
does not make its off-resonance tails exact. Across resonance collisions or
strong sideband excitation, a delayed traveling-wave model or converged
waveform table is needed.

`wdm_nrz_average_response` implements that explicitly limited wideband
approximation. It uses local dynamic moments within 0.08 FSR of the nearest
resonance and exact per-pass quasistatic powers beyond 0.12 FSR. Between those
thresholds a smoothstep convex blend preserves average power balance and
nonnegative powers without introducing a jump into the thermal ODE. A convex
blend is not a new exact physical law: threshold sensitivity is part of the
model uncertainty. Returned diagnostics include the static TCMT/per-pass
discrepancy and the TCMT dynamic correction for each line; neither is a bound
on the omitted dynamics. The current helper rejects nonzero beta2 because an
integer-FSR comb must not be used as a dispersive multi-mode spectrum.

No spontaneous backscatter or nonlinear absorption has been invented to fill
missing measurements. If measurements show split resonances, power-dependent
detuning, or carrier tails, fit an extended model against those observables
before comparing controller performance to device behavior.

## Screening omitted nonlinear effects

`optics.nonlinear_screening` requires the caller to supply effective mode area,
TPA coefficient and Kerr n2. No coefficient is defaulted from another wavelength
or treated as a measured Lightmatter value. These must use consistent modal
overlap conventions; the screening currently assumes one common effective area
for both nonlinearities. The scalar reduction and sensitivity to free-carrier
transport are discussed in
[Trocha et al., Physical Review A 103, 063515, 2021](https://doi.org/10.1103/PhysRevA.103.063515).

On the already-computed **linear** traveling-wave power distribution, it
evaluates

```
P_TPA = integral(beta_TPA * P(z)^2 / A_eff dz)
delta_f_Kerr = -f/ng * n2 * mean_z(P(z)) / A_eff
TPA electron-hole pair generation rate = P_TPA / (2*h*f).
```

It integrates both attenuating half-rings including the drop-coupler power
change. It returns estimated TPA relative to intrinsic loss and loaded decay,
and the estimated Kerr shift relative to linewidth. Carrier density is not
computed because material volume, lifetime, diffusion and junction extraction
are not supplied. Nor is all nonlinear absorbed power assigned instantly to
local heat.

These are perturbation diagnostics, **not guaranteed bounds or a nonlinear
simulation**. A nonlinear shift can bring an initially weakly excited resonance
onto the laser, so a small estimate at one linear point does not exclude another
branch. Significant ratios, carrier accumulation or power-dependent measured
spectra call for a self-consistent carrier/optical/thermal model. Multiple tones
can also interact nonlinearly; summing single-tone screening outputs does not
include cross-phase modulation or nondegenerate TPA.

## Optional validated spectral acceleration

`optical_tables.build_nrz_average_table` tabulates the same hybrid NRZ equations
over one periodic FSR. Its input is detuning from the thermally shifted
zero-junction resonance; the table includes the specified two junction levels.
It can therefore be shared by rings whose reference wavelengths differ, if
their group delay, loss, coupling and data/junction parameters are otherwise
identical. Incident optical power remains a multiplicative factor because the
underlying model is linear in optical power.

Adaptive intervals are tested at quarter, midpoint and three-quarter locations.
An independent random set then checks interpolation after refinement. If random
checks fail, those points become refinement data and a fresh independent set is
used. Exhausting the node/validation budget triggers direct-equation evaluation,
not a relaxed tolerance. The diagnostic report includes node count, memory,
sampled errors and random seed. Linear convex interpolation of all port powers
preserves average power balance and nonnegative powers. Stored energy is scaled
by the loaded decay rate during tolerance checks so its small SI magnitude does
not hide relative errors. OMA is a difference of two interpolated conditional
powers and can accumulate both errors.

The default absolute tolerance is 1e-6 of input power. For the illustrative
0.5/2.5 V ring it generated 6108 nodes using approximately 0.73 MB, with a
2048-point independent maximum error of 9.79e-7. These sampled checks are not a
rigorous uniform bound. Repeat simulation at tighter interpolation tolerance
when its conclusions depend on small optical differences. This acceleration
validates only interpolation of the stated compact equations and does not
improve their agreement with an unmeasured device.

## What an open optical/carrier extraction workflow must supply

This repository has not executed a target-device optical mode, coupler FDTD or
carrier-transport extraction. Its executed spatial example is thermal. The
following is an optional workflow for replacing assumptions once geometry and
material/process inputs exist; it is not a record of completed solver runs.

1. Specify the full waveguide cross-section, layers, material dispersion,
   temperature, doping regions and boundary conditions with provenance. Solve
   vector modes with a free tool such as
   [Femwell](https://helgegehring.github.io/femwell/). Extract effective index,
   group index, field normalization and nonlinear overlap areas. Check mesh and
   domain convergence and, when group index is calculated by a wavelength
   derivative, also refine the wavelength increment. A real-index ideal
   waveguide calculation does not predict roughness, trap or metal absorption.
2. Extract bend and coupler behavior separately. A
   [Meep mode-decomposition calculation](https://meep.readthedocs.io/en/latest/Python_Tutorials/Mode_Decomposition/)
   can provide complex guided-port coefficients and radiation accounting.
   Record source normalization and port reference planes, and refine spatial
   resolution, domain/PML spacing and run time. Compare the resulting S-matrix
   with the current lossless scalar-coupler reduction. Significant reflected
   ports, excess loss, phase imbalance or wavelength dependence require a
   richer model; importing one best-fit scalar K would discard those effects.
3. With actual doping, contacts and transport/material parameters, use
   [DEVSIM](https://devsim.org/introduction.html) for bias-dependent Poisson and
   carrier-transport solutions, terminal charge, current and small-signal
   impedance. Check electrical mesh, contact boundaries and bias continuation.
   Femwell's listed PN depletion example is analytical; it must not be
   described as a full drift-diffusion calibration.
4. Convert the carrier-density fields to optical index/loss changes using a
   wavelength-appropriate material law and the vector-mode overlap. Evaluate
   the difference between the two nonnegative carrier populations at each
   location. A signed fractional power of the removed density is not generally
   the difference of a nonlinear concentration law. If geometry or depletion
   changes the mode substantially, recompute the mode rather than holding one
   overlap factor fixed. This optical/electrical field-to-table adapter is not
   implemented in the current repository.
5. Export voltage, resonance-frequency shift relative to a declared bias,
   additional power loss in Np/m, differential capacitance, relevant series
   resistance and leakage current to `MeasuredJunctionTable`. An extracted
   table belongs to one characterized wavelength/temperature domain; generate
   additional domains or extend the table schema when necessary. DC contact
   resistance is not automatically the high-frequency driver impedance.
6. Validate the compact model against heldout spectra, C-V/I-V/S-parameters,
   finite-rate waveforms and measured temperature conditions. Preserve the
   solver input, software version, mesh/domain checks, normalization, units and
   residuals. Agreement of two solvers given the same assumed cross-section is
   a numerical/model cross-check, not a measurement of Lightmatter geometry.

The source methods and quantities available here provide a path to that
workflow. No executable Meep/Femwell/DEVSIM backend, arbitrary vendor file
importer or geometry-to-junction pipeline is claimed by the current code.

## Numerical checks

The tests cover all-pass closed forms, add-drop energy balance, generalized
critical coupling, distributed energy, exact linewidth, FSR periodicity,
thermal/depletion signs, dispersion derivatives, table bounds, high-Q TCMT
convergence, photon-lifetime transients, exact RC steps and waveform timestep
refinement. The stationary NRZ moments are checked against enumeration of
4096 independently evolved bit histories and numerical integration of the next
bit, as well as average power conservation. These verify equations and
implementation; they do not replace physical device calibration.

The executable [optical accuracy example](../examples/optical_accuracy_checks.py)
records additional numerical evidence in
[artifacts/optical_accuracy.json](../artifacts/optical_accuracy.json). Run it with
`uv run python examples/optical_accuracy_checks.py`. The artifact includes exact
source hashes, parameters, seeds, output windows, runtime and process memory.

For its matched 128-bit finite-RC pattern, successive 1-to-2, 2-to-4 and 4-to-8
substep changes in the through waveform were approximately 2.32, 0.542 and
0.129 uW at 1 mW input, consistent with convergence of the midpoint method.
These comparisons refine internal integration while holding output sampling
at 16 samples/bit; the sampled time-average quadrature is a separate numerical
approximation. The final mean-heat change was about 4.37 nW. This is one
specified waveform/bias, not a worst-case error bound.

Two independent 256/1024-bit patterns compare ideal electrical edges and
finite RC after discarding 32 initial bits. The artifact records empirical mark
and transition probabilities. In those runs, finite-window mean heating varied
more than the paired RC correction. The center-conditional OMA estimates also
varied substantially because the selected midpoint bias has a small stationary
conditional level difference. Those short windows are not statistically
converged measurements of an infinite iid process; two seeds do not define a
confidence interval, and increasing window length need not monotonically
reduce the error in every random realization.

A separate 10,000-point random holdout, with a seed unused during table
construction, checked the 1e-7 interpolation setting over six FSRs and varying
input powers. Maximum through/absorbed errors were approximately 4.41e-8 and
2.66e-8 of input power; energy scaled by loaded decay differed by 9.88e-8.
The OMA difference reached 1.88e-7 because it combines two conditional powers.
Average power conservation held within approximately 9.81e-16 of input power.
These are sampled numerical discrepancies against direct compact equations,
not discrepancies against hardware. The recorded run used about 119 MB peak
process RSS and completed in about 35.5 seconds on this environment.
