# Independent comparison with the permitted textbook

The permitted `interactive-textbooks/lightmatter-capstone` is an educational reference and source of questions. Its generation labels are useful. Its numeric defaults and helper functions were read, checked against primary sources and independent algebra, and not copied into this implementation. This document does not modify that repository.

Relevant snapshots are identified in [SOURCES.md](SOURCES.md). `src/physics/index.ts`, the source-generation ledger, and the relevant source statements in chapters 5, 10, 13, and 18 were inspected. This is a scoped comparison of the model's physics and evidence, not a claim to have re-audited every chapter or external source in the textbook.

## Agreements and corrections

| Textbook concept or claim | Independent assessment | Decision in this project |
|---|---|---|
| Separate device generations, links, products, and laser systems. | Correct and essential. The supplied P26 paper and current EVK50/L20 pages describe different rate and power accounting. | Preserve source-specific labels; do not assemble a fictitious device from every favorable number. |
| 400 GHz per direction and 200 GHz composite spacing for P26. | Directly confirmed in P26 Sec. 2. | Frequency-domain channel planning; periodic alias checks are necessary. |
| 15 GHz/K corresponds to about 86 pm/K at 1310 nm. | Correct local magnitude conversion. Heating normally red-shifts wavelength and lowers frequency. | Use signed frequency slope as the independent effective input. |
| The educational `Γ dnSi/dT` estimate gives about 50 pm/K using assumed group index 4.2. | Correct arithmetic for that approximation. It does not validate those mode parameters for Lightmatter. | Do not replace the public device sensitivity with this assumed estimate. |
| The reported 86 pm/K is impossible because the inferred effective-index coefficient exceeds bulk silicon. | Too strong. The confinement-weighted bulk approximation is not a general Maxwell perturbation bound; effective-index derivatives depend on modal fields, normalization, dispersion, and geometry changes. The data also do not establish the temperature reference or actual group index. | Record a mismatch between assumptions and the reported effective slope; leave its physical decomposition unresolved. |
| Use a single all-pass ring to teach a notch. | Correct for a one-bus ring. P26 explicitly monitors an MRM drop port, and public T26 slide 7 depicts an add-drop structure. | Use separate input and drop couplings, through/drop outputs, internal loss, and a separate absorbed fraction. |
| `fwhmNm` is described as all-pass or symmetric add-drop while using `a*t`. | For an add-drop ring the round-trip feedback product is `a*t1*t2`; with identical individual self-couplings it is `a*t*t`. The helper can serve add-drop only if its `t` argument already means the product, which its general wording does not establish. This follows the [Bogaerts formulas](https://www.photonics.intec.ugent.be/download/pub_3105.pdf). | Derive linewidth from the same couplings/loss as the spectrum; never enter an independent inconsistent Q. |
| The textbook Lorentzian widgets show local line shapes. | Appropriate for local intuition, but omit periodic resonances and can be inaccurate far from a line or at low finesse. | Use a periodic transfer function for spectral/WDM behavior and explicitly limit single-mode dynamic reductions. |
| A 1.8 THz FSR can serve as a ring reference. | The textbook itself attributes it to a 2025 target and assumes `ng=4.2`. P26 does not give FSR. Eight 400 GHz-spaced lines span 2.8 THz. Neither proves the actual 2026 bank uses that FSR or is inconsistent. | Do not transfer the value to the 2026 link. Any synthetic geometry carries its own derived FSR and collision consequences. |
| Approximately 15 µm in the design context is a pitch. | T26 slide 7 explicitly labels diameter. It establishes neither center-to-center pitch nor thermal separation. | Ring pitch and thermal coupling stay unknown; synthetic array spacing is not a manufacturer dimension. |
| 808 K/s heater, laser-sweep equivalence, and package TEC cycling can define one disturbance. | These excite different physical paths. P26 confirms an adjacent local resistive heater and a separate TEC experiment. Laser detuning does not by itself inject thermal energy. | Expose separate heat-source, boundary-temperature, laser-frequency and laser-power inputs. |
| The strongest published stress is above 2000 K/s. | Independently confirmed as a claim on public T26 slide 10, distinct from P26's 808 K/s setup. The slide does not supply a raw waveform or plant identification. | Additional stress context, not a fitted time constant or exact copied waveform. |
| A 25-105 °C, roughly 30.5 h sweep is the paper's slow test. | P26 text/caption state 35-105 °C for 31 h. Current vendor page gives a separate 25-105 °C validated range. | Use the paper's values when describing its experiment. |
| A 5 pm error budget implies 29 µs lag at 2000 K/s. | Correct conditional arithmetic using about 86 pm/K. It is not a measured hardware timing specification and assumes an uncompensated delay model. | Report chosen error budget and measured simulated lag separately from manufacturer evidence. |
| Heater resistance near 56 Ω, 16-bit monitor/DAC, 15 µA current, and 4.7 K/mW can all be treated as measured parameters. | These have different evidentiary status. P26 does not disclose heater I-V or converter resolution. A code axis does not prove resolution. The textbook's 4.7 K/mW combines a tuning-power/FSR calculation and a different effective thermal slope; it is an inference with unresolved transfer between conditions. | Keep runtime values assumed unless the specific source and extraction are independently supplied. No blanket adoption of the textbook parameter object. |
| Nonquadratic heater tuning establishes a unique temperature coefficient of resistance. | Not identifiable from tuning alone: temperature-dependent thermal transport, optical sensitivity, and resistance can produce similar curves. | Separate heater electrical law, heat transport and resonance shift; request current as well as wavelength measurements. |
| The semiconductor helper applies `sign(Δp)*abs(Δp)^0.8` for removed holes. | An odd extension of a nonlinear concentration law is a chosen approximation, not a general differential update at nonzero baseline. If an index law is `n(p)`, a depletion update must use `n(p_after)-n(p_before)` over nonnegative densities. | Use a plainly assumed compact depletion law or bounded measured voltage tables; do not label signed-power extension as TCAD. |
| Interpolated textbook measured curves can be clamped outside their sampled domain. | Clamping is useful for a widget, but a flat tail hides lack of knowledge and can falsely remove actuator or junction slope. | Imported junction tables reject extrapolation; outside-domain operation must be explicitly modeled. |
| Holding a monitor maximum establishes the best modulation bias. | A drop peak identifies resonance for a static state. Finite NRZ swing and cavity memory mean the maximum of average monitor power need not maximize through-port OMA. Symmetric intensity also has side ambiguity. | Treat controller objective, capture side and reference measurement as explicit choices. Monitor lock is not a BER guarantee. |
| A thermistor can reveal the temperature needed by an MRM lock. | It measures its own location and bandwidth. Local ring/heater/package temperatures can differ, especially during fast local heating. | Quantized delayed boundary/package observation may be available; exact optical detuning remains simulation truth, not controller I/O. |
| Learning curves for stabilized relOMA predict the physical loop. | The helper directly constructs an assumed stabilized map; it does not simulate an actuator, thermal state or feedback computation. | Evolve the plant and controller. Treat published relOMA contours as validation evidence only when measurement conditions are known. |

## Independent equations used to resolve ambiguities

From the resonance condition `β(ω,T)L(T)=2πm`, differentiating at fixed longitudinal order gives

```text
dλr/dT = (λr/ng) [ ∂neff/∂T + neff αL ],
ng = neff - λ ∂neff/∂λ,
df/dT = -(c/λr²) dλr/dT.
```

`αL` is the fractional length expansion coefficient. A measured effective `df/dT` already incorporates the device's net response under its measurement conditions; adding an extra expansion term to that measured slope would count it twice. A mode calculation may instead predict each term, but its geometry, material dispersion and temperature dependence must be supplied and converged. The [interferometric extraction study](https://www.photonics.intec.ugent.be/download/pub_3594.pdf) illustrates that these are experimentally identifiable modal quantities.

For two lossless couplers and internal power attenuation `α`, define `t1=sqrt(1-K1)`, `t2=sqrt(1-K2)`, and round-trip field attenuation `a=exp(-αL/2)`. The cavity feedback magnitude is `a*t1*t2`. Intrinsic optical loss includes absorption and scattering; only a specified absorbed portion heats the modeled device. Counting all missing through power as ring heat would also count light exiting the drop port.

A passive thermal network uses absolute temperature states:

```text
C dT/dt = -G T + B P + H Tb.
```

An ambient boundary step changes heat flow, not every node temperature instantaneously. Internal conductances are reciprocal and source deposition conserves watts. A positive heater can add heat but cannot actively cool; following a rising background temperature requires reducing prior heater bias and eventually encounters zero-power saturation. No control algorithm can recover absent physical headroom.

The [dither-lock paper](https://www.ee.columbia.edu/~kishore/publications/JLT_2014.pdf) supports dither as a valid independent control approach. It does not disclose Lightmatter's controller, and this project makes no claim that any implemented baseline reproduces it.

## What remains unresolved

The actual 2026 geometry, FSR, group index, voltage-dependent optical loss, heater characteristics, thermal cross-coupling, optical absorption split, monitor front end, raw thermal waveforms, and proprietary controller remain unidentified. The [parameter inventory](PARAMETER_INVENTORY.md) records each modeled, importable, or deferred item. Adding equations for unknown effects without identifying their coefficients would increase complexity without establishing device accuracy.
