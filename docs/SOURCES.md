# Evidence and calibration ledger

Reviewed 2026-10-09. This project is an independently implemented, public-evidence-informed MRM control simulator. It is not a validated Lightmatter digital twin. Public demonstrations constrain the architecture and useful stress tests; they do not identify the internal plant. No proprietary design kit, raw measurement dataset, or fitted Lightmatter device model was supplied.

## Evidence vocabulary

| Label | Meaning | What it does not imply |
|---|---|---|
| Published fact | A primary source explicitly reports the quantity or architecture. | That another generation has the same value; that uncertainty is zero. |
| Demonstration | A source reports behavior of a specific device, setup, controller, and measurement. | A plant parameter, controller transfer function, or guarantee for all transients. |
| Derived / inferred | Arithmetic or physics applied to stated inputs. | A separate independent measurement. |
| Calibrated | Parameters fitted to identified data, with units, conditions, residuals, and validation. | Validation outside that dataset's domain. Synthetic recovery tests are not device calibration. |
| Assumed | A deliberately chosen scenario or unmeasured parameter. | A manufacturer specification. |
| Unknown | Required information not established by the reviewed evidence. | Zero, negligible, or absent. |

Configuration evidence is a provenance annotation, not a certification. An override remains unverified unless its source, measurement conditions, and calibration results are provided. Numerical tolerances and random seeds are experiment choices, not physical evidence.

## P26: supplied OFC 2026 paper

R. Anand et al., *16-Wavelength 800-Gbps Bidirectional Link over Single-Mode Fiber Using Microring Transceivers*, OFC 2026, paper M4B.3, three pages. The supplied PDF was read completely and all three pages were rendered and visually inspected. Its abstract contains a 2025 copyright line, while the conference header and article footer identify 2026; the latter identify the conference year. The file is not redistributed here.

Original supplied filename: `16-Wavelength_800-Gbps_Bidirectional_Link_over_Single-Mode_Fiber_Using_Microring_Transceivers (1).pdf`.

SHA-256: `00f23df4bdcd73d379bcbb6f3ae2923cbcc7edc9446a6eb92c2166b447562c4e`.

| Location | Directly supported | Correct modeling use |
|---|---|---|
| p. 1, introduction | Approximately 15 GHz/°C MRM thermal sensitivity. | Magnitude of a local effective frequency slope; reference temperature and extraction uncertainty remain unknown. |
| pp. 1-2, architecture | Eight TX MRMs per transceiver; RX has eight coupled-ring add-drop filters; interleaver joins directions. | TX array size and port architecture; do not replace coupled RX filters with one ring and claim equivalence. |
| p. 2, architecture | Two eight-line grids centered near 1310 nm, each with 400 GHz spacing, offset by 200 GHz. | Use frequency spacing, not a constant wavelength spacing; TX-bank and composite-fiber grids differ. |
| p. 1 and conclusion | 50 Gb/s NRZ per wavelength, 400 Gb/s per direction, 800 Gb/s aggregate per fiber. | Link demonstration rate. |
| p. 2, Sec. 3.1 | Separate 56 Gb/s PRBS9 TX eye and receiver characterization. | A different characterization condition from the 50 Gb/s link. |
| p. 2, Sec. 3.1 | 1 km SMF28, six connectors, approximately 3 dB chip-to-chip optical loss. | Link context; not a measured insertion loss for each ring. |
| pp. 2-3, Fig. 3(a) | Package TEC cycling from 35 to 105 °C over 31 hours; eight lanes below 1e-9 BER at 50 Gb/s PRBS9. | Long-duration package-boundary stress context; does not disclose the temperature of every thermal node. |
| p. 3, Fig. 3(b), Sec. 3.2 | Adjacent on-chip resistive aggressor; reported 808 °C/s, approximately 25 to above 105 °C. | Local power deposition scenario; this experiment actually heats the device. |
| p. 3, Sec. 3.2 | BER sampled every 2 seconds, reported near 1e-11; stable MRM drop-port monitor current. | Sampled evidence of control robustness, with limited temporal resolution. |
| p. 3, discussion | Laser absolute wavelength accuracy/stability requirement ±20 GHz for interleaver penalties. | Laser/interleaver budget; not a permissible MRM detuning or measured control error. |
| p. 3, discussion | Approximately 3 W total; 3.8 pJ/bit from SiPh plus SerDes, split approximately 1.7 and 2.1 pJ/bit. | Setup-specific power accounting; not per-ring heater power. |
| p. 3, discussion | Reflection and TX-to-RX backscatter require mitigation. | A declared limitation of a reflection-free TX-bank model. |

### P26 timing ambiguity

The text combines an approximately 80 K excursion, 808 K/s ramp, and a 100 ms period for repeated up/down ramps. A symmetric triangle with those endpoints and slope needs about 99 ms per leg and 198 ms per complete cycle. A 100 ms full-cycle triangle instead needs about 1600 K/s. A nonsymmetric waveform could differ, but the raw waveform is not provided.

Consequently, no exact triangle is called a reproduction of P26. A scenario must state whether it preserves the published slope, period, or a chosen heating-power waveform. For a local heater, the actual ring temperature and its maximum derivative are outputs; a triangular power command does not generally create a triangular temperature. These alternatives are useful tests, not resolution of the source ambiguity.

The 2 s BER sampling also cannot identify individual subsecond error bursts, detector bandwidth, loop delay, or a thermal pole. BER plots with very low values require counting window and error-count information to establish statistical confidence. No BER guarantee is inferred from stable simulated monitor current.

### Arithmetic derived from P26

Using exact vacuum light speed, `df/dT = -15 GHz/K` at 1310 nm corresponds locally to `dλ/dT = +85.8644 pm/K`. The sign follows the ordinary silicon thermo-optic red shift; the paper itself supplies the magnitude. The frequency derivative is the model's independent quantity, so the wavelength derivative changes with wavelength.

An 808 K/s ring-temperature ramp therefore implies approximately 12.12 THz/s resonance-frequency drift. A hypothetical 5 pm error budget would allow only about 72 microseconds of uncompensated delay at that rate. At 2000 K/s the same arithmetic gives about 29 microseconds. These are conditional lag estimates, not measured loop bandwidths or capstone acceptance requirements.

Eight equally spaced lines at 400 GHz span 2.8 THz. This does not reveal the device FSR. Every proposed geometry must check all periodic resonances against all laser lines; copying a different device's FSR is not a valid WDM verification.

## T26: independently inspected public presentation

Lightmatter's [April 1, 2026 presentation page](https://lightmatter.com/resource/high-bandwidth-optical-links-nikhil-kumar-senior-director-photonics/) links the [OFC 2026 slide deck](https://lightmatter.com/wp-content/uploads/2026/04/OFC-2026-16-lambda-Bidi_PAM4-Nikhil-Kumar.pdf). Relevant slides 7, 9, and 10 were rendered and inspected; the public deck retains a proprietary/confidential footer, so this repository links and summarizes it rather than redistributing it. Retrieved PDF SHA-256: `5018c7b005e27a712c88bc73f20db236d9589c0dbbeb1db267d508451a4d2442`.

Slide 7 depicts a roughly 15 µm diameter ring, resistive heater, depletion PN junction, and add-drop schematic. Slide 9 reports controlled operation over a 0-105 °C junction-temperature axis. Slide 10 identifies an on-chip thermal aggressor and claims robustness above 2000 °C/s, with stabilizer DAC traces. These support architecture and an additional stress envelope, not a measured heater I-V curve, ADC resolution, precise geometry, thermal matrix, or disclosed algorithm. They do not replace P26's specific 808 °C/s experiment. A DAC plot extending near 50,000 codes alone does not establish converter resolution or code-to-voltage mapping.

## Current public product context

The [EVK50 page](https://lightmatter.com/products/passage-evk50/) currently advertises 16-line bidirectional 800 Gb/s/fiber, 25-105 °C validation, 56 Gb/s NRZ, and 2.6 pJ/bit including laser. These values are vendor claims for that reference platform; the rate and power-accounting statements differ from P26. They are recorded separately rather than averaged.

The current [Passage L20 page](https://lightmatter.co/products/passage-l20/) specifies 212.5 Gb/s PAM4, 32 data fibers, and 6.4 Tb/s in each direction. It is a different product and is not the NRZ multiwavelength control target modeled here. Neither page supplies a calibrated MRM plant. The search was bounded to public Lightmatter device/link/control material relevant to this model, not a general news survey. Future event abstracts were not treated as completed measurements.

The current [L200/L200X page](https://lightmatter.com/products/l200/) also describes 3D chip-on-wafer optical I/O with PAM4 and 32/64 Tb/s variants. This newer product context reinforces the relevance of heat from a vertically integrated electronic die, but supplies no ring-level thermal network or control transfer function. Its advertised system capabilities are not imported into the P26 reference experiment.

## Physical and numerical references

| ID | Primary reference | Role |
|---|---|---|
| RING | [Bogaerts et al., Silicon microring resonators, 2012, author-hosted paper](https://www.photonics.intec.ugent.be/download/pub_3105.pdf), DOI 10.1002/lpor.201100017 | All-pass/add-drop scattering, group-index FSR, distinction between coupling and internal loss, nonlinear limitations. |
| DITHER | [Padmaraju et al., Wavelength Locking and Thermally Stabilizing Microring Resonators Using Dithering Signals, 2014](https://www.ee.columbia.edu/~kishore/publications/JLT_2014.pdf) | Independent experimental basis for dither locking; not evidence that Lightmatter uses the implemented algorithm. |
| TCMT | [Fan, Suh and Joannopoulos, Temporal coupled-mode theory for the Fano resonance in optical resonators, 2003](https://web.stanford.edu/group/fan/publication/Fan_JOSAA_20_569_2003.pdf) | Energy/port normalization and constraints of isolated-mode envelope dynamics; not an exact delayed full-ring propagator. |
| MODAL | [Dwivedi et al., Experimental Extraction of Effective Refractive Index and Thermo-Optic Coefficients of Silicon-On-Insulator Waveguides using Interferometers, 2015](https://www.photonics.intec.ugent.be/download/pub_3594.pdf) | Need for measured/modal effective-index temperature response rather than an assumed bulk coefficient. |
| COMPACT | [Ansys ring-modulator extraction workflow](https://optics.ansys.com/hc/en-us/articles/360042322794-Ring-Modulator) | Offline component extraction followed by circuit assembly; finite validity range of voltage and wavelength tables. |
| CML | [Ansys ring-modulator compact model](https://optics.ansys.com/hc/en-us/articles/360034710153-ring-modulator-CML-Compiler-Model) | Independent checklist of optical, electrical, thermal, and loading parameters; avoid double-counting electrical filtering. |
| TOOLS | [Tool survey](TOOLS.md) | Maintainer/vendor documentation for each solver's actual role and limitations. |

Equations and dimensional/energy checks are independently derived implementation constraints. A cited equation does not turn an assumed coefficient into measured data.

## Permitted contextual sources

The textbook and conversation establish the capstone objective: a simulation/FPGA model of the core photonic component for developing locking and thermal-control algorithms, with a path to additional components. They are secondary context and are not independent measurements. No code, design, research note, or parameter from another `mrm-model` worktree was used.

| Context source | How used | SHA-256 where applicable |
|---|---|---|
| `interactive-textbooks/lightmatter-capstone/docs/SOURCES_AND_GENERATIONS.md` | Generation map and claims to independently verify; disagreements in [TEXTBOOK_COMPARISON.md](TEXTBOOK_COMPARISON.md). | `e7cc6f7135b94f36a9ee7d97ceb37f03472b08d1165d6b361150b3c190520e4d` |
| `interactive-textbooks/lightmatter-capstone/src/physics/index.ts` | Read-only comparison of educational assumptions and formulas; no copied implementation. | `944071e3ff6d71b3302d0989f83f856050281135a9036d619457cc8d451ffaf5` |
| `photonics-simulations/TOOLS.md` | Read-only inventory of prior learning tools; installed software and old experiment outputs are not evidence of this project's validation. | `3d3d99665c3ff7d4a7c9382d450b94cc997606d463ba3c783df673b8f70de85e` |
| Claude chat `f908ab9f-5e32-4639-970c-b0bfb5469b30.jsonl` | Read-only user intent and educational context. Embedded assistant assertions and workflow reports are not primary evidence. | Not redistributed. |
| Supplied `pasted-text-1.txt` | Requirements checklist and earlier proposal to assess, not a validated model or source of numerical truth. | `b977bfd0be7003d0983c04c405bad7a879f0507cc3a0ad6ba945d54960af4290` |

## What would establish device accuracy

1. Measure through and drop spectra across several temperatures, junction biases, heater settings, and input powers, with calibrated optical reference planes. Constrain FSR/dispersion, both couplings, and loss. Even two-port power spectra can leave a discrete coupling/loss ambiguity; complex phase or an independently characterized coupler/loss measurement may be needed to select a physical branch. See [optical calibration](OPTICAL_CALIBRATION.md).
2. Measure heater voltage and current, local temperature or calibrated resonance shift, and heater/aggressor step responses independently. Fit a passive thermal network to multiple excitations and validate on held-out transients.
3. Measure junction C-V, I-V, series parasitics, complex electro-optic response, and data dependence. Imported curves need the operating range and reference bias; a bandwidth scalar cannot identify chirp and driver behavior.
4. Measure monitor responsivity, collection loss, dark-current versus temperature, noise spectrum, analog bandwidth, sampling/filter timing, quantization, and saturation. Preserve raw timestamps and calibration uncertainty.
5. Validate complete closed-loop behavior with the same observable signals and actuator constraints, including capture, loss of lock, neighboring-ring heating, laser/power steps, and saturation. Reserve independent optical/BER measurements for validation rather than giving them to the controller as hidden state.

Until those measurements exist, parameter sweeps and synthetic calibration tests quantify sensitivity and numerical behavior; they do not quantify error relative to a Lightmatter device.
