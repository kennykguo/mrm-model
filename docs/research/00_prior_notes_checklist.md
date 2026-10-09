# Prior notes Kenny pasted (from another assistant), condensed to a checklist

These notes came from a different assistant and are NOT verified. Use them only as a completeness checklist: every item
below must be either modelled, deliberately excluded with a reason, or listed as unknown in docs/PARAMETERS.md.
Claims about tools, prices and papers in those notes must be re-checked before use.

Recommendations in the notes: build a control-oriented compact model yourself (Python), extract coefficients with
established solvers rather than writing solvers; two operating modes (thermal-control mode with quasi-static optics,
waveform-validation mode with cavity dynamics); calibrate against measured Lightmatter data; separate parameters,
inputs and states; derive dependent quantities from independent ones (Q vs linewidth, FSR vs length vs n_g); multi-pole
thermal model fitted to FEM or step responses; MIMO thermal crosstalk via impulse responses h_ij(t); model the full
control-hardware path (DAC, driver, thermal, optics, monitor, AFE, ADC, latency); test robustness over parameter
uncertainty sets with correlations; reduced-order thermal models (BCI-ROM idea) as an alternative to sharing geometry.

## Inventory (sections A to K of the notes)

A. Geometry and materials: ring radius / round-trip length; bus-ring gap; coupling length and coupler geometry; waveguide
width, height, cross-section type (rib/strip/slab); bend geometry; Si index vs wavelength; Si thermo-optic coefficient;
cladding and BOX geometry and indices; metal/heater geometry; doped junction geometry; fabrication width/thickness
variation; sidewall roughness; layer alignment/process variation; thermal expansion; stress-optic effects.

B. Optical resonator: reference resonance frequency; n_eff; n_g; FSR; intrinsic loss; bus coupling; loaded/intrinsic Q;
coupler phase and wavelength dependence; linewidth; extinction/notch depth; through- and monitor-port transfer
functions; group delay; backreflection/backscattering; mode splitting; polarization dependence (TE/TM); higher-order
transverse modes; coupling between adjacent resonators; intracavity energy.

C. Thermal: thermal resistance; capacitance; time constants; resonance shift per kelvin; reference temperature; heater
power to resonance shift; heater-to-ring delay; inter-ring thermal coupling; multiple thermal modes; spatial gradients
across the ring; package boundary conditions; T-dependent thermal conductivity and heat capacity; optical absorption
heating; data-driver heating; external chip/package heat sources; heat sink/cooling response; TEC dynamics;
device-to-device thermal variation.

D. Heater and actuator: resistance or I-V curve; resistance vs temperature; driver output limits; DAC resolution; DAC
gain/offset error; output noise; command latency; settling time; update rate; slew limit; supply voltage and ripple;
PWM frequency/duty/ripple (if PWM); compliance and safety limits; electrical crosstalk; heater aging/drift;
shutdown/fallback behaviour; whether the command is voltage, current, power or a DAC code.

E. High-speed modulation: mechanism (depletion/injection); drive levels; bias and polarity; junction C(V); junction
resistance and parasitics; carrier concentration vs bias; dn_eff/dV; dalpha/dV; modulation bandwidth; driver output
impedance; package RLC; data pattern and transition density; rise/fall times; driver jitter and noise; self-heating
from data activity.

F. Laser: centre wavelength; power; frequency drift; absolute accuracy; RIN; linewidth/phase noise; power noise;
polarization state; tuning rate (if controllable); laser thermal/electrical dynamics; input coupling loss; reflections
into the laser; multiple WDM carriers.

G. Monitor and ADC: monitor placement/topology; responsivity; dark current; shot noise; electronic noise; monitor
bandwidth; ADC resolution; ADC reference/gain/offset; ADC sample rate; measurement latency; analog filtering;
saturation/clipping; temperature-dependent monitor response; monitor crosstalk; dropped samples; calibration tones and
dither; sensor synchronization.

H. Nonlinear optics: linear absorption heating; TPA; free-carrier generation; FCA; free-carrier dispersion; carrier
lifetime; Kerr; thermo-optic bistability; resonance hysteresis; backscatter/counter-propagating modes; operating-point
dependent resonance shape.

I. Multi-ring / WDM: number and order of rings; ring-to-wavelength assignment; per-ring spectra; per-ring heaters and
offsets; common-bus loss and phase; inter-ring thermal and optical coupling; wavelength-dependent couplers; reflections
between components; RX add-drop filters; interleaver transfer functions; TX/RX grid offsets; interleaver IL,
isolation, dispersion; RX photodiodes and TIAs; polarization rotation/PDL; bidirectional interference/backscatter;
fibre/connector loss and reflections; channel crosstalk; laser source correlation and drift; die thermal variation;
multi-controller synchronization and sequencing.

J. Link quality: modulation format; OMA; ER; IL; eye height/width; rise/fall; received power; receiver noise and
sensitivity; jitter; ISI; crosstalk penalty; pre-FEC BER; BER observation interval; error-counter latency.

K. Initial conditions and simulator behaviour: initial temperatures and heater codes; initial optical/electrical
states; initial resonance offsets and lock status; ambient temperature vs time; package/board temperature vs time;
spatial external heating; laser drift vs time; supply disturbances; device-to-device variation; process corners and
correlated errors; noise distributions; sample rates and event timing; integration step and tolerances; random seeds;
fault injection (lost samples, saturation); parameter validity ranges; uncertainty and extrapolation warnings; test
configurations and expected reference outputs.

The notes also listed what the OFC 2026 paper gives (about 15 GHz/degC, 1310 nm, 8 TX rings, 400 GHz grid, 200 GHz
offset, 50 Gb/s NRZ, 808 degC/s local heater ramps, 35-105 degC over 31 h) and what it does not (heater resistance,
limits, time constants, geometry, indices, coupling, spectra, junction, monitor ADC details, controller, crosstalk,
production distributions). Kenny's textbook covers several of those from other Lightmatter sources (G25 talk/paper,
OFC 2026 slides, Hot Chips) - check REF in the textbook before calling anything unknown.
