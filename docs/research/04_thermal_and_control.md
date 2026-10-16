# 04 - Thermal dynamics data and MRM locking/control literature

Status: research note for the `mrm-model` plant and controller baselines. Written 2026-10-09.
Scope: (A) measured thermal dynamics of microring heaters and crosstalk, (B) disturbances, (C) locking and control
algorithms with enough detail to implement baselines.

## 0. How to read this file

**Provenance words** (same vocabulary as the textbook, `docs/SOURCES_AND_GENERATIONS.md`):

| Word | Meaning here |
|---|---|
| printed | verbatim number in the cited source (abstract, table, text) |
| read-off | digitized from a figure |
| inferred | arithmetic on printed numbers (unit conversion, 10-90 % rise time to time constant, R x C) |
| derived | computed by us from a model or several sources combined |
| assumed | no source |

Extra qualifiers used below: **simulated** (the source prints a finite-element or circuit-simulation result, not a
measurement), **abstract-only** (only the abstract or a citing paper was readable, full text not checked),
**secondary** (number comes from a citing paper, press or blog, not from the original).

**Lightmatter generation tags** (from the textbook): G25 = OFC 2025 Tu3J.6 single-wavelength test vehicle;
G26 = OFC 2026 M4B.3 16-wavelength bidirectional link; L20 = Passage L20; M1000 = Passage M-series; Guide = laser module.
Lightmatter numbers marked `REF.x` come from the textbook's parameter table
`/Users/kenny/interactive-textbooks/lightmatter-capstone/src/physics/index.ts` and keep its provenance word.

**Conversions used everywhere (inferred):**

- c / lambda^2 = 174.7 GHz/nm at 1310 nm, 176.3 GHz/nm at 1304 nm, 212.1 GHz/nm at 1189 nm, 124.8 GHz/nm at 1550 nm.
  [verified: 174.69 / 176.31 / 212.06 / 124.78 GHz/nm]
- Single-pole step response: t(10-90 %) = ln 9 x tau = 2.197 tau. Many papers print a "switching time" or "rise time";
  this is NOT the time constant. Every tau below says which one it is.
- One free spectral range (FSR) is a 2 pi round-trip phase shift, so P_FSR = 2 P_pi.
- Heater thermal resistance seen by the ring R_th [K/mW] = (dlambda/dP [nm/mW]) / (dlambda/dT [nm/K]).

---

## 1. Key findings in one page

1. **Measured fast time constants exist now, and the textbook's assumed 10 us is consistent with them.** imec/KU Leuven
   measured a metal (W) heater above a 15 um diameter C-band ring modulator: heater node tau = 5 us, waveguide
   tau = 10 us (no substrate undercut), and simulated 18.1 us / 38.1 us with a full undercut [CO22-JLT, Table III,
   Fig. 16]. A later imec paper converts measured ring responses to pure-thermal tau = 5 / 9 / 18 us for no / small /
   large undercut [CO24-PH, Table 2]. [verified against both author copies]
2. **The response is multi-pole, not single-pole.** A calibrated FE fit of an imec ring filter needs 2 poles without
   undercut (3.2 us and 13.0 us, 43 % / 57 % of the DC gain) and 4 poles with undercut (2.5, 25.6, 54.5, 124 us) -
   inferred from [CO22-TCPMT, Table I]. [verified: tau = R x C from the printed Table I gives 3.18 / 13.01 us and
   2.51 / 25.6 / 54.5 / 124.0 us, shares 42.7 / 57.3 % and 5.8 / 41.9 / 48.2 / 4.1 %. Nuance: the RC values are a fit
   to FE output; the FE material set is the one calibrated against measurement in [CO22-JLT] (their ref. 17), but this
   filter's step response was not itself measured.] Encapsulated waveguides relax roughly logarithmically in time (2D diffusion),
   i.e. a continuum of time constants [GRAY20, abstract-only].
3. **Crosstalk is slow and small but long-range.** Measured crosstalk between ring modulators at 100 / 200 / 300 um is
   1.3 % to 2.6 % (with an electronic die flip-chipped on top) and decays only slowly with distance [CO24-JOM,
   Table 3]. Simulated neighbour crosstalk arrives with first-order time constants of 4.1 to 9.0 ms, two to three
   orders slower than the self path [CO22-TCPMT, Table II]. [verified: Table II and Table 3 values exact] At 10 to 15 um spacing, microdisk crosstalk is "a few
   percent" [DER14]. The textbook's assumed 10 % nearest / 3 % next-nearest / 0 beyond, with the same dynamics as the
   self path, is not what the data show (Section 5).
4. **3D stacking hurts.** Flip-chipping an EIC on the PIC lowered ring-heater efficiency by 43.3 % and raised
   crosstalk by up to 44.4 % [CO24-JOM, abstract and Table 2]. [corrected: 43.3 % and 44.4 % are the paper's headline
   numbers (abstract, conclusions). The **measured** efficiency drops in Sec. 4.1.2 / Table 2 are 42.4 % with UCUT
   (312 -> 180 pm/mW) and only 20.6 % without UCUT (165 -> 131 pm/mW). The +44.4 % crosstalk is **simulated** (calibrated
   FE model run with and without the EIC, UCUT case, Sec. 4.2); crosstalk was measured only with the EIC on.] Lightmatter's M1000 stacks electronics on the photonic
   die, so this is the closest public analogue to the embargoed Lightmatter EIC-PIC crosstalk.
5. **Two vendors and one measurement agree on 62.5 pm/K for O-band rings.** Lightmatter (via the textbook's [OCPW])
   and Ayar Labs [AYAR25] both treat a 50 nm/s laser sweep as an 800 degC/s temperature ramp (62.5 pm/K, inferred),
   and imec measured 62.54 pm/K on O-band rings [CO24-JOM, Eq. 2]. Lightmatter prints ~15 GHz/degC (86 pm/K) for its
   own G26 MRM. Report both; do not merge. [corrected: [CO24-JOM] prints "62.54 pm/K for the rings measured in the
   O-band" under Eq. 2 but does not say how the 62.54 was obtained (measured, simulated or taken from elsewhere), so the
   provenance is printed, method unstated, not "measured". The 86 pm/K = 15 GHz/K x 1310^2/c is verified (85.9 pm/K).
   Other values seen in this check: 75.5 pm/K (thermistor) and 75.8 pm/K (oven) measured on C-band rings [HAT24].]
6. **Published industrial thermal transients:** ASIC 0 to 500 W step gives about 100 degC/s peak at an optical engine
   3 mm away, about 100 ms after the step; the XPU itself sees 400 degC/s [AYAR25, vendor blog]. [verified; the blog
   says the test "simulates" the step without saying whether physically or numerically, see D5] This agrees in
   magnitude with the textbook's [MRVL] 10 to 120 degC/s. Lightmatter's 808 degC/s (G26 paper) and >2000 degC/s (G26
   slides) are on-die heater aggressors, far above any published academic tracking result.
7. **Published academic tracking rates are slow:** 0.28 degC/s [JAY15], 3.1 degC/s (TEC-limited) [HAT24],
   5 nm/s ~ 62.5 degC/s [AGA16, abstract-only], and a 3 K, 10 Hz optical aggressor (94 to 188 degC/s peak slope,
   derived) [PTC13]. Industry: 800 degC/s emulated by laser sweep (Ayar Labs, Lightmatter), 808 and >2000 degC/s
   physical on-die (Lightmatter G26).
8. **Algorithm literature is rich for error-signal generation and initialization, thin for model-based control.** I
   found no peer-reviewed paper applying MPC, Kalman filtering or reinforcement learning to thermal locking of a
   silicon MRM. One 2024 SPIE paper from an automatic-control group (FAU Erlangen) is titled "Control approaches for
   operating point stabilization of microring resonator modulators under fast perturbations" [CHE24]; its content
   could not be read. Treat model-based control as an open niche for the capstone.
9. **Derived constraint for the Lightmatter case:** at 2000 degC/s x 86 pm/K = 172 pm/ms, a 1 kHz dither (loop
   bandwidth ~100 Hz) would lag by ~1.6 ms, i.e. ~270 pm (0.7 linewidth): dither at the published kHz rates cannot
   meet the G26 aggressor. A 1-LSB-per-decision integrator at Sun et al.'s 76 kHz decision rate slews ~76 codes/ms
   against ~400 codes/ms needed. Both are derived numbers (Section 4.0). [corrected: the dither arithmetic is verified
   (172 pm/ms, 1/(2 pi 100 Hz) = 1.59 ms, 274 pm = 0.73 x 374 pm). The slew need is understated: the textbook maps the
   29,800-code G26 command span onto the ~83 K excursion it covers (359 codes/K, ~0.24 pm/code at 86 pm/K), giving
   ~460 codes/ms half-cycle average, ~720 codes/ms at 2000 degC/s and ~830 codes/ms on the 2300 degC/s edge (textbook
   Ch. 11, inferred). The 1-LSB integrator falls short by ~6x to ~11x, not ~5x. See Section 4.0.]
10. **Platform hint (inferred, not established):** Passage is manufactured on GlobalFoundries Fotonix [DCD24]. The
    GF 45 nm PDK microring modulator described by GF engineers is a 7.5 um radius (15 um diameter) add-drop ring with
    ~5 % drop coupling and an **undercut** heater made of four silicide resistors [NAN23]. Lightmatter's G26 TX ring is
    add-drop with a drop-port monitor and Hot Chips prints a 15 um ring diameter (textbook). If Lightmatter uses a
    similar device, the thermal plant is undercut-like (tau tens of us, larger slow tail). This is a hypothesis to put
    to Lightmatter, not a fact.

---

## 2. Thermal dynamics data (Topic A)

### 2.1 The Lightmatter reference numbers (from the textbook, unchanged)

| Quantity | Value | Gen | Provenance (textbook) |
|---|---|---|---|
| Heater power per FSR, wafer median (60 dies, 25 degC) | 25.45 mW (range 24.9 to 34.1) | G25 | printed [LM-T] |
| Heater efficiency headline | 0.44 nm/mW | G25 | printed [LM-P] |
| Heater efficiency implied by median and FSR 10.3 nm | 0.405 nm/mW (= 14.1 uW/GHz, inferred here) | G25 | inferred |
| Heater electrical | ~55.9 Ohm, 1.195 V, 21.4 mA at one FSR | G25 | inferred / printed |
| Heater-to-ring R_th | 4.7 K/mW | G25 | derived (0.405 / 0.086) [verified against `REF`: 25.45 mW, 10.3 nm -> 0.405 nm/mW, 14.1 uW/GHz, 4.71 K/mW; 1.195 V / 21.4 mA = 55.8 Ohm] |
| Resonance sensitivity | ~15 GHz/degC = 86 pm/K | G26 | printed [LM26-P] |
| Thermal time constants | none published; textbook assumes 10 us fast + 300 us slow (25 % share) | - | assumed |
| Ring-to-ring crosstalk | none published; textbook assumes 10 % / 3 % | - | assumed |

### 2.2 Thermal-data table (heater efficiency, time constants, poles, geometry)

Column "tau" always states the definition. "Poles" = number of exponentials needed to fit the published response.

| # | Source | Device, band, geometry | Heater type / position | Undercut | Efficiency (as printed) | Converted (inferred) | tau (as printed) | Poles / notes |
|---|---|---|---|---|---|---|---|---|
| T1 | imec / KU Leuven, Coenen et al., JLT 2022 [CO22-JLT], Tables II-IV, Fig. 16 | Depletion ring modulator, 15 um diameter, C-band (dn_eff/dT = 1.95e-4 /K simulated) | W metal heater above the Si waveguide | none / incomplete / full | measured 222 +/- 10 pm/mW (2.99 +/- 0.14 K/mW) no UCUT; 300 pm/mW (4.04 K/mW) incomplete; 843 +/- 55 pm/mW (11.35 +/- 0.74 K/mW) full UCUT | 36 uW/GHz (no UCUT), 9.5 uW/GHz (full UCUT) | no UCUT: heater 4.92 us (model) / 5 us (exp), waveguide 10.38 us (model) / 10 us (exp); full UCUT: heater 18.1 us, waveguide 38.1 us (**simulated**, x3.68) | tau defined at 63 % of a first-order fit (authors state first order is adequate per node). UCUT raises efficiency x3.08 (sim), heater-to-WG coupling 58 % -> 85 % (sim). Metal tiling above the ring can cost up to 60 % efficiency. [verified against the author copy: Tables II-IV, Fig. 16, coupling 58.4 % -> 84.5 %] |
| T2 | imec, Coenen et al., IEEE TCPMT 2022 [CO22-TCPMT], Table I | 8-channel double-ring filter, ring diameter 14 um, pitch 100 um, FSR 13.12 nm, C-band | W metal heater above | none / full | 1.52 K/mW (no UCUT, sim); 4.8 K/mW = 319 pm/mW = 41.1 mW/FSR (UCUT, sim); a similar measured filter: 36 mW/FSR | - | Foster fit of the FE step response (**simulated**, calibrated): no UCUT 3.2 us (R 0.651 K/mW) + 13.0 us (0.872 K/mW); UCUT 2.5 us (0.276) + 25.6 us (2.01) + 54.5 us (2.314) + 124 us (0.198) | **2 poles without UCUT, 4 with** (printed). tau = R x C inferred from printed R, C (electrical-equivalent units: 1 Ohm = 1 K/mW, 1 uF = 1 nJ/K). Start-up: device-level ~40 us, circuit-level ~4 ms (printed). 200 kHz heater drive is attenuated to -30 dB (printed). [verified against the accepted version: Table I R, C values, 1.52 / 4.8 K/mW, 41.1 / 36 mW/FSR, 40 us / 4 ms, -30 dB at 200 kHz] |
| T3 | imec, Coenen et al., Photonics 2024 [CO24-PH], Tables 1-2 | Ring modulator with W heater (imec platform) | W heater above | none / small (16 x 24 um^2) / large (34 x 34 um^2) | 10 mW step gives ~2.5 nm (printed, approx.) | ~0.25 nm/mW | pure thermal tau_th converted from measurement: 5 / 9 / 18 us (none / small / large UCUT). Apparent "thermo-optic" tau measured outside the linear regime: cooling 6 / 12 / 24 us, heating 4 / 4 / 4 us | Shows that apparent heating and cooling tau differ only because of the Lorentzian readout; simulated example: heating 0.3 us vs cooling 4.8 us (no UCUT), 0.6 vs 66.2 us (UCUT). Linear UCUT size vs tau (sim). [verified: Tables 1-2. Table 1 also gives the linear-regime simulated tau, equal for heating and cooling: 6 us (no UCUT), 28 us (UCUT)] |
| T4 | imec, Coenen et al., electromigration paper [CO-EM] (IEEE author manuscript; venue not verified), Sec. III, Table I | Ring modulator r = 5 um, 450 x 225 nm WG, O-band 1304 nm, FSR 13.41 +/- 0.16 nm (measured) | W heater, 600 nm wide | none (meas.), UCUT (sim) | 0.183 +/- 0.019 nm/mW (design 2, median of 48), 0.165 +/- 0.005 nm/mW (design 1); heater self-temperature 7.03 degC/mW (measured) | P_FSR = 73 to 81 mW; 31 uW/GHz | - | Simulated waveguide efficiency 3.07 to 4.25 K/mW (no UCUT) and 6.6 to 12.0 K/mW (UCUT); UCUT gain x2.15 to x2.81 (sim). [verified against the author manuscript. Note: its Eq. 3 uses the C-band dn_eff/dT = 1.95e-4 /K at 1304 nm; the model's 3.07 K/mW -> 0.205 nm/mW implies 66.8 pm/K (inferred)] |
| T5 | imec, Coenen et al., J. Opt. Microsyst. 2024 [CO24-JOM], Table 2, Eq. 2 | 4 ring modulators at 100 um pitch (crosstalk test used O-band rings with UCUT and C-band rings without, Fig. 5 caption); EIC flip-chipped on top with 50 um-pitch Cu-Sn microbumps | W heater above | none / UCUT | pm/mW: no UCUT 165 (no EIC) -> 131 (with EIC); UCUT 312 -> 180 | at 62.54 pm/K: no UCUT 2.6 -> 2.1 K/mW; UCUT 5.0 -> 2.9 K/mW (no EIC -> with EIC) | - | **EIC on top: -43.3 % efficiency, +44.4 % crosstalk** (printed). dlambda/dT = 62.54 pm/K measured for O-band rings. [corrected: pm/mW values verified (Table 2). Measured drops are 42.4 % (UCUT) and 20.6 % (no UCUT); +44.4 % is simulated. 62.54 pm/K is printed "for the rings measured in the O-band", method not stated. The no-UCUT rings are C-band (Fig. 5 caption), so 62.54 pm/K should not convert them: at the C-band 74.2 pm/K implied by [CO22-JLT] Table II (222 pm/mW / 2.99 K/mW) they are 2.2 -> 1.8 K/mW; the paper's own FE gives 3.07 -> 2.14 K/mW] |
| T6 | Berkeley / MIT / Ayar Labs 45 nm "zero-change" SOI: Sun et al. VLSI 2015 [SUN15], JSSC 2016 [SUN16]; Stojanovic et al. OE 2018 [STO18] | r = 5 um, ~1180 to 1190 nm, FSR 3.2 THz, loaded Q up to 10 k | silicided c-Si heater embedded in the ring, 500 Ohm | substrate removed in post-processing (flip-chip die) | 3.8 uW/GHz; 600 GHz (3 nm, ~60 K) range [SUN15]; 524 GHz (>50 K) range [SUN16]; 3.53 uW/GHz, 5.4 mW for 1.53 THz (140 K) [STO18, Moazeni PAM4 tuner] | 1.24 nm/mW; 12.2 mW/FSR; ~26 K/mW (140 K / 5.4 mW) | not printed. [SUN15] states its controller (one decision per 13 us) runs slower than the heater-to-ring time constant -> tau < ~13 us (inferred, weak) | Resonance drift printed as 10 GHz/K (~47 pm/K at 1189 nm). [verified for [SUN15]: 10 GHz/K, 600 GHz (3 nm, ~60 K), 3.8 uW/GHz, "tuning controller operates slower than the heater-to-ring temperature time constant"; [SUN16], [STO18] not re-checked] |
| T7 | Sandia, Watts et al., CLEO 2009 [WAT09] | Second-order ARM filter, 6 um diameter | n-doped (2e18) Si heater **inside** the ring, contacted by thin Si tethers | no | 4.4 uW/GHz; 16 nm at 9 mW, 32.85 nm at 18 mW | 1.8 nm/mW | **tau = 1 us** (single-exponential fit to a 200 kHz, 0.3 Vpp square-wave drive, measured) | 1 pole fits. FEM predicts sub-1 us and 5 uW/GHz for 5 um OD. [verified against the CLEO 2009 PDF] |
| T8 | Gan et al., Photonics in Switching 2007 [GAN07] (secondary) | Over-clad metal heater, optimized | metal above | no | 28 uW/GHz [WAT09]; ~42 mW/FSR [P] | 0.29 nm/mW | 6.4 us "time constant" [WAT09]; 14 us "tuning speed" [P] | 14 us = 2.2 x 6.4 us, so [P]'s 14 us is almost certainly the 10-90 % time (inferred). [verified: [WAT09] prints "28 uW/GHz tuning with a 6.4 us time constant" for optimized over-clad heaters, citing Gan] |
| T9 | Dong et al. (Kotura/Oracle), OE 18(10) 2010 [DONG10a] | Microring filters with thermal isolation trenches | heater on ring | trenches only | 21 mW/FSR | - | "< 10 us" tuning time constant (abstract) | rings spaced 15 um show "almost no" crosstalk (abstract, qualitative) |
| T10 | Dong et al., OE 18(19) 2010 [DONG10b] | Racetrack resonator, free-standing (undercut) | heater on ring | full | 2.4 mW/FSR | - | 10-90 % switching time ~170 us (abstract) -> **tau ~ 77 us** (inferred) | Titriku et al. [TIT14] model this device with a 77 us time constant, consistent. [verified: abstract prints 2.4 mW/FSR and "10%-90% switching time ~170 us"; 170 / ln 9 = 77.4 us; [TIT14] Sec. II uses 77 us citing Dong] |
| T11 | Cunningham et al. (Oracle), OE 18(18) 2010 [CUN10] | Add-drop filter, r = 100 um, 130 nm SOI CMOS | integrated CMOS heater | backside substrate removal under device | 3.9 mW per 2 pi (one FSR); ~20x better than unprocessed | - | not in abstract | - |
| T12 | Zheng et al. (Oracle), OE 22(10) 2014 [ZHE14] | 10 Gb/s ring modulator + CMOS driver | integrated heater | selective substrate removal | ~4.5 nm open-loop range at ~14 mW total incl. heater driver | >= 0.32 nm/mW | not in abstract | closed loop tracked >500 GHz with 20 fJ/bit controller energy |
| T13 | Parsons, ..., Bergman, Rizzo, Sci. Rep. 15, 29906 (2025) [PAR25] (AIM 300 mm) | (a) vertical-junction microdisk r = 4.5 um; (b) lateral ring r = 20 um; (c) racetrack; (d) MZI phase shifter | doped Si heaters (no metal heater layer in this process): interior for disk/ring | none / wafer-scale UCUT | P_pi: disk 26.14 -> 5.32 mW (x4.92); ring 100.0 -> 3.99 mW (x25.1); racetrack 63.1 -> 1.31 mW (x48.1); MZI 43.46 -> 1.07 mW (x40.5) | P_FSR = 2 P_pi: disk 52.3 -> 10.6 mW | disk (**simulated**): rise/fall 11.0/11.3 us -> 96.7/95.7 us with UCUT. MZI (**measured**): rise/fall 12.6/25.3 us -> 1.853/1.258 ms with UCUT | rise-time definition not stated in the text I read; if 10-90 %, disk tau ~5 us -> ~44 us (inferred). UCUT disk becomes thermally unstable from optical self-heating at -14 dBm input. [corrected: the Methods (arXiv:2503.09009v2) define rise/fall as the 10 % and 90 % thresholds of the optical transient, so tau = t/2.197 is inferred: disk 5.0 -> 44 us, MZI 5.7 us -> 0.84 ms. The paper is inconsistent on the disk: the Results call its transients simulated, the Methods say disk and MZI transients were measured. P_pi values and -14 dBm verified] |
| T14 | Atabaki et al., OE 21(13) 2013 [ATA13] | Microdisk 4 um diameter | current pulses through the Si device layer | no | 2.06 nm/mW | - | 85 ns reconfiguration (pulsed drive) | shows the heater node can be sub-us when heat is generated in the guiding layer |
| T15 | Intel: J. Sun et al., JLT 37, 110 (2019) [JSUN19] (abstract-only, via citing papers) | 128 Gb/s PAM4 MRM | integrated heater | not stated | 19.5 mW per pi; full FSR (6.6 nm) tunable | 39 mW/FSR, 0.17 nm/mW | not found | - |
| T16 | Intel: Huang et al., IEDM 2023 [HUA23] | 8 all-pass MRMs on one bus, r ~ 6 um, FSR ~11.3 nm, 200 GHz grid | integrated heater per ring | not stated | not printed | - | not printed | heater power explicitly excluded from the energy numbers |
| T17 | GlobalFoundries, Nandi et al., ECIO 2023 [NAN23] | GF 45 nm CMOS-SiPh PDK MRM, r = 7.5 um, 56 Gbaud, add-drop with ~5 % drop coupling, O-band (1308.5 nm laser) | 4 silicide resistors (2 inside, 2 outside the ring) | **undercut** heater | heater currents 15.1 / 13.7 / 11.5 mA at three lock points (printed); efficiency not printed | - | not printed; integrator time constant set "comparable" to the heater time constant | analog drop-port lock (Section 3, C3). [verified against the ECIO 2023 PDF: 7.5 um radius, ~5 % drop coupling, undercut heater of four silicide resistors, 15.1 / 13.7 / 11.5 mA, 1308.5 nm] |
| T18 | Kramnik et al., arXiv:2411.05921 (2024) [KRA24] | GF 45RFSOI ring, outer r ~19 um, 2.8 um wide | resistive heater in ring centre, 10-bit DAC with delta-sigma | no | 0.62 nm total at 5.1 mW | 0.12 nm/mW | not printed | a layout bug reduced efficiency (printed) |
| T19 | Sandia, DeRose et al., OI 2014 [DER14] | Microdisk heater-modulators, gaps 3 to 20 um | interior ohmic heater | no | 5 mW gives 63.2 K (metal-connected) or 70.4 K (isolated) peak rise (simulated) | 12.6 to 14.1 K/mW (inferred, peak temperature, not waveguide average) | time constants vs gap simulated (values not extracted from the 2-page paper) | see crosstalk table. [verified: 63.2 / 70.4 degC at 5 mW, "spacing ... 3-20 um"] |
| T20 | Arbabi & Goddard, IEEE Photon. J. 2012 [ARB12] (abstract-only) | Si microring self-heating | optical absorption | - | - | - | - | thermal response modelled and measured as a **second-order** linear system |
| T21 | Gray et al., Phys. Rev. Applied 2020 [GRAY20] (abstract-only) | Si microrings with lateral p-i-n diodes (CMOS foundry) | carrier sweep-out / absorption | - | - | - | - | thermo-optic relaxation in encapsulated waveguides is ~logarithmic in time (2D diffusion) |

Groups asked about but not found with usable numbers in public sources: **IBM** (no microring heater tau or
efficiency found), **Ayar Labs TeraPHY** (no device-level heater tau found; only system-level ramp tests, Section 3),
**HP** (only the bias-tuning numbers of the TAMU/HP work, Section 4 C13), imec heater-architecture comparisons by
Masood et al. (GFP 2012/2013, cited by imec, not retrieved).

### 2.3 Multi-timescale structure (what a plant model must contain)

| Level | Physical body (Cauer view) | Time scale | Evidence | Provenance |
|---|---|---|---|---|
| L0 | heater strip itself | 1 to 5 us (metal above); sub-us for heat generated in the guiding layer | 4.92 / 5 us heater node [CO22-JLT]; 85 ns pulsed Si heater [ATA13] | printed |
| L1 | ring waveguide and surrounding oxide | 1 us (interior heater) ... 10 us (metal above, no UCUT) ... 18 to 124 us (undercut) | T1, T2, T3, T7, T10 | printed / simulated / inferred |
| L2 | die-level lateral spreading, neighbour rings | 4.1 to 9.0 ms for neighbours 1 to 7 at 100 um pitch; circuit-level start-up ~4 ms | [CO22-TCPMT] Table II, Fig. 14 | simulated |
| L3 | package, lid, cold plate, ASIC heat | ~30 to ~100 ms to the peak dT/dt after an ASIC power step | [AYAR25] ~100 ms; textbook [MRVL] ~30 ms | printed (vendor / secondary) |
| L4 | TEC / chassis / ambient | minutes to hours | G26 TEC sweep 35 to 105 degC over 31 h [LM26-P] | printed |

Notes:

- The Foster fit (sum of exponentials) is the form to **fit**; the Cauer ladder is the form to **reason** with (place
  a sensor, change the stack). [CO22-TCPMT] gives both for the imec ring; the textbook (Ch. 11) says the same.
- **Heating and cooling must have the same tau in a linear thermal system.** Published "heating faster than cooling"
  results usually come from reading temperature through the Lorentzian; [CO24-PH] reproduces this artefact. When our
  simulator reports a time constant from an optical trace it must apply the same caution.
- [CO24-PH] linear-regime rule for characterization: keep the heater-induced shift below half a linewidth,
  i.e. P_heater < lambda / (2 Q eta_h) (our reading of their "50 % of FWHM" criterion; their printed Eq. 17 was
  extracted ambiguously from the PDF).

### 2.4 Thermal crosstalk versus distance (static and dynamic)

| # | Source | Structure | Distance (centre to centre unless stated) | Static crosstalk C = d_lambda(victim) / d_lambda(heated) | Dynamics | Provenance |
|---|---|---|---|---|---|---|
| X1 | [CO24-JOM] Table 3 | 4 ring modulators on one bus, EIC flip-chipped, natural convection | 100 / 200 / 300 um | **measured** no UCUT 2.62 / 2.33 / 1.94 %; UCUT 1.65 / 1.43 / 1.32 %. Simulated 2.32 / 2.06 / 1.92 % and 1.68 / 1.49 / 1.39 % | - | printed. Crosstalk "quasi-constant" beyond ~200 um. Simulated top cold plate (0.1 K/W) cuts long-range crosstalk by 49.9 %. [verified: Table 3 values exact. Note the no-UCUT rings are C-band and the UCUT rings O-band (Fig. 5 caption)] |
| X2 | [CO22-TCPMT] Table II, Figs. 13-15 | 8-ch filter, 100 um pitch (first neighbour ~60 um from the heat source per the paper) | 1st to 7th neighbour | cumulative coupling onto one channel 2.8 % (printed) | first-order neighbour responses with tau = 4.14, 5.67, 6.57, 7.21, 7.82, 8.48, 8.99 ms (UCUT; no-UCUT within 2 %). f3dB: heated channel 30 kHz vs first neighbour 30 Hz | **simulated**, printed. Trenches cut coupling ~10 % only within ~50 um of the trench. Package thermal resistance (4 / 11 / 20 K/W) changes the coupling. [verified: Table II (no-UCUT 4.09 to 8.87 ms, within 1.3 %), "main channel ... approximately 2 um from the heat source and the first neighbor at 60 um", 30 kHz vs 30 Hz, 2.8 %] |
| X3 | [DER14] | microdisk heater-modulators | 3 to 20 um (sim); measured set | "a few percent" at ~10 um (measured) and beyond 15 um (simulated); a shared metal-1 line up to doubles crosstalk at large separation | time constants vs gap simulated (not extracted) | printed (qualitative magnitudes); measurement and simulation differ by about x3 (printed). [verified. Correction to the column header: [DER14] gives "spacing" / "gaps" between disks (edge to edge), not centre-to-centre distances] |
| X4 | [DONG10a] abstract | rings with air trenches | 15 um spacing | "almost no" thermal crosstalk | - | printed, qualitative |
| X5 | arXiv:1007.1169 (8-channel reconfigurable microring filters; secondary via search snippet) | 8 microring filters | not extracted | 1 nm shift on one ring moved the adjacent channel 0.03 nm (~3 %) | - | printed in source, distance unknown |
| X6 | [KRA24] | GF 45RFSOI rings | 325 um | aggressor ring heater pulsed max/min; closed loop held lock, open loop lost it | - | printed, qualitative |
| X7 | Lightmatter | - | - | **not published** (textbook) | - | - |

Derived scaling for short pitches (use only near field, d <= ~60 um): 1D diffusion through oxide,
tau ~ 4 d^2 / (pi^2 alpha_SiO2) with alpha_SiO2 = 7.5e-7 to 8.4e-7 m^2/s gives tau ~ 0.11 to 0.12 ms at 15 um,
0.43 to 0.49 ms at 30 um and 1.7 to 2.0 ms at 60 um (derived). The FE result at 60 um is 4.1 ms, about 2x the 1D
estimate, so scaling the FE value instead (4.1 ms x (d / 60 um)^2) gives 0.26 ms at 15 um and 1.0 ms at 30 um
(derived). Use 0.1 to 0.3 ms at 15 um as the plausible band. [CO22-TCPMT] checks the same d^2 law
(f3dB ratio = (L_i / L_m)^2 = 900 for 60 um vs 2 um) against its FE model. At long range the simulated tau rises only
from 4 to 9 ms between the 1st and 7th neighbour, so the d^2 law must not be extrapolated beyond ~100 um.
[verified: the formula is [CO22-TCPMT] Eq. 4 and the numbers recompute (0.122 / 0.109 ms at 15 um, 1.95 / 1.74 ms at
60 um; 4.1 ms x (15/60)^2 = 0.26 ms). Note: imec's calibrated oxide properties ([CO22-JLT] Table I: k = 1 W/m/K,
rho = 2200 kg/m^3, c_p = 635 J/kg/K) give alpha = 7.2e-7 m^2/s, slightly below the range used here; at 60 um that gives
2.0 ms, still ~2x below the FE 4.1 ms. Cross-check: the paper's own d^2 anchor (heated channel 30 kHz at ~2 um, i.e.
tau ~5.3 us) scaled to 15 um gives 0.30 ms, at the top of the 0.1 to 0.3 ms band.]

### 2.5 Recommended thermal parameterisation for `mrm-model` (derived)

| Parameter | Default | Sweep range | Basis |
|---|---|---|---|
| Heater-to-ring R_th | 4.7 K/mW | 2.1 to 11.4 K/mW | default = textbook (derived, G25); range = imec measured no-UCUT ring with EIC on top (131 pm/mW / 62.54 pm/K = 2.1 K/mW, inferred) to full UCUT (11.35 K/mW, printed) [CO24-JOM, CO22-JLT]. [corrected: that no-UCUT ring is a C-band device (Fig. 5 caption of [CO24-JOM]), so 62.54 pm/K (O-band) does not apply; with the C-band 74.2 pm/K implied by [CO22-JLT] it is 1.8 K/mW, the paper's own FE gives 2.14 K/mW. Use 1.8 K/mW as the lower sweep bound. 11.35 +/- 0.74 K/mW verified] |
| Self-path Foster poles, "metal above, no UCUT" | 3.2 us (43 %) + 13.0 us (57 %) | scale both x0.5 to x2 | inferred from [CO22-TCPMT] Table I; 63 % point ~10 us matches [CO22-JLT] measured 10 us and the textbook's assumed 10 us. [corrected: the 63 % point of 0.427(1 - e^(-t/3.18 us)) + 0.573(1 - e^(-t/13.01 us)) is **7.3 us** (10 % at 0.60 us, 90 % at 22.7 us; derived), not ~10 us. It is the same order as the measured 10 us of a different imec device (15 um modulator vs 14 um filter) and inside the x0.5 to x2 sweep. Cross-check: the 63 % point of the UCUT 4-pole fit is 37.7 us, matching the simulated 38.1 us of [CO22-JLT]] |
| Self-path Foster poles, "undercut" variant | 2.5 us (6 %) + 25.6 us (42 %) + 54.5 us (48 %) + 124 us (4 %) | - | inferred from [CO22-TCPMT] Table I (simulated) |
| Self-path, "interior heater" variant | single pole 1 us | 0.3 to 3 us | [WAT09] measured |
| Die / package tail on the self path | one pole 1 to 10 ms carrying 2 to 10 % of DC gain | 0 to 20 % | assumed; motivated by [CO22-TCPMT] circuit-level 4 ms and by EIC heat spreading [CO24-JOM] |
| Crosstalk static, neighbour at ~pitch | 3 % | 1 to 10 % | X1 to X5 (few % at 10 to 100 um); 10 % kept as pessimistic corner (textbook) |
| Crosstalk static, all other rings (floor) | 1.5 % each | 0.5 to 2.6 % | X1 (1.3 to 2.6 % out to 300 um, with EIC) |
| Crosstalk dynamics | first order per pair. Pitch ~100 um: 4.1 ms (1st neighbour) rising to 9.0 ms (7th). Shorter pitch: tau_xt ~ 4.1 ms x (d / 60 um)^2, floor 0.1 ms, where d is the heat-source-to-victim distance as defined in [CO22-TCPMT] (60 um for their 100 um pitch) [2026-10-16: mrm.plant.thermal takes d as the centre distance less both ring radii (imec R ~ 20 um, inferred), so the g26 50 um pitch with r_ref 6.7 um gives d = 36.5 um and 1.52 ms] | x0.3 to x3 | X2 (simulated, printed) plus the derived near-field d^2 law |
| Absorbed-light heating | injected at the waveguide node (no heater-node lag) | - | physics; [ARB12], [DEC19] |
| External aggressors (on-die heater, ASIC, TEC) | separate inputs with their own transfer functions h_agg(t) | - | G26 uses three different aggressor methods (textbook correction 3) |

Implementation note: represent the array as an N x N matrix of impulse responses h_ij(t), each a sum of first-order
terms (Foster), plus aggressor columns. This is the MIMO form suggested in the prior-notes checklist and is exactly
what [CO22-TCPMT] builds with VCCS-coupled RC networks (a single shared resistor network underestimates coupling to
far neighbours when all heaters are on, per their Fig. 9). [corrected: two separate failures in [CO22-TCPMT]: with all
heaters on equally, no current flows in the coupling resistors at all, so the resistive model shows no coupling
(Fig. 9); with a single heater on, the resistive model is right for the main channel and first neighbour but
underestimates coupling to further neighbours (Fig. 11).]

---

## 3. Disturbances (Topic B)

### 3.1 Disturbance table

| # | Disturbance | Value | Method / where it enters | Source | Provenance |
|---|---|---|---|---|---|
| D1 | On-die heater aggressor next to one MRM | 808 degC/s, 25 degC to >105 degC, "period of 100 ms"; BER sampled every 2 s stayed at 1e-11; drop-port monitor flat | resistive heater adjacent to the MRM (G26) | [LM26-P] Fig. 3(b) | printed. Note: 808 degC/s x 0.1 s = 81 K ~ the 80 K swing, so each ramp lasts ~100 ms; whether "period" means one ramp or one up-down cycle is not stated (inferred). [verified against the local PDF, Sec. 3.2: "ramped up and down between 25oC and >105oC with a period of 100ms", BER "sampled every 2 seconds, remained stable at 10-11", ramp rate "808oC/s within the MRM". If "period" were one up-down cycle, each 80 K ramp would take 50 ms (1600 degC/s), which contradicts the printed 808 degC/s, so the arithmetic favours one ramp per 100 ms (inferred); the paper still does not say so] |
| D2 | Same, newer configuration | sawtooth ~-2300 to +2200 degC/s, ~131 ms period, ~30 to ~113 degC | on-die aggressor (G26 slides) | textbook `REF.transientCPerS`, [LM26-S] | read-off (textbook) |
| D3 | Optically emulated ramp | 50 nm/s laser sweep = 800 degC/s | tunable laser; the die is never heated | textbook [OCPW] | printed / inferred (textbook) |
| D4 | Slow package cycling | 35 to 105 degC over 31 h, 8 lanes BER < 1e-9 | TEC on the package | [LM26-P] Fig. 3(a) | printed (textbook also lists a 30.5 h, 25 to 105 degC soak and a separate 1.3 to 1.4 degC/s cycling test) |
| D5 | ASIC power step seen by an optical engine | 0 -> 500 W ASIC: ~100 degC/s brief peak over a few degC about 100 ms after the step, OE ~3 mm away; XPU itself 400 degC/s | package-level heat flow | [AYAR25] Ayar Labs blog, 28 Aug 2025 | printed (vendor, method not peer-reviewed). [verified, with a caveat: the blog says the test "simulates an ASIC or XPU increasing its power dissipation from 0 to 500W"; it does not say whether this is a physical thermal test vehicle or a numerical model. It also says optical engines "will be up to 25mm away" in real systems, so 3 mm is a near-worst case] |
| D6 | Emulated fast ramp (Ayar) | tunable laser swept "10 to 50 nm" (per second, inferred from the stated 160 to 800 degC/s equivalence) emulating 160 to 800 degC/s; essentially no bit errors during the ramp; separate thermal cycling 30 to 80 degC | laser sweep | [AYAR25] | printed (vendor); implies a 62.5 pm/K conversion (inferred) |
| D7 | XPU power-state change at the PIC | ~10 degC/s (organic substrate), ~100 degC/s (bridge), ~120 degC/s (interposer) within ~30 ms | package | textbook [MRVL] (SemiAnalysis report of Marvell ECTC 2026) | secondary |
| D8 | EAM-based photonic fabric claim | tolerance to excursions up to 200 degC/s and >85 degC swings | Celestial AI (now Marvell), Hot Chips 2025 | [CEL25] Converge Digest | secondary |
| D9 | Static die gradient and hot spot | 63 K across one instrumented array; PIC peak 100 degC beside a >900 W ASIC on a 25 degC cold plate | M1000 | textbook `REF.dieGradientK`, `REF.picPeakC` [HC] | printed (textbook) |
| D10 | Academic TEC cycling while locked | 25 to 80 degC, max 3.1 degC/s (limited by the TEC), 8-channel RX error-free at 16 Gb/s | TEC under interposer | [HAT24] Fig. 3a | printed. [verified: "maximum of rate of 3.1 degC/s", limited by the TEC; C-band (DFB array around 1550 nm)] |
| D11 | Academic tracking spec | 5 nm/s, stated equivalent ~62.5 degC/s; 5 nm (~62.5 degC) tuning range | thermal and laser-wavelength variation | [AGA16] | printed (abstract-only) |
| D12 | Abrupt laser jumps | up to 150 pm steps, OMA held at 2 Gb/s | laser | Agarwal et al., A-SSCC 2015 (via [AGA16] search record) | printed (abstract-only) |
| D13 | Optical (pump-laser) heating aggressor | 3 K at 10 Hz sinusoid | visible laser on chip | [PTC13] | printed; peak slope 94 to 188 K/s depending on whether 3 K is peak-to-peak or amplitude (derived) |
| D14 | Slow ambient in a photoconductive-heater demo | 5 degC at 0.28 degC/s, clear eyes at 12.5 Gb/s | stage temperature | [JAY15] | printed |
| D15 | Simulated aggressor (imec) | up to 5 degC with a 1 s time constant; 10 ms control step, 0.1 degC hysteresis, 0.01 mW heater step | package heat source in an RC model | [CO22-TCPMT] Sec. III-D | simulated |
| D16 | Laser absolute accuracy (G26 requirement) | +/-20 GHz; demonstrated +/-15 GHz; roadmap 3 GHz | Guide laser module | [LM26-P], textbook `REF.laserAccuracyGhz` | printed |
| D17 | Laser drift | 0.02 nm / 24 h; power 2.5 % / 24 h | Guide | textbook `REF.laserDriftNm24h`, `REF.laserPowerDrift24hPct` | as in textbook |
| D18 | DFB temperature coefficient (generic O-band) | 0.09 nm/degC (Coherent DFB15 chip), 0.1 nm/degC (Thorlabs DFB1310P) | laser TEC error | vendor datasheets [DFB] | printed. Derived: 1 K of laser-TEC error at 0.1 nm/K = 17.5 GHz at 1310 nm, i.e. most of the +/-20 GHz budget |
| D19 | Monitor swing under polarization scramble | 7.7 dB (code pair 7,000 to 41,500), 8 to 9 dB band | RX polarization-diverse monitor PDs (G26) | textbook `REF.monitorPolSwingDb` [LM26-P] | read-off (textbook) |
| D20 | Squelch | modulation amplitude cut to <= -15 dBm with average power preserved | OCI MSA | textbook [OCI] via [OCP] | printed (normative) |
| D21 | Data 1/0 balance changes | random ones-density changed every 200 ms, including all-0 and all-1 | data pattern -> ring absorption | [SUN15] Fig. 5 | printed |
| D22 | Neighbour heater steps | see crosstalk table X1-X6 | other rings | Section 2.4 | - |

### 3.2 Laser wavelength and power drift - what to model

- **Absolute offset:** draw a per-line offset in [-20, +20] GHz (G26 requirement) and a correlated common-mode offset
  (one laser module); default +/-15 GHz (demonstrated). Printed values, textbook.
- **Slow drift:** 0.02 nm / 24 h and 2.5 % power / 24 h (textbook). These are negligible for the thermal loop but
  matter for any controller that uses an absolute power setpoint (C3).
- **Laser TEC error:** with an O-band DFB at 0.09 to 0.1 nm/K (vendor-printed), each 0.1 K of laser temperature error
  moves the line by ~1.6 to 1.75 GHz (derived). Model a laser-temperature random walk plus a step input.
- **Fast laser steps:** published locks survive 150 pm abrupt laser jumps (D12). Include a laser-step test.
- **Correlation with ring temperature:** none published. The Guide laser is a second servo coupled to the ring loop
  (textbook). Keep the laser temperature an independent input.

### 3.3 Data-pattern-dependent self-heating

What is published:

- Mechanism: the ring absorbs a different fraction of light for 1s and 0s, so its temperature follows the running
  ones-density through the thermal low-pass; free-carrier dispersion shifts the resonance blue while heating shifts it
  red (several sources, e.g. [DEC19] abstract, [SUN15]).
- [SUN15]/[SUN16]: with average-power locking, random ones-density changes every 200 ms closed the eye completely
  (the controller cannot tell a density change from a resonance drift); independent 1/0 level tracking fixed the lock
  point but sudden density changes still caused transient eye closure until a feed-forward "self-heating canceller"
  (heater correction computed from the bit counts and the eye height L_diff) was enabled (printed, qualitative).
- [DEC19]: a time-domain model including TPA, FCA, thermal effects shows resonance bistability (initialization must be
  done from the correct side) and an optimal input power beyond which modulation degrades (abstract-only).
- [PAR25]: an undercut microdisk becomes thermally unstable already at -14 dBm input (printed), so undercut raises
  self-heating sensitivity together with heater efficiency.

Derived magnitude for the reference ring (all inputs except R_th and dlambda/dT are assumed):
dT = dP_abs x R_th,self. With R_th,self ~ R_th = 4.7 K/mW (assumed equal to the heater path) and a change in absorbed
power dP_abs of 0.1 / 0.25 / 0.5 mW (assumed; G25 has ~4 dBm = 2.5 mW at the ring, textbook), the shift is
40 / 101 / 202 pm, i.e. 11 / 27 / 54 % of the 374 pm linewidth (Q = 3500). This is large, so the absorbed fraction is a
first-order unknown (the textbook already lists it as unknown). [verified: 0.1 / 0.25 / 0.5 mW x 4.7 K/mW x 86 pm/K =
40.4 / 101 / 202 pm; 1310 nm / 3500 = 374 pm; ratios 10.8 / 27.0 / 54.0 %]

Derived statistics argument: at 50 Gb/s one 10 us thermal time constant spans 5e5 bits; for scrambled random data the
standard deviation of the ones-density over that window is 0.5 / sqrt(5e5) = 0.07 % [verified: 7.07e-4]. Pattern-dependent self-heating
therefore matters mainly at **traffic start/stop, idle and squelch, training patterns and unscrambled framing**, not
in steady scrambled traffic. Our test set must include those events explicitly (density steps 0 -> 0.5 -> 1).

---

## 4. Algorithm catalogue (Topic C)

### 4.0 Common notation and three derived constraints

- k = control sample, T_s = control period. u[k] = heater command (mW, or DAC code), saturated to [0, U_max]: heaters
  only heat (one-sided actuator). y[k] = sensor sample (drop-port photocurrent unless stated). r = setpoint.
- sat(x) = clamp to the actuator range; slew(x) = rate limit (textbook: ~460 codes/ms needed, inferred).
- Unless stated, "lock" means holding the detuning within the textbook's 5 pm budget.

Derived constraint 1 (dither speed). A lock-in or extremum-seeking loop must have its bandwidth well below the dither
frequency f_d (typically <= f_d / 10). With f_d = 1 kHz (Padmaraju) the loop bandwidth is <= ~100 Hz, so a
172 pm/ms ramp (2000 degC/s x 86 pm/K, G26) produces a lag of order 1 / (2 pi 100 Hz) = 1.6 ms, i.e. ~270 pm of
error (derived, first-order estimate). To meet G26 with a dither scheme the dither has to sit at >= 100 kHz, which
the heater's own pole (tau 1 to 40 us, Section 2) attenuates strongly - so a dither on the heater is a poor fit;
a dither on the junction bias (fast) is a better candidate. [verified, with a refinement: for the 5 pm budget the
loop needs K_v >= 172 pm/ms / 5 pm = 34 /ms, i.e. a crossover of >= ~5.5 kHz, so f_d >= ~55 kHz under the f_d/10 rule;
">= 100 kHz" includes about 2x margin (derived). The f_d/10 rule is a design rule of thumb (assumed). For scale,
[CO22-TCPMT] prints -30 dB heater-to-waveguide response at 200 kHz and a 30 kHz 3 dB point for its heated filter.]

Derived constraint 2 (integrator slew). With ~29,800 codes over ~13 nm (textbook `REF.heaterDacCodeRange`,
`REF.heaterTuningRangeNm`), one code is ~0.44 pm on average (inferred; the real map is quadratic in voltage). A
1-code-per-decision integrator at 76 kHz ([SUN15] decision rate) slews 76 codes/ms against ~394 codes/ms needed for
172 pm/ms (derived). Step size or update rate must grow ~5x, or a feed-forward / type-2 term must carry the ramp.
[corrected: 13 nm is the G25 heater's full range at 1.5 V; the 29,800-code G26 command span does not cover it. The
textbook (Ch. 11 constants, `CODES_PER_K`) maps the code span onto the ~83 K excursion the published trajectory covers
(30 to 113 degC): 359 codes/K, i.e. ~0.24 pm/code at 86 pm/K (inferred). Its 460 codes/ms is the half-cycle average
(29,800 codes in ~65 ms). At the 2000 degC/s design rate the need is ~720 codes/ms, and ~830 codes/ms on the
2300 degC/s cooling edge (textbook, inferred). A 1-LSB integrator at 76 kHz (76 codes/ms, [SUN15] rate verified)
therefore falls short by ~6x (average) to ~11x (steepest edge), not ~5x. The conclusion (bigger steps, faster
updates, or a feed-forward / type-2 term) stands and gets stronger.]

Derived constraint 3 (ramp error and loop type). A type-1 loop (one integrator) tracks a ramp with a constant error
e_ss = ramp / K_v. A type-2 loop (two integrators, or an integrator plus a ramp feed-forward) has zero steady-state
ramp error but needs more phase margin around the thermal poles. The G26 sawtooth (D2) is a sequence of ramps, so
type-2 behaviour or an explicit ramp estimator (C15) is the natural baseline to beat.

Each entry gives: idea, sensor, actuator, update law (pseudocode), tuning parameters, failure modes, published
performance. Implement every entry against the same plant interface.

---

### C1. Wavelength search, lock acquisition and initialization (FSM)

- **Sensor:** any monotonic power monitor (drop-port PD, in-ring PD, RSSI of the receiver).
- **Actuator:** heater (coarse), optionally bias (fine).
- **Update law:**

```text
INIT:    u <- U_start            # start on the side from which the published flow approaches (see failure modes)
SEARCH:  while y < Y_thr:        # coarse strides until a resonance is seen
             u <- u + s * dU_coarse ; wait(T_settle >= 3..5 * tau_dominant)
MAP:     for n in 0..N-1:        # fine sweep across the feature, record (u_n, y_n)
             u <- u + s * dU_fine ; wait(T_settle) ; store y
         u_star <- argmax(y)  (peak lock)  or  u where y = alpha * max(y)  (slope lock)
RETURN:  u <- U_start ; ramp to u_star from the SAME direction as MAP   # hysteresis / bistability
HANDOFF: start the tracking loop (C2..C9) with its integrator preloaded to u_star
```

- **Tuning parameters:** start side and sweep sign s; dU_coarse (<= 1/2 linewidth in heater units), dU_fine
  (~linewidth / 10), T_settle, threshold Y_thr, averaging per point, timeout.
- **Failure modes:** locks onto the wrong comb line or onto a line already taken by another ring (WDM); FSR aliasing
  (G26: 8 x 400 GHz spans 2.8 THz, more than the design FSR, textbook correction 9); self-heating bistability makes the
  map depend on sweep direction; settle time too short for an undercut ring (tens of us x 5); heater range exhausted.
- **Published:**
  - [SUN15] FSM init -> search (large strides until the 0-level L0 drops below R_sweep) -> sweep (1 LSB, maps
    PH_opt, L0_opt, Ldiff_opt) -> reset/return (always approach the lock point from the same heating direction because of
    laser-heating bistability) -> lock. Decision rate 76 kHz, worst-case lock time 7 ms with 1-LSB search (printed).
  - [TIM12] FPGA "automated wavelength recovery" on an ARM ring, 200 us lock-in time (printed); sub-1 mW for a
    +/-10 degC span (the proceedings PDF prints +/-100 C; the MIT annual report prints +/-10 degC; arithmetic with
    4.4 uW/GHz supports +/-10 degC - inferred).
  - [HAT24] 8-channel RX: reference RSSI recorded unlocked; rings assigned sequentially; a ring placed on an already
    occupied line shows a much smaller RSSI rise, so a minimum RSSI increase is enforced to guarantee unique
    assignment (printed).
  - [KRA24] hot-to-cold sweep to find the photocurrent curve, reset hot, second sweep to the rising edge, then PI
    (printed).
  - [NAN23] analog loop: integrator ramps the heater from boot until V_PD = V_ref; can lock anywhere on the rising edge
    (printed).

### C2. Peak tracking by perturb-and-observe (hill climbing)

- **Sensor:** drop-port / in-ring / RX photocurrent that **peaks** on resonance (G26 TX monitor peaks, textbook).
- **Actuator:** heater.
- **Update law:**

```text
every T_s:
    y0 <- mean(y over M samples) at u
    apply u + delta ; y_plus  <- mean(y over M)
    apply u - delta ; y_minus <- mean(y over M)
    u <- the one of {u - delta, u, u + delta} with the largest y      # or u += K*(y_plus - y_minus)
```

- **Tuning parameters:** perturbation delta (heater volts or codes), averaging M, period T_s, optional deadband.
- **Failure modes:** perturbation shows up on the data; gradient vanishes on the flat top (noise dominates); max
  tracking rate ~ delta_lambda / (3 T_s); a power step inside one perturbation window is misread as a gradient;
  converges to a local maximum (side lobe, neighbour channel).
- **Published:** [HAT24] heater step 336 uV (empirical), several samples averaged per step, FPGA reading TIA RSSI via
  an 8-ch ADC, 8 DACs; tracked 25 to 80 degC at up to 3.1 degC/s (TEC-limited), error-free 16 Gb/s, 75 degC range
  (printed). A 2024 arXiv preprint reports perturb-and-observe at a 4 kHz iteration rate with TiN heaters and Ge PDs
  (secondary, from search; not verified).

### C3. Side-of-fringe setpoint lock (PI on mean monitor power)

- **Sensor:** mean drop-port, through-port, junction or in-ring photocurrent.
- **Actuator:** heater.
- **Update law (discrete PI with conditional-integration anti-windup):**

```text
e[k]  = s * (r - y[k])                 # s = +1/-1 chosen by which slope is used
if not saturated(u[k-1]) or sign(e[k]) != sign(u[k-1] - U_lim):
    I[k] = I[k-1] + Ki * T_s * e[k]
u[k]  = sat( Kp * e[k] + I[k] )
r     = alpha * y_peak (from C1)       # e.g. alpha = 0.95, or "6 dB insertion-loss point", or mid-slope
```

- **Tuning parameters:** Kp, Ki, T_s, setpoint fraction alpha, deadband, anti-windup limits; optional input-power
  normalization (C4).
- **Failure modes:** laser power change moves the lock point (dlambda ~ (dP/P) x y / slope); data-density change on a
  modulated ring (C6 fixes this); polarization-induced monitor swing (G26 RX monitor 7.7 dB, D19) if the monitor sees
  it; crossing the peak flips the slope sign and the heater runs away (one-sided actuator -> limit cycle or loss of
  lock); self-heating cliff on one side; works through squelch only if average power is preserved (OCI does preserve
  it).
- **Published:**
  - [PAD12], [PAD13] Columbia/McMaster: mean optical power monitored at the drop port (defect-mediated Si PD in [PAD13]),
    PID to the heater; error-free under thermal fluctuations that otherwise make the modulator fail (printed, numbers
    not in the abstracts).
  - [NAN23] GF 45 nm: Ge PD at the MRM drop port, TIA (5 kOhm, 1 GHz), difference amplifier vs V_ref, integrator with
    time constant comparable to the heater's, current mirror into the heater; footprint 200 x 220 um; lock anywhere on
    the rising edge; heater currents 11.5 to 15.1 mA (printed).
  - [SHA22] Intel 4 x 112 Gb/s: reverse-biased MRM junction photocurrent as the monitor; heater tuned until the MRM sits
    at a 6 dB insertion-loss point ([HUA23] describes the loop); < 0.05 dB TDECQ penalty over 10 degC (printed,
    abstract-level). No drop port needed.
  - [KRA24] PI at 95 % of the way from dark current to maximum photocurrent, deadband to stop DAC dithering, 10 Hz
    (Python), ~2 kHz (FPGA), ~60 kHz possible on-chip (printed).
  - [LEE16] Oracle 4 x 25 Gb/s RX, closed-loop thermal stabilization through a 4.5 nm input wavelength shift, polarization
    scrambling and >60 degC swings (printed, abstract).

### C4. Ratio-of-two-taps (normalized) error

- **Sensor:** two photocurrents whose ratio is independent of input power: drop / (drop + through), drop / input tap,
  or two drop-port PDs at different detunings.
- **Actuator:** heater.
- **Update law:** as C3 with y replaced by rho[k] = y_a[k] / (y_a[k] + y_b[k]) (or y_a / y_in); r = rho at the target
  detuning from a calibration sweep.
- **Tuning parameters:** as C3 plus the ratio setpoint and the dark-current offsets subtracted before division.
- **Failure modes:** dark current and offsets make the ratio power-dependent at low light; two PDs with different
  responsivity or temperature coefficients; extra tap loss; division noise near zero light; polarization if the two taps
  see different polarization mixtures.
- **Published:** I did not find a peer-reviewed silicon-MRM paper whose primary contribution is a two-tap ratio loop.
  [AGA16] (drop-port OMA monitor) and [COX14] (balanced detection) are stated to be insensitive to input power; a UBC
  thesis record describes a ring temperature controller robust to input-power fluctuations (not read). Treat C4 as
  standard practice (assumed) and as a cheap upgrade of C3. Lightmatter G26 has a drop-port monitor; an input tap is not
  reported (textbook).

### C5. Dither and lock-in (extremum-seeking control)

- **Sensor:** any monitor (drop or through) plus synchronous demodulation.
- **Actuator:** heater (Padmaraju) or junction bias (faster, not demonstrated in the papers read here).
- **Update law (first-harmonic lock = peak/notch; second-harmonic lock = inflection):**

```text
d[k]   = A * sq(2*pi*f_d*k*T_s)                  # square-wave dither (Padmaraju) or sine
apply  u[k] + d[k]
z1[k]  = LPF( HPF(y[k]) * d[k - m] )             # m = phase-compensating delay through the thermal pole
z2[k]  = LPF( HPF(y[k]) * d2[k - m] )            # d2 = dither at 2*f_d, for inflection lock
u[k+1] = sat( u[k] + K_i * T_s * z1[k] )         # drive z1 -> 0  (extremum)    -- or z2 -> 0 (max slope)
```

- **Tuning parameters:** dither amplitude A, frequency f_d, demodulation phase/delay m, LPF corner, integrator gain,
  harmonic (1st vs 2nd), dither waveform; for many rings, distinct f_d per ring (FDM) or orthogonal codes.
- **Failure modes:** the dither is visible on the data and costs extinction (measured: 0.1 K square dither reduced a
  Q ~ 14,000 ring's extinction ratio by 1.9 dB, 0.2 K by 4.8 dB - printed [PTC13]); the loop bandwidth is limited to
  ~f_d / 10 (constraint 1); phase lag through the thermal pole flips the sign if f_d approaches 1 / (2 pi tau);
  neighbour rings pick up each other's dither through crosstalk (only ms-slow, so mostly a DC effect); first-harmonic
  lock sits at the peak, which is the wrong point for a modulator.
- **Published:**
  - [PTC13], [PD] (Padmaraju et al. JLT 2014): 15 um radius depletion MRM, Ti heater 1 um above, defect-enhanced Si PD at
    the drop port, Q ~ 14,000; 1 kHz square-wave thermal dither of 0.1 K; analog mixer (AD633), RC low-pass, PID; held
    lock against a 3 K, 10 Hz optical aggressor; initial lock in ~50 s with a slow ramp, ~ms with a faster ramp
    (printed). [verified against the TECHCON 2013 PDF: 15 um radius, Ti heater, Q ~14,000, 1 kHz square wave,
    0.1 / 0.2 K dither -> 1.9 / 4.8 dB ER reduction (measured, simulation identical), 10 Hz 3 K, 50 s; "~ms" not
    re-checked] A later wavelength dither below 0.004 nm (~491 MHz) was enough (ECOC 2013 abstract, secondary).
  - [GRM23] Politecnico di Milano (Grimaldi et al., 2023): dither-based lock at the zero of the second derivative
    (maximum slope) of a 50 Gb/s MRM, independent of optical power, calibration-free, FPGA with 30 ms response time
    (printed, abstract-level; venue listed as JLT in citing papers, not verified).
  - Agarwal et al., ESSCIRC 2017 (imec): dither-based locking accuracy 0.4 % (2.2 pm, 0.03 degC) (secondary, from
    search record).
  - A CLEO 2014 abstract describes an extremum-seeking adaptive controller tuning both the resonant phase and a coupling
    phase of a microring BPSK modulator (abstract-level, secondary).

### C6. Bit-statistics "eye-max" locking (Sun et al.)

- **Sensor:** drop-port PD integrated over a window plus the 1/0 counts of the transmitted bits in the same window.
- **Actuator:** heater (sigma-delta DAC in [SUN15]).
- **Update law:**

```text
for each window w (W = 16..128 bits):
    Q_w  = integral of drop-port photocurrent over the window        # integrator + SAR ADC
    n1_w, n0_w = counts of ones and zeros in the same window
collect >= 2 windows with different n1/W; solve least squares:
    Q_w = n1_w * L1 + n0_w * L0                                      # L1, L0 = optical 1- and 0-levels
Ldiff = L1 - L0                                                      # vertical eye opening proxy
FSM: search/sweep/return as in C1 (map Ldiff(u)), then LOCK: perturb u by 1 LSB, keep the move if Ldiff increases
self-heating canceller (feed-forward): u += G_sh * d(n1/W) * Ldiff  when the ones-density changes
```

- **Tuning parameters:** window W, decision rate, step size, canceller gain G_sh, thresholds R_sweep.
- **Failure modes:** needs data (fails in squelch / idle, when L1 ~ L0 - textbook); needs access to the transmitted
  bits (SerDes integration); weak drop tap limits SNR; slew limit (constraint 2); PAM4 needs a different objective
  (the platform later added a PAM4 tuner, [STO18]).
- **Published:** [SUN15] 45 nm SOI, 5 Gb/s, 1187.2 nm ring locked to 1189 nm laser; decision rate 76 kHz, lock <= 7 ms,
  front-end sensitivity < 500 nA with ~3.5 uA available; 600 GHz (3 nm, ~60 K) range at 3.8 uW/GHz; tuner 275 fJ/bit
  (39 fJ/bit front end) (printed). [verified for [SUN15]: 76 kHz ("2 controller clock cycles at 5Gbps"), 7 ms upper
  bound with 1-LSB search, < 500 nA vs ~3.5 uA, 600 GHz / 3 nm / ~60 K, 3.8 uW/GHz, 275 fJ/bit, 1187.2 / 1189 nm,
  1/0 balance changed every 200 ms; 39 fJ/bit not re-checked] [SUN16] JSSC: 524 GHz (>50 K) range, 0.74 mW logic (printed, abstract). Both
  versions are real; the range differs between them.

### C7. Drop-port OMA monitoring lock (Agarwal et al., imec)

- **Sensor:** drop-port PD with a circuit that measures the optical modulation amplitude (swing), not just the mean.
- **Actuator:** heater.
- **Update law (generic form):** maximize OMA_hat(u) by hill climbing (C2) or lock OMA_hat to a fraction of its maximum
  on a chosen slope (C3); the published circuit details were not readable here.
- **Failure modes:** needs modulation (squelch); OMA estimate depends on data pattern and driver swing.
- **Published:** [AGA16] JSSC 2016: tuning range 5 nm (~62.5 degC), tracking 5 nm/s (~62.5 degC/s), locking precision
  < 10 pm (~0.125 degC, within ~2 % of maximum OMA), robust to input-power variation, 2 Gb/s modulation (printed,
  abstract-only).

### C8. BER / eye-monitor driven tuning

- **Sensor:** receiver BER counter or eye-margin monitor (often at the far end, i.e. needs a return channel).
- **Actuator:** heater.
- **Update law:** slow hill climb or golden-section search on BER(u) or eye height(u); dwell long enough to count
  errors (BER 1e-9 needs ~1e10 bits for statistics).
- **Failure modes:** very slow at low BER; needs link-level protocol; the error counter in G26 is sampled every 2 s
  (textbook).
- **Published:** [ZOR13] Sandia: wavelength locking and stabilization over a 32 K range by monitoring BER (printed via
  [P]); forecast 55 fJ/bit at 10 Gb/s (printed via [P]).

### C9. Balanced homodyne error signal (Cox et al.)

- **Sensor:** unbalanced MZI with the ring in one arm and a phase shifter in the other, recombined on a 50 % coupler
  into a balanced PD pair.
- **Actuator:** heater (lock), plus a one-time DC phase-shifter setting that moves the zero crossing.
- **Update law:**

```text
e[k]   = I_PD1[k] - I_PD2[k]               # odd-shaped error with zero at resonance when phi = 0
u[k+1] = sat( u[k] + K_i * T_s * e[k] )    # integral-only controller was sufficient
phi    = 0 for a filter (lock on resonance); ~pi/4 for an OOK modulator (side lock); phi = pi/2 gives no lock
```

- **Tuning parameters:** phase phi, integrator gain, coupler balance (epsilon ~ 0.5).
- **Failure modes:** extra MZI, phase shifter and two PDs per ring; phase drift moves the lock point (at pi/2 there is
  no zero crossing); polarization and etalons in the reference arm.
- **Published:** [COX14] Sandia: integral-only control (SRS SIM960, or a single op-amp integrator); filter held on
  resonance while the laser swept 1532 to 1542 nm (1.25 THz) for >100 s; a microdisk modulator locked with low BER while
  the chip went from 5 to 60 degC (printed).

### C10. Pound-Drever-Hall-like and data-aided coherent error signals

- No silicon-MRM thermal-locking paper using classic PDH sidebands was found. [COX14] lists PDH among prior techniques
  and notes it needs a high-speed phase modulator and demodulation electronics (printed).
- Our view (derived, not published): in a TX MRM the data modulation already puts energy far above the thermal
  bandwidth; correlating the monitor with the transmitted data (C6) is the data-aided analogue of PDH. A bias-domain
  dither at >100 kHz (C5 on the junction) is the closest implementable PDH-like baseline.

### C11. Temperature-sensor feedback and feed-forward

- **Sensor:** on-die diode / PTAT / thermistor near the ring.
- **Actuator:** heater.
- **Update law (feed-forward plus optical trim):**

```text
T_hat[k]  = F_s( T_sensor[k] )                           # sensor-to-ring lag/lead filter (Cauer model), calibrated
u_ff[k]   = u_0 - (T_hat[k] - T_0) * (dlambda_dT / eta_h)  # heater mW that cancels the ring shift
u[k]      = sat( u_ff[k] + u_fb[k] )                     # u_fb from any optical loop C2..C9 (slow trim)
```

- **Tuning parameters:** sensor gain and offset, filter F_s, trim-loop gains; per-ring gains if the gradient matters.
- **Failure modes:** the sensor is not at the ring (M1000 shows 63 K across an array, D9); sensor lag differs from ring
  lag; cannot see laser drift, self-heating, polarization or fab offsets; needs calibration of dlambda/dT (the value is
  disputed: 86 vs 62.5 pm/K, Section 1).
- **Published:** [DER10] Sandia: first MRM with both a resistive heater and a diode temperature sensor; sensor linear
  over > 85 degC (printed, abstract). [SAE15] Caltech: monolithic PTAT measuring ring temperature directly, closed loop at
  20 Gb/s under thermal perturbations (printed, abstract). [ZHA22] diode sensor plus CMOS loop, 30 to 100 degC tuning,
  simulation only; unity-gain bandwidth limited by slow heat transfer (printed, abstract). [HAT24] thermistor calibrated
  to 75.5 pm/degC vs 75.8 pm/degC in an oven, error < 0.4 degC over 80 degC (printed). No paper found that demonstrates
  pure sensor feed-forward with an optical trim on an MRM.

### C12. In-resonator sensing (photoconductive heater, CLIPP, defect PD, junction photocurrent)

- **Sensor:** intracavity power read without a tap: photoconductance of a doped heater [JAY15, JAY19], contactless
  capacitive probe (CLIPP) [GRI14], defect-mediated Si PD [PAD13], reverse-biased modulator junction [SHA22], p-i-n
  diodes in the ring [KRA24].
- **Actuator:** heater (sometimes the same element).
- **Update law:** C2 or C3 on the in-ring signal; the signal peaks on resonance like the G26 drop-port monitor.
- **Failure modes:** small signal (Kramnik: ~1 mA/W, printed); signal depends on bias and temperature; heater-sense
  multiplexing; in the photoconductive case the sensing current also heats.
- **Published:** [JAY15] UBC: n-doped heater photoconductance locks first- and second-order filters, 12.5 Gb/s with 5 degC
  at 0.28 degC/s (printed). [JAY19] 31 rings in a 16 x 16 switch and a 14-ring CROW aligned automatically (printed).
  [GRI14] CLIPP with CMOS readout: tuning, locking, labelling, swapping (printed, qualitative); a related GFP 2014 paper
  reports a 150 ms lock response (secondary). [SHA22] see C3.

### C13. Dual actuator: fast bias (or capacitive) inner loop + slow thermal outer loop

- **Sensor:** any of the above.
- **Actuators:** junction bias / capacitive tuning (fast, small range, nearly zero static power) and heater (slow, large
  range).
- **Update law (mid-ranging):**

```text
inner (fast):   v_b[k+1] = sat_b( v_b[k] + K_b * e[k] )            # holds lock, small range
outer (slow):   u[k+1]   = sat( u[k] + K_h * (v_b[k] - v_b_mid) )  # re-centres the bias by moving the heater
```

- **Tuning parameters:** inner and outer gains (outer bandwidth >= 10x below inner), bias range and centre.
- **Failure modes:** bias changes modulation depth and OMA (operating point moves on the transfer curve); limited bias
  range; loop interaction if bandwidths are too close.
- **Published:** [LI13] TAMU/HP ISSCC 2013 bias-based stabilization: bias tuning ~6.8 uW/GHz over 0.28 nm (numbers as
  quoted by [TIT14]). [TIT14] simulated dual-loop (coarse thermal + fine bias) up to 16x faster tuning (printed,
  simulation, assumes 77 us thermal tau and a 1 MHz FSM). [OMI24] Penn (Aflatouni group), JSSC: 8-ch 256 Gb/s
  monolithic 45 nm TX with ~400 pm capacitive tuning at near-zero power and autonomous per-channel locking (printed,
  abstract-only).

### C14. Multi-ring arbitration and crosstalk-aware control

- **Sensor:** per-ring monitors (C1/C2), sometimes RSSI only.
- **Actuator:** all heaters.
- **Policies** [CHOI]: Lock-to-Deterministic (exact order), Lock-to-Cyclic (any cyclic shift = barrel shift),
  Lock-to-Any (needs a crossbar or byte reorder). LtC needs less tuning range than LtD because a cyclic shift absorbs a
  global offset; LtA needs the least.
- **Sequential tuning baseline (printed in [CHOI]):** order rings by target position; tune each to the first available
  laser line; move to the next. Fails by "stealing" (a blue-shifted early ring takes a line meant for a later one),
  causing zero-lock, duplicate lock or lane-order errors when the tuning range is below the FSR.
- **Proposed algorithm [CHOI] (record + single-step matching):**

```text
for each ring i: sweep, record search table ST_i = heater codes at intracavity-power peaks
for each adjacent pair (i, j=i+1 mod N):                       # relation search
    aggressor i locks to one entry (lock-to-first, then lock-to-last); victim j re-sweeps
    the masked entry in ST_j gives the relation index RI(i,j)
build Lock Allocation Table by offsetting ST_i by RI and arranging in target order
assign every ring its aligned entry in one pass (no duplicates by construction)
```

- **Decoupling of heater crosstalk:** u = K^-1 * (desired temperature changes) with K the measured crosstalk matrix;
  Thermal Eigenmode Decomposition drives all heaters along eigenvectors of the coupling matrix and converges in fewer
  iterations than per-heater control [MIL19, printed abstract]. Princeton weight banks: model-based crosstalk
  calibration scales O(N) and reaches 3.8-bit feed-forward weight accuracy (printed, abstract) [TAIT16].
- **Bang-bang with hysteresis (simulation baseline):** [CO22-TCPMT]: every 10 ms, if T < T_ref - 0.1 degC add 0.01 mW,
  if T > T_ref + 0.1 degC subtract 0.01 mW (printed; simulation only, uses ring temperature as if it were measurable).
- **Tuning parameters:** sweep resolution, assignment policy, crosstalk matrix estimate and refresh rate, loop
  sequencing (sequential vs simultaneous).
- **Failure modes:** duplicate / zero lock, lane order errors, FSR variation, tuning-range variation, and crosstalk
  making independent loops fight (only if diagonal dominance is lost; measured crosstalk of 1 to 3 % keeps it,
  Section 2.4).
- **Published numbers:** [CHOI] simulation parameters (printed): 8 channels (16 also), 200 GHz = 1.12 nm grid at
  1300 nm, ring blue bias 4.48 nm, laser global variation +/-9 nm, laser local variation 25 % of grid, ring global
  +/-6 nm, ring local +/-2.24 nm, FSR 8.96 nm with 1 % variation, tuning-range variation 10 %; all "sigma" values are
  half-ranges of uniform distributions; 10,000 trials. Simulator: https://github.com/wdmsim/wdm-simulator (open source).
  Earlier: [GEO11] barrel shifter / bit-reorder mux to cut tuning range; [KRI11] Oracle, CMOS manufacturing to reduce
  tuning requirements; [HAT20] automated tuning and channel selection for cascaded rings.

### C15. Model-based and learning controllers (MPC, Kalman, extremum seeking, adaptive, RL)

- **What is published for silicon rings:**
  - MPC: [CHE24] FAU Erlangen (Cherian, Michalka, Murray, Roell, Graichen), Proc. SPIE 12891, 128910D (2024),
    "Control approaches for operating point stabilization of microring resonator modulators under fast perturbations".
    Only the bibliographic record was readable; whether it uses MPC is **not verified**. An FAU thesis advert from the
    same chair lists faster state estimation as a topic (secondary).
  - Kalman / state estimation: an extended-Kalman projection tracks ring resonances in biosensing with 2 to 3x lower
    3-sigma noise than Lorentz/Fano fits (ACS Sensors, printed abstract). No MRM-locking use found.
  - Extremum seeking: dither locking (C5) is ESC; the CLEO 2014 BPSK microring controller is an ESC example.
  - RL: one search result attributes a Q-learning controller that kept cascaded-ring Vernier filters aligned across the
    C-band for ~10 h to an unidentified paper; authors and venue could not be confirmed (treat as unverified).
- **Baseline to implement (derived design, not published):**

```text
state x = [T_fast_i, T_slow_i (Foster states per ring), T_xt (crosstalk states), d_amb, d_amb_rate]
model  x[k+1] = A x[k] + B u[k] + E w[k] ;  y[k] = h(x[k], lambda_laser) + v[k]    # nonlinear Lorentzian readout
EKF/UKF: estimate x, including the ambient ramp rate d_amb_rate (this is what lets the loop act like type 2)
MPC:   minimize sum ||lambda_res(x) - lambda_target||^2 + rho ||du||^2
       s.t. 0 <= u <= U_max, |du| <= slew (460 codes/ms, textbook), horizon ~ 5 tau_dominant
fallback: if innovation too large -> re-acquire (C1)
```

- **Tuning parameters:** model order, Q/R covariances, horizon, weights, constraint set.
- **Failure modes:** model mismatch (tau spread x0.5 to x2, crosstalk unknown), nonlinear readout (sign ambiguity at
  the peak, Section 2.3 pitfalls), computational cost on an FPGA, observability of the ambient vs self-heating
  (needs a second sensor or the bit statistics).

### C-summary. Published performance at a glance

| Method | Source | Lock range | Tracking / disturbance | Loop speed | Power / energy | Provenance |
|---|---|---|---|---|---|---|
| Bit statistics (C6) | [SUN15] / [SUN16] | 600 GHz (~60 K) / 524 GHz (>50 K) | random 1/0 balance every 200 ms | 76 kHz decisions, lock <= 7 ms | 275 fJ/bit / 0.74 mW logic; heater 3.8 uW/GHz | printed |
| Drop-port OMA (C7) | [AGA16] | 5 nm (~62.5 degC) | 5 nm/s (~62.5 degC/s), < 10 pm precision | - | low power (abstract) | printed, abstract-only |
| Dither (C5) | [PTC13] / [PD] | heater range | 3 K at 10 Hz | 1 kHz dither | ER cost 1.9 dB at 0.1 K | printed |
| Dither, inflection (C5) | [GRM23] | - | wavelength + thermal drift at 50 Gb/s | 30 ms response (FPGA) | - | printed, abstract |
| Homodyne (C9) | [COX14] | 1.25 THz laser sweep; 5 to 60 degC chip | - | integral-only | - | printed |
| BER (C8) | [ZOR13] | 32 K | - | slow | 55 fJ/bit forecast | printed via [P] |
| Mean power PI (C3) | [ZHE14] | > 500 GHz tracked | - | - | 20 fJ/bit controller, ~14 mW for 4.5 nm | printed |
| Junction photocurrent (C3/C12) | [SHA22] | - | < 0.05 dB TDECQ penalty over 10 degC | - | - | printed, abstract |
| P&O on RSSI (C2) | [HAT24] | > 1 FSR (~26 nm) heater | 3.1 degC/s, 75 degC range | FPGA | - | printed |
| Photoconductive heater (C12) | [JAY15] | - | 5 degC at 0.28 degC/s | - | - | printed |
| AWR search (C1) | [TIM12] | +/-10 degC (see C1) | - | 200 us lock-in | < 1 mW | printed |
| Industry: TeraPHY | [AYAR25] | 30 to 80 degC cycling | 800 degC/s emulated (laser) | - | - | printed (vendor) |
| Industry: Lightmatter G26 | [LM26-P], textbook | 0 to 105 degC junction (textbook) | 808 degC/s physical, >2000 degC/s (slides) | not published | ~25 mW/FSR heater (G25) | printed / read-off |

---

## 5. Disagreements with the textbook (and agreements worth recording)

1. **Rise times listed as time constants.** Ch. 11 (`src/chapters/Ch11.tsx`, table at lines ~1056-1061 with column
   header "Time constant", prose at lines ~1036, 1046, 2665, 2843) lists "~170 us" for undercut and "14 us" for the best
   conventional heater, citing [P]. Dong et al. print ~170 us as the **10-90 % switching time** [DONG10b abstract], i.e.
   tau ~ 77 us (inferred; [TIT14] uses 77 us). Watts et al. quote Gan et al.'s optimized over-clad heater as a 6.4 us
   **time constant** [WAT09]; 14 us = 2.2 x 6.4 us is its 10-90 % time (inferred). Our view: relabel the column or
   convert, and cite the primary papers (Watts 2009, Dong 2010a/b, Cunningham 2010, Gan 2007) rather than the review.
2. **Crosstalk magnitude and shape.** `REF.crosstalkNearest = 0.1`, `REF.crosstalkNext = 0.03`, farther entries zero
   (Ch. 11 crosstalk table, `crosstalkMatrix()` in `src/physics/index.ts` lines ~635-646). Measured: 1.3 to 2.6 % at
   100 to 300 um with an EIC on top, nearly flat with distance [CO24-JOM]; "a few percent" at 10 to 15 um [DER14].
   Our view: nearest-neighbour 10 % is a pessimistic corner, not a default; a long-range floor of ~1 to 2 % to **every**
   ring must be added (it is common-mode and grows with the number of active heaters). The textbook itself already
   argues that 10 % is "if anything, too high" (Ch. 11).
3. **Crosstalk dynamics.** `stepThermal()` (`src/physics/index.ts` lines ~661-676) applies the same fast/slow poles to
   the crosstalk terms as to the self term, so a neighbour heats with the 10 us dynamics. Simulated neighbour responses
   are first order with tau = 4.1 to 9.0 ms at 100 um pitch, and the neighbour 3 dB frequency is ~1000x lower than the
   heated channel's (30 Hz vs 30 kHz) [CO22-TCPMT]. Our view: crosstalk paths need their own (slower) poles; at 15 to 30 um
   pitch the near-field estimate is ~0.1 to 1 ms (derived, Section 2.4).
4. **Thermal time constants are no longer sourceless in general.** The textbook says no Lightmatter or OCP source gives
   any time constant (`REF.tauThUs` comment, `docs/SOURCES_AND_GENERATIONS.md` Section 5). That remains true for
   Lightmatter, but non-Lightmatter measured values now bracket the assumption: 1 us (interior heater), 5 to 10 us
   (metal above, no undercut), 9 to 38 us (undercut, measured-converted or simulated), ~77 us (fully released
   racetrack). The textbook's assumed 10 us agrees with the imec measured waveguide tau of 10 us (agreement). The
   assumed 300 us slow pole at 25 % has no direct support; the imec fits put 40 to 60 % of the DC gain in poles of
   13 to 55 us and only ~4 % above 100 us (probe-station boundary), with die/package poles at ms and beyond.
   [verified, clarified: "40 to 60 %" is per pole. No UCUT: 57 % in the 13.0 us pole. UCUT: 42 % at 25.6 us plus 48 % at
   54.5 us, i.e. 90 % combined in 25.6 to 54.5 us poles, and 4.1 % at 124 us.]
5. **Sun et al. range differs between versions.** Ch. 14 (lines ~3516, 5256, and the summary at ~5423) gives
   "600 GHz" for [S], which cites both VLSI 2015 and JSSC 2016. VLSI 2015 prints 600 GHz (3 nm, ~60 K) and 275 fJ/bit;
   JSSC 2016 prints 524 GHz (>50 K) and 0.74 mW logic. Report both.
6. **Review citation details.** The source lists of Ch. 00, 01, 02, 03, 05, 06, 09, 10, 11 and 14 give Padmaraju &
   Bergman as "Nanophotonics 2(4), 2013" (e.g. `Ch14.tsx` line ~5436, `Ch11.tsx` line ~4748). The journal's issue
   listing (nanophotonics-journal.com, volume 3, issue 4) and imec's citation in [CO22-JLT] give Nanophotonics vol. 3,
   pp. 269-281 (2014); 2013 is the online date of the author copy. Minor, but the textbook is strict about citations.
   [verified via Crossref for doi:10.1515/nanoph-2013-0013: vol. 3, issue 4-5, pp. 269-281, print Aug 2014, online
   11 Sep 2013. Corrected chapter list: the "2(4), 2013" form also appears in Ch. 12, 13, 15, 16, 17 and 18
   (16 chapters in all).]
7. **62.5 pm/K is a real measured O-band value, not only a bench conversion.** `REF.dLambdaDT_benchPmK = 62.5` is
   labelled "inferred from two printed numbers". imec measured 62.54 pm/K on O-band rings [CO24-JOM, Eq. 2], and Ayar
   Labs uses the same 50 nm/s = 800 degC/s mapping [AYAR25]. [corrected: [CO24-JOM] prints 62.54 pm/K "for the rings
   measured in the O-band" without stating how the value was obtained; call it printed, method unstated, not measured.
   Ayar's mapping is verified: 10 to 50 nm (per second, inferred) = 160 to 800 degC/s, i.e. 62.5 pm/K.] Our view: keep 86 pm/K (Lightmatter, printed) as the
   design value, but treat ~62.5 pm/K as a well-supported external O-band value and sweep 60 to 90 pm/K. This sharpens
   the textbook's open question (correction 1) rather than closing it.
8. **New public data on EIC-to-PIC thermal coupling.** `docs/SOURCES_AND_GENERATIONS.md` Section 5 lists photonic-
   electronic die crosstalk as "measured but embargoed" for Lightmatter. imec publishes a 3D EIC-on-PIC case: -43.3 %
   heater efficiency, +44.4 % ring-to-ring crosstalk [CO24-JOM]. [corrected: measured efficiency loss 42.4 % (UCUT) and
   20.6 % (no UCUT); the +44.4 % crosstalk rise is simulated, not measured. See Section 1, item 4.] Not Lightmatter data, but the right order of magnitude
   for M1000-style stacking.
9. **Industrial ASIC-step rate (agreement).** The textbook's [MRVL] 10 to 120 degC/s is secondary (SemiAnalysis). Ayar
   Labs independently prints ~100 degC/s peak at an optical engine 3 mm from a 0 -> 500 W ASIC step, with the XPU at
   400 degC/s [AYAR25]. This supports the textbook's correction 4 (the 800 degC/s figure is not the processor).

---

## 6. Open questions (to ask Lightmatter or to resolve by simulation)

1. Lightmatter MRM heater: type (metal above, silicide in/around the ring, doped Si), undercut or not, and the measured
   step response (tau of heater and waveguide nodes, number of poles). The GF PDK ring in [NAN23] has an undercut
   silicide heater; is the G25/G26 ring derived from it?
2. Ring pitch on the G26 TX bus and the measured crosstalk matrix (static and dynamic), with and without the stacked EIC.
3. What "period of 100 ms" means for the 808 degC/s test (one ramp or one up-down cycle), and whether the heater
   aggressor is driven as a power ramp or a step (D1).
4. Which dlambda/dT applies (86 pm/K printed vs ~62.5 pm/K measured elsewhere in the O-band), and at what temperature.
5. Absorbed-power fraction and self-heating thermal resistance of the MRM at the operating bias (sets the data-pattern
   disturbance size; Section 3.3).
6. Is the G26 lock signal the drop-port mean power (C3), a dither (C5), bit statistics (C6), or a combination? What is
   its update rate and the DAC slew limit actually used?
7. The full texts not readable here: [CHE24] (FAU, MPC?), [AGA16] (OMA monitor circuit and loop), [SUN16] (JSSC
   details), [GRM23] (inflection lock), [SHA22] (Intel photocurrent loop), [OMI24] (capacitive locking), DeRose 2014
   time-constant-vs-gap curve, Masood et al. imec heater comparisons.
8. Self-heating dynamics: [ARB12] says second order; [GRAY20] says log-like. Which form fits a 45 nm-class MRM?
9. Package-level poles (L3): need a CPO package step response from any vendor in peer-reviewed form.

---

## 7. References

Lightmatter and textbook sources are cited with the textbook's tags ([LM-P], [LM-T], [LM26-P], [LM26-S], [HC], [OCPW],
[MRVL], [OCI], [OCP]) and are listed in `docs/SOURCES_AND_GENERATIONS.md` of the textbook. Others:

- [AGA16] S. Agarwal, M. Ingels, M. Pantouvaki, M. Steyaert, P. Absil, J. Van Campenhout, "Wavelength locking of a Si
  ring modulator using an integrated drop-port OMA monitoring circuit," IEEE JSSC 51(10), 2328-2344 (2016).
  Abstract-level. imec record: https://imec-publications.be (search title).
- [ARB12] A. Arbabi, L. L. Goddard, "Dynamics of self-heating in microring resonators," IEEE Photonics J. 4(5),
  1702-1711 (2012), doi:10.1109/JPHOT.2012.2214205. Abstract-level.
- [ATA13] A. H. Atabaki et al., "Sub-100-nanosecond thermal reconfiguration of silicon photonic devices," Opt. Express
  21(13), 15706 (2013), doi:10.1364/OE.21.015706.
- [AYAR25] V. Stojanovic, "Let's Get Serious: TeraPHY Optical Engine Passes the Test for AI Scale-Up at Volume," Ayar
  Labs blog, 28 Aug 2025. https://ayarlabs.com/blog/lets-get-serious-teraphy-optical-engine-passes-the-test-for-ai-scale-up-at-volume/
- [BHA24] P. Bhargava (Ayar Labs), "A 256Gbps microring-based WDM transceiver with error-free wide temperature
  operation for co-packaged optical I/O chiplets," VLSI Symposium 2024, C22.1 (50 to 110 degC dynamic ramp; rate not
  found). Abstract-level.
- [CEL25] Converge Digest, "Hot Chips 2025: Celestial AI CTO details in-die optical I/O" (secondary).
  https://convergedigest.com/hot-chips-2025-celestial-ai-cto-details-in-die-optical-i-o/
- [CHE24] A. J. Cherian, A. Michalka, K. Murray, G. Roell, K. Graichen, "Control approaches for operating point
  stabilization of microring resonator modulators under fast perturbations," Proc. SPIE 12891, 128910D (2024),
  doi:10.1117/12.3000834. Bibliographic record only: https://cris.fau.de/publications/333720977
- [CHOI] S. Choi, V. Stojanovic, "Scalable Wavelength Arbitration for Microring-based DWDM Transceivers,"
  arXiv:2411.14810 v2 (Feb 2025). https://arxiv.org/abs/2411.14810 ; simulator https://github.com/wdmsim/wdm-simulator
- [CO22-JLT] D. Coenen, H. Oprins, Y. Ban, F. Ferraro, M. Pantouvaki, J. Van Campenhout, I. De Wolf, "Thermal modelling
  of silicon photonic ring modulator with substrate undercut," J. Lightwave Technol. 40(13), 4357-4363 (2022),
  doi:10.1109/JLT.2022.3162987. Author copy: https://lirias.kuleuven.be/retrieve/0266e645-ff06-4863-ac57-a65e77b7b785
- [CO22-TCPMT] D. Coenen, H. Oprins, P. De Heyn, J. Van Campenhout, I. De Wolf, "Analysis of thermal crosstalk in
  photonic integrated circuit using dynamic compact models," IEEE Trans. CPMT 12(8), 1350-1357 (2022),
  doi:10.1109/TCPMT.2022.3195699. Accepted version:
  https://imec-publications.be/bitstreams/9876a625-1df8-4d9b-baa5-c7e6e699d2ca/download
- [CO24-JOM] D. Coenen, M. Kim, H. Oprins, Y. Ban, D. Velenis, J. Van Campenhout, I. De Wolf, "Thermal modeling of
  hybrid three-dimensional integrated, ring-based silicon photonic-electronic transceivers," J. Opt. Microsyst. 4(1),
  011004 (2024, corrected Mar 2024), doi:10.1117/1.JOM.4.1.011004 (CC BY 4.0).
  https://lirias.kuleuven.be/retrieve/dfb9dc3b-e76f-4be4-ad9c-8583e785d51a
- [CO24-PH] D. Coenen, M. Kim, H. Oprins, J. Van Campenhout, I. De Wolf, "A critical analysis of the thermo-optic time
  constant in Si photonic devices," Photonics 11(7), 603 (2024), doi:10.3390/photonics11070603.
  https://imec-publications.be/bitstreams/ff18e9a9-7e8a-43d3-9296-4cb1b613ca88/download
- [CO-EM] D. Coenen, K. Croes, A. Tsiara, H. Oprins, V. Simons, O. Varela Pedreira, Y. Ban, J. Van Campenhout,
  I. De Wolf, "Electromigration performance improvement of metal heaters for Si photonic ring modulators," IEEE author
  manuscript (venue not verified). https://lirias.kuleuven.be/retrieve/680718
- [COX14] J. A. Cox, A. L. Lentine, D. C. Trotter, A. L. Starbuck, "Control of integrated micro-resonator wavelength via
  balanced homodyne locking," Opt. Express 22(9), 11279 (2014), doi:10.1364/OE.22.011279.
  https://www.sandia.gov/app/uploads/sites/145/2021/11/control_of_integrated_micro-resonator.pdf
- [CUN10] J. E. Cunningham et al., "Highly-efficient thermally-tuned resonant optical filters," Opt. Express 18(18),
  19055-19063 (2010), doi:10.1364/OE.18.019055. Abstract-level.
- [DCD24] DatacenterDynamics, "Lightmatter turns to GlobalFoundries to bring photonics interconnect to market in 2025"
  (Nov 2024): Passage produced on GF Fotonix. https://www.datacenterdynamics.com/en/news/lightmatter-turns-to-globalfoundries-to-bring-photonics-interconnect-to-market-in-2025/
- [DEC19] M. de Cea, A. H. Atabaki, R. J. Ram, "Power handling of silicon microring modulators," Opt. Express 27(17),
  24274 (2019), doi:10.1364/OE.27.024274. Abstract-level.
- [DER10] C. T. DeRose et al., "Silicon microring modulator with integrated heater and temperature sensor for thermal
  control," CLEO 2010, CThJ3, doi:10.1364/CLEO.2010.CThJ3. Abstract-level.
- [DER14] C. T. DeRose, N. J. Martinez, R. D. Kekatpure, W. A. Zortman, A. L. Starbuck, A. Pomerene, A. L. Lentine,
  "Thermal crosstalk limits for silicon photonic DWDM interconnects," IEEE Optical Interconnects 2014 (SAND2014-1156C).
  https://www.osti.gov/servlets/purl/1140533
- [DFB] Coherent DFB15 10 Gb/s DFB chip datasheet (0.09 nm/degC); Thorlabs DFB1310P spec sheet (0.1 nm/degC typ.).
- [DONG10a] P. Dong et al., "Low power and compact reconfigurable multiplexing devices based on silicon microring
  resonators," Opt. Express 18(10), 9852-9858 (2010), doi:10.1364/OE.18.009852. Abstract-level.
- [DONG10b] P. Dong et al., "Thermally tunable silicon racetrack resonators with ultralow tuning power," Opt. Express
  18(19), 20298-20304 (2010), doi:10.1364/OE.18.020298. Abstract-level.
- [GAN07] F. Gan et al., "Maximizing the thermo-optic tuning range of silicon photonic structures," Photonics in
  Switching 2007, pp. 67-68. Not read; numbers via [WAT09] and [P].
- [GEO11] M. Georgas, J. Leu, B. Moss, C. Sun, V. Stojanovic, "Addressing link-level design tradeoffs for integrated
  photonic interconnects," IEEE CICC 2011. Not read; cited via [CHOI].
- [GRAY20] D. Gray et al., "Thermo-optic multi-stability and relaxation in silicon microring resonators with lateral
  diodes," Phys. Rev. Applied 14, 024073 (2020); arXiv:2003.09978. Abstract-level.
- [GRI14] S. Grillanda et al., "Non-invasive monitoring and control in silicon photonics using CMOS integrated
  electronics," Optica 1(3), 129 (2014); arXiv:1405.5794. Abstract-level.
- [GRM23] Grimaldi, Zanetto, Toso, Perino, Petrini, Morichetti, Melloni, Ferrari, Sampietro (Politecnico di Milano),
  "Self-stabilized 50 Gb/s silicon photonic microring modulator" (2023); venue listed as JLT in citing works (not
  verified); abstract-level. https://re.public.polimi.it/handle/11311/1227512
- [HAT20] M. Hattink, Z. Zhu, K. Bergman, "Automated tuning and channel selection for cascaded micro-ring resonators,"
  Proc. SPIE 11308, 160-168 (2020). Not read; cited via [CHOI].
- [HAT24] M. Hattink, J. Robinson, L. Y. Dai, Z. Zhu, A. Novick, M. Glick, K. Bergman et al., "Wide temperature range
  uncooled 2.5D integrated silicon photonic DWDM receiver," GOMACTech 2024.
  https://lightwave.ee.columbia.edu/sites/default/files/content/docs/Papers/2024/GOMACTech_2024_RX_Temp_Control_4_page.pdf
- [HUA23] D. Huang et al. (Intel), "A silicon photonic 8 x 32 Gbps/lambda WDM transceiver with integrated laser array and
  SOA for optical I/O," IEDM 2023, paper 8-6. https://mys.mapyourshow.com/mys_shared/iedm23/handouts/8-6_Mon_15581.pdf
- [JAY15] H. Jayatilleka et al., "Wavelength tuning and stabilization of microring-based filters using silicon
  in-resonator photoconductive heaters," Opt. Express 23(19), 25084 (2015); arXiv:1507.00686. Abstract-level.
- [JAY19] H. Jayatilleka et al., "Photoconductive heaters enable control of large-scale silicon photonic ring resonator
  circuits," Optica 6(1), 84 (2019), doi:10.1364/OPTICA.6.000084. Abstract-level.
- [JSUN19] J. Sun, R. Kumar, M. Sakib, J. B. Driscoll, H. Jayatilleka, H. Rong (Intel), "A 128 Gb/s PAM4 silicon
  microring modulator with integrated thermo-optic resonance tuning," J. Lightwave Technol. 37, 110-115 (2019).
  Abstract-level via citing works.
- [KRA24] D. Kramnik et al., "Scalable feedback stabilization of quantum light sources on a CMOS chip,"
  arXiv:2411.05921 (2024). https://arxiv.org/html/2411.05921
- [KRI11] A. V. Krishnamoorthy et al., "Exploiting CMOS manufacturing to reduce tuning requirements for resonant optical
  devices," IEEE Photonics J. 3(3), 567-579 (2011). Not read; cited via [CHOI], [P].
- [LEE16] D. Y. Lee et al. (Oracle), "Error-free operation of a polarization-insensitive 4 lambda x 25 Gbps silicon
  photonic WDM receiver with closed-loop thermal stabilization of Si microrings," Opt. Express 24(12), 13204 (2016).
  Abstract-level.
- [LI13] C. Li et al., "A ring-resonator-based silicon photonics transceiver with bias-based wavelength stabilization
  and adaptive-power-sensitivity receiver," ISSCC 2013, pp. 124-125. Numbers via [TIT14].
- [MIL19] M. Milanizadeh, D. Aguiar, A. Melloni, F. Morichetti, "Canceling thermal cross-talk effects in photonic
  integrated circuits," J. Lightwave Technol. 37(4), 1325-1332 (2019). Abstract-level.
- [NAN23] R. Nandi et al. (GlobalFoundries), "Thermal stabilization of micro-ring modulator using a monolithically
  integrated analog feedback circuit," ECIO 2023.
  https://www.ecio-conference.org/wp-content/uploads/2023/05/MRM_thermal_stabilization_ECIO23.pdf
- [OMI24] K. Omirzakhov, H. Hao, A. Pirmoradi, F. Aflatouni, "Energy efficient monolithically integrated 256 Gb/s optical
  transmitter with autonomous wavelength stabilization in 45 nm CMOS SOI," IEEE JSSC, doi:10.1109/JSSC.2024.3511673.
  Abstract-level.
- [P] K. Padmaraju, K. Bergman, "Resolving the thermal challenges for silicon microring resonator devices,"
  Nanophotonics 3(4-5), 269-281 (2014; online 2013), doi:10.1515/nanoph-2013-0013. Author copy:
  https://www.ee.columbia.edu/~kishore/publications/NP_2013.pdf
- [PAD12] K. Padmaraju et al., "Thermal stabilization of a microring modulator using feedback control," Opt. Express
  20(27), 27999 (2012). Abstract-level.
- [PAD13] K. Padmaraju et al., "Integrated thermal stabilization of a microring modulator," Opt. Express 21(12), 14342
  (2013). Abstract-level.
- [PAR25] R. Parsons et al., "Highly uniform thermally undercut silicon photonic devices in a 300 mm CMOS foundry
  process," Sci. Rep. 15, 29906 (2025), doi:10.1038/s41598-025-14480-4.
- [PD] K. Padmaraju, D. F. Logan, T. Shiraishi, J. J. Ackert, A. P. Knights, K. Bergman, "Wavelength locking and
  thermally stabilizing microring resonators using dithering signals," J. Lightwave Technol. 32(3), 505-512 (2014).
  Full text not read; numbers from [PTC13] and [P].
- [PTC13] K. Padmaraju, D. F. Logan, J. J. Ackert, A. P. Knights, K. Bergman, "Initialization and stabilization of
  microring resonators for next-generation silicon photonic interconnects," TECHCON 2013.
  https://www.ee.columbia.edu/~kishore/publications/TECHCON_2013.pdf
- [SAE15] S. Saeedi, A. Emami, "Thermal stabilization of micro-ring resonator modulators through direct measurement of
  ring temperature," Opt. Express 23(17), 21875 (2015). Abstract-level.
- [SHA22] J. Sharma et al. (Intel), "Silicon photonic microring-based 4 x 112 Gb/s WDM transmitter with
  photocurrent-based thermal control in 28-nm CMOS," IEEE JSSC 57(4), 1187-1198 (2022). Abstract-level.
- [STO18] V. Stojanovic et al., "Monolithic silicon-photonic platforms in state-of-the-art CMOS SOI processes
  [Invited]," Opt. Express 26(10), 13106 (2018).
  https://sclaser.mit.edu/wp-content/uploads/2018/11/PubOpticsExpress_May07_2018.pdf
- [SUN15] C. Sun, M. Wade, M. Georgas, S. Lin, L. Alloatti, B. Moss, R. Kumar, A. Atabaki, F. Pavanello, R. Ram,
  M. Popovic, V. Stojanovic, "A 45nm SOI monolithic photonics chip-to-chip link with bit-statistics-based resonant
  microring thermal tuning," VLSI Circuits 2015.
  https://sclaser.mit.edu/wp-content/uploads/2015/08/VLSI-2015-45nm-SOI_Pub_6924411_June2015_RRam.pdf
- [SUN16] C. Sun et al., "A 45 nm CMOS-SOI monolithic photonics platform with bit-statistics-based resonant microring
  thermal tuning," IEEE JSSC 51(4), 893-907 (2016), doi:10.1109/JSSC.2016.2519390. Abstract-level.
- [TAIT16] A. N. Tait et al., "Multi-channel control for microring weight banks," Opt. Express (2016). Abstract-level.
- [TIM12] E. Timurdogan et al., "Automated wavelength recovery for microring resonators," CLEO 2012, CM2M.1.
  https://pmg.mit.edu/documents/2012Timurdogan-3.pdf (excerpts)
- [TIT14] A. Titriku, C. Li, A. Shafik, S. Palermo, "Efficiency modeling of tuning techniques for silicon carrier
  injection ring resonators," IEEE OIC 2014.
  https://people.engr.tamu.edu/spalermo/docs/2014_ring_resonator_tuning_modeling_titriku_oic.pdf
- [WAT09] M. R. Watts, W. A. Zortman, D. C. Trotter, G. N. Nielson, D. L. Luck, R. W. Young, "Adiabatic resonant
  microrings (ARMs) with directly integrated thermal microphotonics," CLEO 2009, CPDB10.
  https://pmg.mit.edu/documents/2009Watts-2.pdf
- [ZHA22] H. Zhang, Y. Yuan, Z. Zhang, B. Huang, H. Chen, "Integrated temperature sensor and close-loop control circuit
  for silicon microring modulator," Proc. SPIE 12314, 123140U (2022). Abstract-level.
- [ZHE14] X. Zheng et al. (Oracle), "A high-speed, tunable silicon photonic ring modulator integrated with
  ultra-efficient active wavelength control," Opt. Express 22(10), 12628 (2014). Abstract-level.
- [ZOR13] W. A. Zortman, A. L. Lentine, D. C. Trotter, M. R. Watts, "Bit-error-rate monitoring for active wavelength
  control of resonant modulators," IEEE Micro 33(1), 42-52 (2013). Numbers via [P].

---

## Verification log

Independent check, 2026-10-09. Method: the cited primary source was opened where it could be reached (author copies
and accepted versions from imec / KU Leuven, conference PDFs, the local OFC 2026 PDF, Crossref, arXiv). The numbers
were then recomputed in Python. Verdicts: **confirmed** (source and arithmetic agree), **corrected** (edited in place
above with a "[corrected: ...]" note), **not checked** (left as the original agent wrote it).

| # | Claim (where in this file) | Verdict | Evidence |
|---|---|---|---|
| V1 | G26 aggressor: 808 degC/s, 25 to >105 degC, "period of 100ms", BER sampled every 2 s stays at 1e-11, drop-port monitor flat; TEC sweep 35 to 105 degC over 31 h, BER < 1e-9; ~15 GHz/degC (D1, D4, Sec. 1) | confirmed | Local PDF of OFC 2026 M4B.3, Sec. 3.2 and Fig. 3 caption. 15 GHz/K x 1310^2/c = 85.9 pm/K. The arithmetic (808 x 0.1 s = 81 K) favours "period" = one ramp. The paper does not say so |
| V2 | c/lambda^2 conversions and 10-90 % = 2.197 tau (Sec. 0) | confirmed | Recomputed: 174.69 / 176.31 / 212.06 / 124.78 GHz/nm; ln 9 = 2.197 |
| V3 | [CO22-JLT]: 222 +/- 10 pm/mW (2.99 K/mW) no UCUT, 300 pm/mW (4.04) incomplete, 843 +/- 55 pm/mW (11.35 K/mW) full; heater / WG tau 4.92 / 10.38 us (model), 5 / 10 us (measured); UCUT 18.1 / 38.1 us (simulated); x3.08, x3.68; 15 um diameter (T1, Sec. 1 item 1) | confirmed | Author copy, Tables II-IV, Fig. 16, abstract. Coupling 58.4 % -> 84.5 % |
| V4 | [CO22-TCPMT] Foster poles 3.2 / 13.0 us (43 / 57 %) and 2.5 / 25.6 / 54.5 / 124 us; units 1 Ohm = 1 K/mW, 1 uF = 1 nJ/K (T2, Sec. 1 item 2) | confirmed (with nuance) | Accepted version, Table I. R x C recomputed exactly; R sums 1.523 and 4.798 K/mW match the printed 1.52 and 4.8 degC/mW. Nuance: it is an FE fit using the material set calibrated in [CO22-JLT]. This filter's step response was not measured |
| V5 | "63 % point ~10 us" of the 2-pole no-UCUT fit (Sec. 2.5) | **corrected** | Solved numerically: 63 % at 7.3 us (10 % at 0.60 us, 90 % at 22.7 us). The UCUT 4-pole 63 % point is 37.7 us, which agrees with the simulated 38.1 us of [CO22-JLT] |
| V6 | [CO22-TCPMT] crosstalk: first-order neighbour tau 4.14 to 8.99 ms (UCUT), first neighbour 60 um from the heat source, 30 kHz vs 30 Hz, 2.8 % cumulative, device 40 us / circuit 4 ms, -30 dB at 200 kHz, bang-bang 10 ms / 0.1 degC / 0.01 mW, 5 degC aggressor with 1 s tau (X2, L2, D15, C14) | confirmed | Accepted version, Table II, Sec. III-A to III-D. No-UCUT taus 4.09 to 8.87 ms (within 1.3 %) |
| V7 | [CO24-JOM] Table 2 (165 -> 131, 312 -> 180 pm/mW), Table 3 crosstalk (2.62 / 2.33 / 1.94 %, 1.65 / 1.43 / 1.32 %, simulated values), quasi-constant beyond 200 um, cold plate -49.9 % (T5, X1) | confirmed | Lirias author copy, Tables 2-3, Sec. 4.2 |
| V8 | "-43.3 % efficiency, +44.4 % crosstalk" presented as measured EIC effects (Sec. 1 item 4, T5, Sec. 5 item 8) | **corrected** | The measured drops are 42.4 % (UCUT) and 20.6 % (no UCUT) (Sec. 4.1.2). The 44.4 % comes from the calibrated FE model run with and without the EIC (Sec. 4.2). Crosstalk was measured only with the EIC on |
| V9 | "imec measured 62.54 pm/K on O-band rings" (Sec. 1 item 5, T5, Sec. 5 item 7) | **corrected** | Eq. 2 text: "62.54 pm/K for the rings measured in the O-band"; how the value was obtained is not stated. The paper also applies it to the C-band no-UCUT rings (its simulated K/mW and pm/mW rows give exactly 62.5 for all four cases) |
| V10 | R_th sweep lower bound 2.1 K/mW = 131 / 62.54 (Sec. 2.5) | **corrected** | The no-UCUT rings are C-band (Fig. 5 caption). At the C-band 74.2 pm/K implied by [CO22-JLT] Table II, 131 pm/mW = 1.8 K/mW; the paper's FE gives 2.14 K/mW. New bound: 1.8 K/mW |
| V11 | [CO24-PH] pure-thermal tau 5 / 9 / 18 us; apparent cooling 6 / 12 / 24 us, heating 4 / 4 / 4 us; simulated 0.3 / 4.8 and 0.6 / 66.2 us (T3, Sec. 1 item 1) | confirmed | Accepted version, Tables 1-2. Its Eq. 17 extracts as 2 lambda / (Q eta_h). That disagrees with its own "50 % of FWHM" text (lambda / (2 Q eta_h)). The report's reading follows the text |
| V12 | Dong 2010b: 2.4 mW/FSR, 10-90 % switching ~170 us -> tau ~77 us; [TIT14] uses 77 us; LI13 bias 6.8 uW/GHz over 0.28 nm, 1 MHz FSM, 16x (T10, Sec. 5 item 1, C13) | confirmed | Abstract via Semantic Scholar API; [TIT14] PDF Sec. II and abstract. 170 / 2.197 = 77.4 us |
| V13 | [WAT09]: tau 1 us from a 200 kHz 0.3 Vpp square-wave fit, 4.4 uW/GHz, 16 nm at 9 mW, 32.85 nm at 18 mW; Gan's optimized over-clad heater 28 uW/GHz with a 6.4 us time constant (T7, T8, Sec. 5 item 1) | confirmed | CLEO 2009 PDF (pmg.mit.edu). 14 us / 6.4 us = 2.19, so the 10-90 % reading of [P]'s 14 us is plausible (still inferred) |
| V14 | [SUN15]: 76 kHz decision rate, lock <= 7 ms, < 500 nA vs ~3.5 uA, 600 GHz (3 nm, ~60 K), 3.8 uW/GHz, 275 fJ/bit, 10 GHz/K, controller slower than the heater-to-ring tau, 200 ms 1/0 changes (T6, C1, C6, D21) | confirmed | VLSI 2015 PDF (sclaser.mit.edu). [SUN16] 524 GHz / 0.74 mW not checked (abstract not reachable) |
| V15 | Derived constraint 1: 2000 degC/s x 86 pm/K = 172 pm/ms; 1 kHz dither -> ~100 Hz loop -> 1.6 ms -> ~270 pm = 0.7 linewidth; ">= 100 kHz" dither needed (Sec. 1 item 9, Sec. 4.0) | confirmed (refined) | 172 x 1.59 = 274 pm = 0.73 x 374 pm. For the 5 pm budget the minimum is f_d ~55 kHz under the f_d/10 rule; 100 kHz includes ~2x margin |
| V16 | Derived constraint 2: 0.44 pm/code (13 nm / 29,800), ~394 codes/ms needed, ~5x shortfall (Sec. 1 item 9, Sec. 4.0) | **corrected** | The method mixes the G25 full heater range (13 nm at 1.5 V) with the G26 command span. The textbook (Ch11.tsx `CODES_PER_K`, `SLEW_PEAK`, `SLEW_EDGE`) maps 29,800 codes to the ~83 K excursion: 359 codes/K, 0.24 pm/code. That gives 460 codes/ms on average, ~720 codes/ms at 2000 degC/s and ~830 codes/ms at the 2300 degC/s edge. Shortfall vs 76 codes/ms is ~6x to ~11x |
| V17 | [AYAR25]: 0 -> 500 W, ~100 degC/s peak at an OE ~3 mm away, ~100 ms; XPU 400 degC/s; 10 to 50 nm sweep = 160 to 800 degC/s; 30 to 80 degC cycling (D5, D6, Sec. 1 item 6) | confirmed (caveat added) | Blog, 28 Aug 2025. The blog says the test "simulates" the ASIC step but does not say whether physically or numerically; OEs "up to 25mm away" in real systems |
| V18 | [PAR25] P_pi values and 10-90 % definition unknown (T13) | **corrected** | arXiv:2503.09009v2: P_pi values confirmed (26.136 -> 5.316, 100.01 -> 3.99, 63.11 -> 1.31, 43.46 -> 1.07 mW). The Methods define rise/fall as 10 %/90 % thresholds, so the disk tau is 5.0 -> 44 us and the MZI tau 5.7 us -> 0.84 ms (inferred). The paper contradicts itself on whether the disk transients were simulated or measured |
| V19 | [HAT24]: 3.1 degC/s (TEC-limited), 336 uV step, 75.5 vs 75.8 pm/degC (79 simulated), error-free 16 Gb/s over 75 degC, FSR ~26 nm (D10, C2, C11) | confirmed | GOMACTech 2024 PDF. C-band (DFB array around 1550 nm) |
| V20 | [PTC13] dither: 15 um radius, Q ~14,000, 1 kHz square wave, 0.1 / 0.2 K -> 1.9 / 4.8 dB ER loss, 3 K at 10 Hz, ~50 s initial lock (C5, D13) | confirmed | TECHCON 2013 PDF. Peak-slope arithmetic 94 / 188 K/s recomputed |
| V21 | [NAN23] GF 45 nm PDK MRM: 7.5 um radius, ~5 % drop coupling, undercut heater of four silicide resistors, 15.1 / 13.7 / 11.5 mA, TIA 5 kOhm / 1 GHz, 200 x 220 um, 1308.5 nm (T17, C3, Sec. 1 item 10) | confirmed | ECIO 2023 PDF. The "Lightmatter uses a similar device" step stays a hypothesis |
| V22 | [DER14]: 63.2 / 70.4 degC at 5 mW, "a few percent" at ~10 um (measured), > 15 um (simulated), metal line up to 2x, measured vs simulated differ ~x3 (T19, X3) | confirmed (header nuance) | OSTI PDF. The distances are disk-to-disk spacings (gaps), not centre-to-centre |
| V23 | Near-field d^2 law tau = 4 d^2 / (pi^2 alpha): 0.11 to 0.12 ms at 15 um, 1.7 to 2.0 ms at 60 um (Sec. 2.4) | confirmed | Same as [CO22-TCPMT] Eq. 4; recomputed. imec's calibrated oxide (k = 1, c_p = 635) gives alpha = 7.2e-7 m^2/s, slightly below the range used |
| V24 | Self-heating 40 / 101 / 202 pm (11 / 27 / 54 % of 374 pm) and ones-density sigma 0.07 % (Sec. 3.3) | confirmed | Recomputed |
| V25 | Padmaraju & Bergman = Nanophotonics 3(4-5), 269-281, 2014 (online 2013); textbook chapters with "2(4), 2013" (Sec. 5 item 6) | confirmed / chapter list **corrected** | Crossref for doi:10.1515/nanoph-2013-0013. The outdated form is in 16 chapters (00, 01, 02, 03, 05, 06, 09 to 18), not 10 |
| V26 | Resistive coupling failure "per Fig. 9" (Sec. 2.5 implementation note) | **corrected** | Fig. 9 shows the all-heaters-on failure (no coupling current flows). Fig. 11 shows far-neighbour underestimation with a single heater on |
| V27 | [CO-EM] O-band r = 5 um ring: 0.183 +/- 0.019 nm/mW, FSR 13.41 +/- 0.16 nm, heater 7.03 degC/mW, UCUT x2.15 to x2.81 (T4) | confirmed | Lirias author manuscript. Its model uses dn_eff/dT = 1.95e-4 /K at 1304 nm. The model pair 3.07 K/mW -> 0.205 nm/mW implies 66.8 pm/K |
| V28 | Textbook-derived G25 numbers (25.45 mW, 0.405 nm/mW, 14.1 uW/GHz, 4.7 K/mW, 55.9 Ohm) (Sec. 2.1) | confirmed | Matched against `REF` in `src/physics/index.ts` and recomputed |

Not checked here (left as written): [SUN16] 524 GHz / 0.74 mW; [STO18] 3.53 uW/GHz; [AGA16] details beyond the abstract
summary seen in search; [TIM12] +/-10 degC; [JSUN19]; [GRM23]; [CHE24]; [OMI24]; [DFB] datasheet coefficients;
[LM26-S] sawtooth values (taken from the textbook `REF`, read-off).

Effect on the plant model:
- Default no-UCUT self-path 63 % time is 7.3 us, not 10 us.
- R_th lower sweep bound is 1.8 K/mW.
- EIC-stacking efficiency loss is 20 to 42 % measured, depending on the undercut.
- The heater slew a controller must deliver is ~460 to ~830 codes/ms.
All other key coefficients checked here stand as written.
