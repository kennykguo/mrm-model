# 01: Lightmatter public corpus and process platform

Research note for the `mrm-model` plant model. Compiled 2026-10-09.

Scope: everything public that I could reach about Lightmatter's microring modulators (MRMs), their heaters,
monitors, drivers and control. Also the public stack data for the silicon-photonics process the rings are made
in, and a check of the textbook's `REF` table (`lightmatter-capstone/src/physics/index.ts`) against all of it.
The textbook repo was read, not modified.

**Provenance vocabulary** (same as `docs/SOURCES_AND_GENERATIONS.md` in the textbook):
`printed` = verbatim in the source. `read-off` = digitized from a published figure (this note's own
digitization; the method is in Section 8). `inferred` = arithmetic on printed values. `derived` = computed
by us with a model. `assumed` = no source.

**Generation tags** (textbook's): G25 = OFC 2025 Tu3J.6 single-wavelength test vehicle. G26 = OFC 2026 M4B.3
16-λ bidirectional link (Passage EVK50). EVK100 = 16-λ PAM4 demonstrator. L20 = shipping Passage L20.
M1000 = Passage M-series interposer. Guide = laser module. New tags used here: **PAT** = a Lightmatter patent
document (no hardware generation), **GF** = GlobalFoundries platform data (not Lightmatter),
**ext** = another group's device on the same platform family.

---

## 0. Key findings (short)

1. **The process is printed.** The Lightmatter-hosted OFC 2025 submission says the G25 designs were
   fabricated on the **GF 45SPCLO** platform (GF Fotonix). A Nov 2024 announcement says GF Fotonix makes
   Passage. Public GF data for 45SPCLO: **160 nm photonics SOI, 88 nm CMOS SOI, 2 µm BOX**, SOI and SiN
   waveguides, Ge-epi photodiodes, 9-level BEOL. The textbook's 400 × 200 nm Lightmatter waveguide
   (SMTAI render) conflicts with the 160 nm photonics SOI layer. [verified: LM-P §3.1 sentence and GF-IMAPS
   slide 4 line "Dual SOI thickness: 160nm photonics, 88nm CMOS; 2µm BOX" both re-read; note the 160 nm value
   has only this one public source]
2. **The GF PDK ring modulator** (GF, ECIO 2023) has a 7.5 µm radius (15 µm diameter, the same as
   Lightmatter's "diameter ~15 µm"). It is add-drop, has about 5 % drop-port coupling and a Ge PD at the
   drop port, and uses an undercut heater made of four silicide resistors. This matches the G26 topology. It is
   evidence that the G26 MRM is close to the GF PDK ring (inferred, not confirmed). [verified against the ECIO
   2023 text: 7.5 µm radius, "approximately 5% coupling power at the drop-port", "four silicide resistors -
   two inside and two outside"]
3. **Six Lightmatter patent documents describe ring control.** US20240405908 covers heater-only tuning and
   remapping all rings by one channel ("barrel shift") when heater power becomes excessive.
   US11409045 / US11686902 cover a nA-class integrating ADC and a heater drive of 8-bit DAC plus 8-bit PWM
   (about 16 bit), and imply a thermal time constant above 2.56 µs. US12470304 (published as US20230308188)
   covers a heater ramp that maximizes a monitor detector, then MZI and add-filter ramps, then TX–RX pairing by
   per-pair dither tones from 1 kHz to 1000 kHz. US12355492 covers a drop-port PD on every TX ring for
   recalibration and modulator FSR < 11.44 nm. WO2025122797 covers wavelength-set remapping.
   US12147078 covers a polarization locker. [verified for US20240405908, US11409045, US12470304 and
   US12355492 by OCR of the USPTO image PDFs; WO2025122797 and US12147078 not re-checked]
4. **Two textbook read-offs do not hold up.** The G25 heater-sweep spectra are **windowed traces about 1 nm
   wide**, so the figure gives **no FSR bound**; `fsrMeasuredBoundNm = 14.1` is unsupported. The "0.9 Vpp
   on the bench" is the **pattern-generator swing into the on-chip driver**, not the MRM junction swing; the
   driver is rail-to-rail by design. [verified: both re-checked against the embedded slide-8 image and the
   LM-P text]
5. **The textbook splits one experiment in two.** The 30.5 h cycling run appears in the paper as 35–105 °C
   (TEC) and in four other Lightmatter documents as 25→105→25 °C. The identical BER traces show it is the
   same experiment. The plotted trace spans about 33–108 °C (read-off).
6. **"800 °C/s" labels two different experiments.** The Sep 2025 blog uses it for a 50 nm/s laser sweep.
   The Oct 2025 SMTAI and PECC decks use it for the on-die heater plot, which is identical to the paper's
   808 °C/s figure.
7. **The OFC 2026 slide-10 aggressor, re-digitized.** T ≈ 25 → 113 °C. dT/dt is a sawtooth of about
   −2200 → +2200 °C/s, period ≈ 130.5 ms (7.66 Hz). The mean ramp is about 1350 °C/s. The heater DAC swings
   about 20–22 k codes per lane, the lane offset is 7.5–9.5 k codes, and the peak slew is about 700 codes/ms
   (the mean per lane is about 340 codes/ms). [verified by independent re-digitization: T 24.6 → 112.6 °C,
   peaks every 130.2–131.1 ms, red dT/dt −2130 … +2170 °C/s, bottom lane 19.6 k → 41.8 k, top lane ≈ 29.5 k →
   49.2 k. Shape note: the dT/dt ramps are concave, not linear; T fits T_max − ΔT·|sin(πt/P)| (see D7)]
8. **Two energy figures and two line rates for G26.** The SiPh die is **1.7 pJ/bit** in the paper and
   **2.6 pJ/bit "photonics"** in the slides and blog. The rate is 50 Gb/s NRZ in the paper and **53.125 Gbaud
   NRZ** in "Wide and Parallel Wins". Both pairs are reported here and not reconciled. [corrected: both pairs
   are most likely different scopes, not disagreements. SIB prints "4.6 pJ/bit, inclusive of the 2.6 pJ/bit for
   the photonics and external light source (Guide)", so 2.6 includes the laser while the paper's 1.7 is the SiPh
   die alone (laser ≈ 0.9 pJ/bit, inferred). 53.125 Gbaud is the standard line rate of a nominal 50 Gb/s lane
   (53.125/50 = 17/16, the IEEE 802.3 50G FEC + transcoding overhead); the paper's measured PRBS rates were
   50 Gb/s and 56 Gb/s. See D15]

---

## 1. Corpus inventory

| Tag | Document | Date | Where | What it gives the plant model |
|---|---|---|---|---|
| LM-P | R. Baghdadi, C. Dorta-Quinones et al., "Monolithically Integrated Microring Transmitter and Receiver for High-Density 3D Co-Packaged Optics", OFC 2025 Tu3J.6. **Submission version** (Lightmatter-hosted, pdf title "OFC 2025 submission.docx") | Apr 2025 | lightmatter.co/wp-content/uploads/2025/04/OFC-2025-submission.pdf | Process (GF 45SPCLO), 1.8 THz design FSR, 0.44 nm/mW heater, ER > 18 dB, 30.9 ± 2 GHz EO BW, driver and AFE specs, 4 dBm input, 4.9 dB modulation IL, heater-sweep spectra Fig. 1(e) |
| LM26-P | R. Anand, …, N. Kumar*, …, N. Harris, "16-Wavelength 800-Gbps Bidirectional Link over Single-Mode Fiber Using Microring Transceivers", OFC 2026 M4B.3 (IEEE Xplore copy, local file) | Mar 2026 | `~/Downloads/16-Wavelength_800-Gbps_…(1).pdf` | ~15 GHz/°C, add-drop MRM with drop-port monitor, 808 °C/s, 31 h TEC, ±20 GHz laser, energy, Fig. 2–3 traces |
| LM26-S | N. Kumar, OFC 2026 talk deck (18 slides, "16 lambda Bidi/PAM4") | Apr 2026 | lightmatter.co/wp-content/uploads/2026/04/OFC-2026-16-lambda-Bidi_PAM4-Nikhil-Kumar.pdf | Diameter ~15 µm, resistive heater, depletion PN; relOMA maps (slide 9); aggressor, DAC codes, 30 h plot (slide 10); 66.8 GHz S21 (slide 8) |
| YOLE | N. Kumar, Yole Group market briefing at OFC 2026 | Apr 2026 | …/2026/04/OFC-2026-Yole-Nikhil-Kumar.pdf | EVK50 live demo (4+4 band, > 1500 °C/s, 25–105 °C), Guide ±15 GHz demonstrated, L20 |
| QDSOA | A. Sludds et al., OFC 2026 M1B.1 deck, QD-SOA in an 8λ 50G NRZ CPO link | Apr 2026 | …/2026/04/OFC-2026-presentation-9dB-…-Alex-Sludds.pdf | G26-hardware link budget: laser 10 dBm/λ, TP2 −2 dBm avg/λ, 12 dB TX path loss |
| HC | D. Bunandar, Hot Chips 2025, Passage M1000 (34 slides) | Aug 2025 | …/2025/08/57-lightmatter_hotchips_2025_final.pdf | Same ring slide; relOMA maps; "~1 mW" MRM power; bump pitch ~120 µm; 8λ @ 56 G per link; laser 16λ, 100 mW/fiber |
| SMTAI | J. Rosenberg, SMTA International 2025 | Oct 2025 | …/2025/10/2025-SMTAI-Womens-Leadership-Session-DFX.pdf | 400 × 200 nm waveguide render (slide 15), 3 µm pitch (slide 16), 800 °C/s **on-die heater** label (slide 19), "FAB: GF & TSMC" for fiber-attach options (slide 29) |
| PECC | D. Bunandar, Optica PECC Summit | Oct 2025 | …/2025/10/PECC-Darius-Bunandar.pdf | MRM "~10 µm" (slide 5); same 800 °C/s on-die label (slide 9) |
| SIB | "Seeing is Believing" blog (signed N. Harris) | 5 Sep 2025 | lightmatter.co/blog/seeing-is-believing-… | MRMs 5–15 µm diameter, ~1 mW; 50 nm/s laser sweep over 4 nm "representing" 800 °C/s; 4.6 pJ/bit wall-plug incl. 2.6 photonics |
| WPW | N. Harris, "Wide and Parallel Wins" blog | 15 Mar 2026 | lightmatter.co/blog/wide-and-parallel-wins/ | 53.125 Gbaud NRZ; "800 °C/sec and beyond"; ring-to-wavelength sequencing; ELS ≤ 1 MHz, ≥ 30 dB SMSR |
| BIDI | "One Fiber, Two Directions" technical brief | Sep 2026 | …/2026/09/BiDi-ScaleUp-Tech-Brief_Sept-26.pdf | L20 1311/1331 nm; every port programmable to transmit on either λ |
| HOTI26 | N. Harris, Hot Interconnects 2026 deck | Aug 2026 | …/2026/08/Nick-Harris_…_HotI2026.pdf | L20 30 W TDP; Guide VLSP 3 GHz frequency accuracy, 0.25 dB power accuracy |
| HOTI25, ISES | N. Harris HotI 2025 deck; R. Jain ISES Korea 2025 deck | Aug 2025 | …/2025/08/Nick-Harris-Lightmatter-HOTI-Deck-2025.pdf; …/ISES-Korea-08-2025_Ritesh-Jain.pdf | "5 dB transmit eff. loss"; 25→105→25 °C; 0–105 °C |
| ARXIV | A. Madhavan, P. Carson, T. Groves, T. Graham, arXiv:2609.01821 (HotI 2026) | Sep 2026 | arxiv.org/abs/2609.01821 | §III-A: MRMs "just 100 microns beneath" the SerDes; 4.3 pJ/bit (2.3 pJ/bit PIC + laser) |
| PAT-1 | US20240405908A1, "Techniques for efficient tuning of micro-ring modulators for WDM", J. Rosenberg, A. Kopa | prio. 2023-06-01, pub. 2024-12-05 | patents.google.com | Heater-only tuning; remap every ring by one channel when power is excessive |
| PAT-2 | US11409045B2 / US11686902B2 (WO2021003221A1), "Photonics stabilization circuitry", C. Dorta-Quinones, C. Ramey, O. O. Yildirim, C. Ravi, S. Gupta, N. C. Harris | prio. 2019-07-02 (provisional titled "Ring/Disk Resonator Stabilization") | patents.google.com | Integrating ADC; 8-bit DAC + 8-bit PWM heater drive; timing |
| PAT-3 | US20230308188A1 → US12470304B2, "Photonic communication platform and related circuits", C. Dorta-Quinones, R. Braid, A. Kopa, …, N. C. Harris, D. Bunandar | prio. 2022-03-28, granted 2025-11-11 | patents.google.com | Ramp-and-maximize ring lock; dither pairing 1–1000 kHz |
| PAT-4 | US12355492B2 (US20250105921A1), "Spectrally interleaved optical transceivers", K. Liu, B. Shah, S. Sane, J. Rosenberg, N. Kumar, A. Kopa, C. Dorta-Quinones, S. Klinger, D. Bunandar, N. C. Harris, S. A. Srinivasan, E. Greenwald | prio. 2023-09-27, granted 2025-07-08 | patents.google.com | Separate TX/RX buses; drop-port PD per TX ring; FSR < 11.44 nm; MZI interleaver loop |
| PAT-5 | WO2025122797A1 / US20250192912A1, "Wavelength remapping in an on-chip WDM solution", N. Kumar, S. A. Srinivasan, D. Bunandar | prio. 2023-12-07 | patents.google.com | Butterfly network that remaps wavelength sets |
| PAT-6 | US12147078B2, "Polarization locker for fiber connections", D. Bunandar | prio. 2022-11-11, granted 2024-11-19 | USPTO | One end active, the other passive |
| PAT-7 | US20260095014A1 "Laser system with active wavelength control"; US20260095015A1 / WO2026072846A1 "… integrated wavelength control" | prio. 2024-09-27 | Google Patents index | **Titles only**: Google Patents returned HTTP 503 on every attempt |
| GF-IMAPS | GF, "Establishing a packaging ecosystem for GlobalFoundries leadership silicon photonics wafers", IMAPS 2022 (slides) | 2022 | imapsource.org (article 91148) | 45SPCLO stack: 160/88 nm dual SOI, 2 µm BOX |
| GF-BIAN | Y. Bian (GF), "300-mm Monolithic Silicon Photonics Foundry Technology", Optica webinar | Oct 2025 | optica.org webinar pdf | Losses, ring BW > 35 GHz, sealed-undercut heaters, PDK model features |
| GF-ECIO | R. Nandi, …, A. Aboketaf, …, M. Rakowski (GF), "Thermal Stabilization of Micro-ring Modulator using a Monolithically Integrated Analog Feedback Circuit", ECIO 2023 | Apr 2023 | ecio-conference.org | **GF 45 nm PDK MRM**: R = 7.5 µm, add-drop, ~5 % drop, 4 silicide undercut heaters |
| BU-45CLO | K. Al Qubaisi, A. Khilo, H. Gevorgyan, M. Popović, "Microring Modulators in a New Silicon Photonics-Optimized 45 nm … Platform", CLEO 2021 | 2021 | par.nsf.gov/servlets/purl/10284518 | First rings on 45CLO: Q, FSR, GHz/V, radius |
| EUPR, CMC | Europractice and CMC pages for GF 45SPCLO | live pages | europractice-ic.com; cmc.ca | BEOL, voltages, substrate, MPW |
| Not reached | The 17-slide OFC 2025 talk [LM-T]; ECTC 2026 Ahmed et al.; IEEE-final version of Tu3J.6; Rakowski OFC 2020 full text; Bhargava VLSI 2024 full text | — | paywalled or not found | `REF` values that rest only on these are marked "not re-checked" |

All Lightmatter URLs are on `lightmatter.co` / `lightmatter.com` (the same site).

---

## 2. Full parameter table

Columns: name | value | unit | range or spread | generation | provenance | exact source. Where two sources
disagree, both rows are kept.

### 2.1 Platform and geometry

| Name | Value | Unit | Range | Gen | Prov. | Source |
|---|---|---|---|---|---|---|
| Process | GF 45SPCLO (Fotonix), 45 nm RF-SOI CMOS + photonics, monolithic | — | — | G25 | printed | LM-P §3.1 ("fabricated in the GF 45SPCLO platform") |
| Process for G26 MRMs | not printed; the paper says the MRMs are driven by "monolithically integrated CMOS drivers" | — | — | G26 | inferred: same monolithic platform | LM26-P §2 |
| Foundry for Passage | GF Fotonix selected to mass-produce Passage | — | — | M1000/L-series | printed (press) | Converge Digest, 19 Nov 2024 |
| Other fab mentioned | "FAB: GF & TSMC" (fiber-attach option table) | — | — | — | printed | SMTAI slide 29 |
| Photonics SOI thickness | 160 | nm | — | GF | printed | GF-IMAPS slide 4 |
| CMOS SOI thickness | 88 | nm | — | GF | printed | GF-IMAPS slide 4 |
| BOX thickness | 2 | µm | — | GF | printed | GF-IMAPS slide 4 |
| Lightmatter waveguide render | 400 wide × 200 tall | nm | — | (SMTAI, unspecified product) | printed (render labels) | SMTAI slide 15 |
| MRM diameter | ~15 | µm | — | G26 slide, HC | printed | LM26-S slide 7; HC slides 15–16 ("Yes, ~15µm in diameter") |
| MRM diameter (blog) | 5–15 | µm | — | — | printed | SIB |
| MRM size (deck) | ~10 | µm | — | — | printed | PECC slide 5 |
| GF PDK MRM radius | 7.5 | µm | — | GF | printed | GF-ECIO |
| Waveguide pitch on die | 3 | µm | — | — | printed | SMTAI slide 16; also PAT-3 |
| TX area / RX area | 0.0028 / 0.0032 | mm² | — | G25 | printed | LM-P §2 |
| TX/RX area, M1000 slide | 0.015 | mm² | — | M1000 | printed | HC slide 14 |
| Bump pitch / size | ~120 / ~80 | µm | — | M1000 | printed | HC slide 14 |
| MRM to SerDes (vertical) | 100 | µm | — | M1000/Passage | printed | ARXIV §III-A |

### 2.2 Spectral and optical

| Name | Value | Unit | Range | Gen | Prov. | Source |
|---|---|---|---|---|---|---|
| Design FSR | 1.8 (= 10.30 nm at 1310) | THz | — | G25 | printed (nm: inferred) | LM-P §2 |
| Measured FSR | **not published** | — | — | G25/G26 | — | Fig. 1(e) traces are windowed (Section 6, D2) |
| Modulator FSR bound in patent | < 11.44 (~2 THz) | nm | — | PAT-4 | printed | US12355492 |
| Channel plan, G26 | 2 × 8 λ, 400 GHz grid centred at 1310 nm, offset 200 GHz | — | — | G26 | printed | LM26-P §2 |
| Laser comb, read-off | λ1 ≈ 1301.5 to λ16 ≈ 1318.9 | nm | ±0.1 | Guide/M1000 | read-off | HC slide 17 |
| Patent example grid (8 ch) | 1301.47, 1302.60, 1303.73, 1304.87, 1306.01, 1307.14, 1308.28, 1309.43 | nm | — | PAT-4 | printed | US12355492 |
| Heater-sweep resonances 0 → 1.5 V | 1301.89, 1301.97, 1302.18, ~1302.55, 1303.05, 1303.69, 1304.44, ~1305.3, 1306.26, 1307.29, 1308.41, 1309.59, 1310.85, 1312.13, 1313.48, 1314.89 | nm | ±0.03 | G25 | read-off | LM26-S slide 8 embedded image (same data as LM-P Fig. 1(e)) |
| Notch depth vs heater V (0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.4, 1.5 V) | 27.9, 29.8, 29.8, 24.4, 22.1, 19.8, 18.0, 16.9 | dB | ±0.5 | G25 | read-off | same image |
| DC ER claim | > 18, "voltage-dependent" | dB | — | G25 | printed | LM-P §3.1 |
| Modulation IL | ≈ 4.9 at ≈ 4 dBm input | dB | — | G25 | printed | LM-P §3.2 |
| Transmit penalty, MRM | < 5 | dB | — | HC/HOTI25 | printed | HC slide 16 table; LM26-S slide 9 ("<5dB transmit eff. loss") |
| Q | 3500 (talk) | — | 3900–4200 (talk linewidth) | G25 | **not re-checked** (talk not reached) | textbook [LM-T] |
| Q, GF/45CLO rib ring (context) | 5000 loaded | — | — | ext | printed | BU-45CLO |
| Group index, 45CLO rib ring (R = 8 µm, w = 410 nm) | 3.91 | — | — | ext | inferred (FSR 8.6 nm, R 8 µm, λ ≈ 1300 nm) | BU-45CLO [verified: 3.909 at 1300 nm; 3.97 if λ = 1310 nm or if the printed "1.5 THz" is used, so 3.91–3.98] |
| Strip neff / ng, 400 × 160 nm, λ = 1310 | 2.36 / 4.04 | — | crude | — | derived (effective-index method, straight guide) | this note [verified: semi-vectorial FD, oxide clad, Li-1980 Si + Malitson SiO2, 10 nm grid → 2.360 / 4.07] |
| Strip neff / ng, 400 × 200 nm | 2.53 / 4.16 | — | crude | — | derived (same) | this note [verified: same FD → 2.531 / 4.18] |

### 2.3 Modulation and driver

| Name | Value | Unit | Range | Gen | Prov. | Source |
|---|---|---|---|---|---|---|
| Junction | depletion PN | — | — | G26 slide | printed | LM26-S slide 7; HC slide 15 |
| Driver | single-ended inductorless CMOS inverter chain, 50 Ω input termination, rail-to-rail output = supply | — | — | G25 | printed | LM-P §2 |
| Driver output swing | 1.3 (simulated, at 1.3 V supply) | Vpp | — | G25 | printed | LM-P §2 |
| Pattern-generator swing into the driver | 0.9 (through ~6 dB cable/PCB loss at Nyquist) | Vpp | — | G25 bench | printed | LM-P §3.2 |
| MRM junction swing during bench test | **not printed**; about the supply by design | Vpp | — | G25 | inferred | LM-P §2 + §3.2 |
| EO 3 dB BW | 30.9 ± 2 at −1 V (population) | GHz | ±2 | G25 | printed | LM-P §3.1 |
| EO 3 dB BW, plotted devices | 31.8 at −1 V; 35.4 at −2 V (legend); caption "near 34.1" at −2 V | GHz | — | G25 | printed | LM-P Fig. 1(f); HC slide 15 [verified by OCR of the LM-P figure: legend "-1V, 31.8 GHz", "-2V, 35.4 GHz"; caption "near 34.1 GHz for a -2V bias"; §3.1 "30.9 ± 2 GHz for a bias voltage of -1V"] |
| EO 3 dB BW, newer slide | 66.8 at −2 V reverse bias, "detuning 4 dB", RF 0 dBm, "Reticle 0, device 25"; curve shows ≈ +2 dB peaking near 35–40 GHz | GHz | — | **labelled with the Tu3J.6 citation** | printed (peaking: read-off) | LM26-S slide 8 [verified: header "Reticle 0, device 25, RF power 0 dBm, reverse bias -2 V, detuning 4 dB", marker "3-dB BW 66.8 GHz", footer "R. Baghdadi et al., OFC 2025. Paper Tu3J.6"; S21 rises from ≈ −42 to ≈ −40 dB before roll-off] |
| Modulation efficiency | 50 (talk) | pm/V | 52.8 avg | G25 | not re-checked | textbook [LM-T] |
| Bias axis of the controlled maps | 1.0 to 3.75 | V | — | G26 slide | printed (axis) | LM26-S slide 9; HC slide 16 |
| Data rate | 50 Gb/s NRZ per λ (eyes and BER curves at 56 Gb/s PRBS9) | Gb/s | — | G26 | printed | LM26-P §1, §3 |
| Data rate (blog) | 53.125 Gbaud NRZ | Gbaud | — | G26 | printed | WPW [corrected: not a conflict with 50 Gb/s; 53.125 Gbaud is the line rate of a nominal 50 Gb/s lane (×17/16 FEC + transcoding overhead, inferred). WPW calls the same demo "800 Gbps on a single fiber"] |

### 2.4 Heater (actuator)

| Name | Value | Unit | Range | Gen | Prov. | Source |
|---|---|---|---|---|---|---|
| Heater type | "resistive heater for tuning" | — | — | G26 slide | printed | LM26-S slide 7 |
| Heater efficiency | 0.44 | nm/mW | — | G25 | printed | LM-P §3.1 [verified: "The integrated heaters provided tuning of 0.44 nm/mW"] |
| Thermal tuning at 1.5 V | 13.0 (1301.9 → 1314.9) | nm | ±0.05 | G25 | read-off | LM26-S slide 8 image |
| Heater P per FSR, V per FSR, Ω, mA | 25.45 mW, 1.195 V, ~56 Ω, 21.4 mA | — | wafer maps | G25 | **not re-checked** (talk) | textbook [LM-T] |
| Heater sign | heaters can only add heat; cooling only from the chip | — | — | PAT-1 | printed | US20240405908 |
| Heater command, G26 | "MRM Stabilizer Value (DAC)" codes 19.2 k to 49.3 k over the aggressor | code | — | G26 slide | read-off | LM26-S slide 10 |
| DAC width | ≥ 16 bit (codes exceed 2^15 = 32768) | bit | — | G26 | inferred | LM26-S slide 10 |
| Lane-to-lane offset (8 lanes, same aggressor) | 7.5 k at maxima, 9.5 k at minima | code | — | G26 | read-off | LM26-S slide 10 |
| Per-lane swing for ~88 °C | 20 k (top lane) to 22.3 k (bottom lane) | code | — | G26 | read-off | LM26-S slide 10 |
| Code per °C | 227 to 253 (mean); ≈ 320 near the cusp (low codes) | code/°C | — | G26 | inferred from read-off | LM26-S slide 10 |
| Heater slew | mean ≈ 340 per lane over a half period; peak ≈ 700 just after the temperature cusp | code/ms | ±15 % | G26 | read-off | LM26-S slide 10 [verified: checker's re-digitization gives bottom lane 19.6 k → 41.8 k (22.2 k swing, 341 codes/ms mean), lane offsets ≈ 7.4 k at maxima and ≈ 9.4–10 k at minima, and 580–620 codes/ms averaged over 1–6 ms after each cusp (95th percentile of \|slope\| 660), consistent with a ≈ 700 codes/ms instantaneous peak] |
| Heater drive architecture (patent) | 8-bit DAC + 8-bit PWM (1/256 steps over a 2.56 µs period), about 16 bit total; code dithering N/N+1 | — | — | PAT-2 | printed | US11409045 [verified: "controlled in 1/256 increments over a 2.560 us period", "e.g., 16 bits"] |
| Control precision (patent) | 0.001 °C; ~1 part in 50 000 of applied power | — | — | PAT-2 | printed | US11409045 [verified] |
| GF PDK MRM heater | undercut heater, four silicide resistors (two inside, two outside the ring) | — | — | GF | printed | GF-ECIO |
| GF PDK MRM heater currents in lock examples | 11.5, 13.7, 15.1 | mA | — | GF | printed | GF-ECIO |
| GF ring-filter heaters | "high efficiency sealed undercut (airgap) heaters" | — | — | GF | printed | GF-BIAN |
| MRM power class | "~1 mW"; "as little as 1 mW" | mW | — | HC/SIB | printed | HC slide 16 table; SIB |

### 2.5 Thermal plant and disturbances

| Name | Value | Unit | Range | Gen | Prov. | Source |
|---|---|---|---|---|---|---|
| Resonance T-sensitivity | ~15 GHz/°C = 85.9 pm/K at 1310 nm | GHz/°C | — | G26 | printed (pm: inferred) | LM26-P §1 [verified: printed as an aside, "intrinsic thermal sensitivity of MRMs (~15 GHz/°C)"; 15 GHz × λ²/c = 85.86 pm. Physics cross-check (derived): a 400 × 160 nm oxide-clad strip at 1310 nm gives ≈ 61 pm/K with dn_Si/dT = 1.86e-4 /K, ≈ 64 pm/K with 1.94e-4 /K (300 K); see 02_materials.md for the 71 pm/K ceiling. The printed value is ~1.3–1.4× above a strip-ring estimate] |
| Bench equivalence | 50 nm/s ÷ 800 °C/s = 62.5 | pm/K | — | 2025 | inferred | SIB |
| Thermal time constant | **not published**. PAT-2 needs τ_th > 2.56 µs PWM period (design premise) | µs | > 2.56 | PAT-2 | inferred lower bound | US11409045 |
| Ring-to-ring thermal crosstalk | not published; PAT-1 description treats it as negligible (paraphrase, not re-verified verbatim) | — | — | PAT-1 | — | US20240405908 [verified verbatim, ¶[0048]: "In other words, thermal crosstalk is negligible." ¶[0050] adds that the heated MRM's "elevated local temperature decays away from the device". This is a design assumption, not a measurement] |
| Junction range with loop closed | 0–105 | °C | — | G26/HC | printed | LM26-S slide 9; HC slide 16; HOTI25 |
| Slow cycle (paper) | TEC, 35 → 105 °C, 31 h, 8 lanes, BER < 1e-9 at 50 Gb/s | — | — | G26 | printed | LM26-P Fig. 3(a) |
| Slow cycle (same plot, other labels) | 25 → 105 → 25 °C (ΔT = 80 °C), "over 30 hours" | — | — | G26 | printed | SMTAI slide 19; PECC slide 9; SIB; HOTI25 |
| Slow cycle trace | 33 → 108 °C, ≈ 90 cycles, period ≈ 20.3 min, ≈ 30.4 h, ramp ≈ 0.124 °C/s | — | ±1 °C, ±3 cycles | G26 | read-off | LM26-S slide 10 left |
| On-chip aggressor (paper) | 808 °C/s, 25 → > 105 °C, period 100 ms, one MRM, local resistive heater | — | — | G26 | printed | LM26-P §3.2 [verified: "a temperature ramp rate of 808°C/s within the MRM … ramped up and down between 25oC and >105oC with a period of 100ms"; caption says "between 25°C and 105°C"] |
| On-chip aggressor (same plot, 2025 label) | 800 °C/s "using an on-die heater aggressor" | °C/s | — | 2025 | printed | SMTAI slide 19; PECC slide 9 |
| Laser-sweep emulation | 50 nm/s over 4 nm "representing" 800 °C/s; TX and RX heaters follow a trapezoid of period ≈ 0.44 s (0.08 s ramps inferred) | — | — | 2025 | printed / read-off | SIB Fig. 1 |
| Fast aggressor (slide) | T ≈ 25 → 113 °C; dT/dt sawtooth ≈ −2200 → +2200 °C/s; period ≈ 130.5 ms (7.66 Hz); rise ≈ 65 ms, fall ≈ 65 ms; mean ≈ 1350 °C/s | — | ±1 °C, ±100 °C/s, ±1 ms | G26 | read-off | LM26-S slide 10 ("On Chip Thermal Aggressor") [verified numbers; corrected shape: the T trace fits T(t) = 113.6 − 88.5·\|sin(π(t − t0)/130.7 ms)\| °C with 0.34 °C rms (read-off + derived fit). A parabola (linear dT/dt) fits 5× worse (1.6 °C rms) and would need a 2780 °C/s peak. The rectified-sine form gives peak \|dT/dt\| = πΔT/P ≈ 2130 °C/s and mean 2ΔT/P ≈ 1350 °C/s, matching the plotted red trace] |
| Drop-port error during the slide-10 aggressor (added by checker) | signed error ≈ −0.3 % while T rises, ≈ +0.4 % while T falls; spikes to ≈ +3.0 / −1.9 % at the cusps | % | ±0.3 % | G26 | printed (panel title "MRM Drop-Port PD Error Signal", axis "Signed % Error", legend "L0 … L7 % Error") / read-off (values) | LM26-S slide 10: third panel of the embedded image (PDF image object 184), cropped out of the visible slide |
| Fast aggressor (headline) | > 2000 | °C/s | — | G26 | printed | LM26-S slide 10; YOLE |
| Live demo | > 1500 °C/s; 25–105 °C sweep | — | — | EVK50 demo | printed | YOLE |
| Fig. 3(b) inset sampling | 35 plotted peaks between 73.0 and 79.8 s: apparent spacing ≈ 0.20 s, peak heights 95–113 °C. Against the printed 100 ms period, the trace is under-sampled or aliased | — | — | G26 | read-off | LM26-P Fig. 3(b) |
| BER in fast test | text: "stable at 10^-11"; caption: < 1e-9; plotted line ≈ 8e-13 (inset axis read-off); sampled every 2 s | — | — | G26 | printed / read-off | LM26-P §3.2, Fig. 3(b) |

### 2.6 Sensing (monitor)

| Name | Value | Unit | Range | Gen | Prov. | Source |
|---|---|---|---|---|---|---|
| TX lock sensor | monitor PD at the MRM **drop port** | — | — | G26 | printed | LM26-P §3.2 |
| MRM topology | add-drop: input, thru, drop (κ_t, κ_d); heater and PN inside the ring | — | — | G26 | printed (schematic) | LM26-S slide 7 |
| G25 MRM topology | single-bus symbol in the schematic, **but** the slide that cites Tu3J.6 shows an input/thru/drop micrograph | — | — | G25 | printed; conflict | LM-P Fig. 1(b); HC slide 15; LM26-S slide 8 |
| Monitor current during fast aggressor | ≈ 15.5, flat | µA | ±0.5 | G26 | read-off | LM26-P Fig. 3(b) [verified: green trace sits just above the 15 µA tick in both panels; text: "monitor photocurrent at the MRM drop port (green) remained stable"] |
| GF PDK MRM drop coupling | ≈ 5 % "coupling power at the drop-port" | — | — | GF | printed | GF-ECIO |
| GF PDK loop TIA | 5 kΩ gain, 1 GHz BW; analog integrator to current mirror to heater; lock on the rising (blue) edge of the drop peak | — | — | GF | printed | GF-ECIO |
| RX polarization monitors | "Single-Polarization Photodetector" ADC codes: IN7 ≈ 7.5 k – 41.5 k; IN0 ≈ 15 k – 54 k | code | ±0.5 k | G26 | read-off | LM26-P Fig. 2(e); LM26-S slide 6 |
| Same, text | photocurrent swings between maximum and zero | — | — | G26 | printed | LM26-P §3.1 |
| Monitor ADC (patent) | current-mode dual-slope integrating ADC with analog baseline subtraction; nA sensitivity; C_int 10–20 pF (16 pF); T_int 1–10 µs (2 µs = 400 × 5 ns); Vref ≈ 500 mV | — | — | PAT-2 | printed | US11409045 |
| Calibration vector rate (patent) | every 10 µs, 100 µs or 1 ms | — | — | PAT-2 | printed | US11409045 |
| AFE (RX, G25) | 86 dBΩ, 16 GHz, 6.4 pA/√Hz, 80 µA average corner (all simulated) | — | — | G25 | printed | LM-P §2 |
| Ge PD BW | 56 at −1 V | GHz | — | G25 | printed | LM-P §1, §3.1 |

### 2.7 Link, power, laser

| Name | Value | Unit | Range | Gen | Prov. | Source |
|---|---|---|---|---|---|---|
| relOMA reference | 1 mW of laser into the MRM | — | — | HC | printed | HC slide 16 note |
| relOMA, loop closed | −2.1 (≈ 1.0–1.15 V) … −3.0 (≈ 3.15 V at 0 °C, ≈ 3.4 V at 105 °C) | dB | < 1 dB spread | G26 | printed contours / read-off positions | LM26-S slide 9 |
| relOMA, loop open | peak ≈ 69.6 °C; −8 dB contours at ≈ 66.8 and 72.4 °C; −16 dB at ≈ 62 and 74.5 °C; −24 dB at ≈ 57 and 79 °C (asymmetric) | °C | ±0.5 °C | G26 | read-off at 2.25 V | LM26-S slide 9 |
| RX sens. at BER 1e-9, 56 G PRBS9 | best ≈ −14.7, worst ≈ −12.5 | dBm OMA | ±0.3 | G26 | read-off | LM26-S slide 6(b) = LM26-P Fig. 2(b) |
| Chip-to-chip optical loss | ~3 (6 connectors, 1 km SMF28) | dB | — | G26 | printed | LM26-P §3.1 |
| Laser into TX, G26 link budget | 10 | dBm/λ | — | G26 hardware | printed | QDSOA slide 4 |
| TX output TP2 | −2 avg (12 dB laser→TP2) | dBm/λ | — | G26 hardware | printed | QDSOA slide 4 |
| PD input | −10 OMA | dBm/λ | — | G26 hardware | printed | QDSOA slide 4 |
| Input power at ring, G25 | ≈ 4 | dBm | — | G25 | printed | LM-P §3.2 |
| Laser module | 16λ, 100 mW/fiber (→ 8.0 dBm/λ if 16 λ share it; 11.0 dBm/λ for 8 λ) | — | — | M1000/Guide | printed (dBm: inferred) | HC slide 33; SMTAI slide 38 |
| Guide VLSP | > 100 mW per fiber, 32 fibers, 16 λ, 200 GHz grid ± 20 GHz | — | — | Guide | printed | YOLE |
| Laser accuracy | ±20 GHz needed; ±15 GHz demonstrated; 3 GHz (VLSP claim) | GHz | — | G26/Guide | printed | LM26-P §4; YOLE; HOTI26 [verified; note: LM26-P §4 says the ±20 GHz requirement was "achieved in our laser system"; the YOLE slide prints "ACCURACY OF +/- 15 GHZ" as a Guide VLSP performance claim, without the word "demonstrated"] |
| Guide line peaks on OSA | ≈ +10 to +12 | dBm | OSA RBW unknown | Guide | read-off | HC slide 17 |
| Energy, G26 paper | 3.8 total (1.7 SiPh die + ~2.1 SerDes), ~3 W | pJ/bit | — | G26 | printed | LM26-P §4 |
| Energy, G26 slides/blog | 2.6 "photonics" + 2 XSR SerDes; 4.6 wall-plug | pJ/bit | — | G26 | printed | LM26-S slide 4; YOLE; SIB [corrected scope: SIB prints "inclusive of the 2.6 pJ/bit for the photonics and external light source (Guide)", so 2.6 includes the laser; the paper's 1.7 is the SiPh die only. Implied laser share ≈ 0.9 pJ/bit (inferred); the deck and YOLE print "2.6 PJ/BIT PHOTONICS" without the qualifier] |
| Energy, arXiv | ≈ 4.3 (2.3 PIC + laser) | pJ/bit | — | Passage | printed | ARXIV §III-A |
| Energy, G25 | 84.4 mW Tx+Rx → 1.51 pJ/bit at 56 Gb/s | — | — | G25 | printed | LM-P §4 |
| L20 | 1311/1331 nm, 32 × 200 G PAM4, 6.4 Tb/s per direction, 30 W TDP max, 5 pJ/bit | — | — | L20 | printed | BIDI; HOTI26; YOLE |

---

## 3. What patents and public documents say about the control loop

### 3.1 Lightmatter's own statements about the loop (papers, slides, blogs)

- **Sensor.** In G26 the transmitter's stabilization sensor is a monitor photodiode at the MRM's own drop
  port. During the 808 °C/s aggressor its current stays flat at about 15 µA (LM26-P §3.2, Fig. 3(b)). So the
  loop holds the **drop-port photocurrent constant**, which is direct evidence of a set-point regulator on a
  drop-port signal. [verified, and strengthened by the checker: the slide-10 embedded image (PDF object 184)
  has a third panel, cropped out of the visible slide, titled "MRM Drop-Port PD Error Signal" with axis
  "Signed % Error" and one trace per lane (L0–L7). During the > 2000 °C/s aggressor the error stays within
  ≈ −1.9 … +3.0 %, with a small offset that flips sign with the ramp direction (≈ −0.3 % while heating,
  ≈ +0.4 % while cooling; read-off). A ramp-direction-dependent offset is consistent with a loop that has a
  single integrator (inferred, not confirmed)]
- **Actuator.** There is one resistive heater per ring, inside the ring (LM26-S slide 7 schematic). The
  firmware variable is "MRM Stabilizer Value (DAC)". All 8 lanes move together, offset by 7.5–9.5 k codes
  (LM26-S slide 10).
- **Initialization.** "Initialization and stabilization algorithms automatically locked the MRMs and MRRs
  to the input laser wavelength" (LM-P §3.2, paraphrased). The 2026 link is "initialized and stabilized
  automatically" (LM26-P §2). WPW says the MRMs need intricate sequencing that assigns unique rings to specific
  wavelengths, and claims eight years of MRM-control work. The TEC is not required for operation (LM26-P §2).
- **What is never published:** loop rate, dither frequency or amplitude, set-point definition (peak or
  fraction of peak), filter structure, calibration storage, the search algorithm, and the temperature to which
  the 15 GHz/°C figure refers.

### 3.2 Patents (Lightmatter assignee)

**PAT-1, US20240405908A1** (Rosenberg, Kopa; priority 2023-06-01; pending). Efficient tuning for WDM.
- Each MRM has a phase shifter (heater or PN) for coarse tuning. Ring diameters may be deliberately varied so
  that the unbiased resonances do not overlap.
- Heaters only add heat. The controller holds each ring's **local** temperature constant while the
  **global** chip temperature moves.
- **Algorithm.** (702) Apply a first biasing condition that maps rings to wavelengths, for example A→λ1, B→λ2,
  C→λ3, D→λ4. (704) Measure the power needed to hold it. "Excessive" means above a threshold, or the local
  temperature is outside an allowed range. (706) If excessive, lower the heater power so each ring moves one
  channel and the map rotates, for example A→λ2, B→λ3, C→λ4, D→λ1. The wrap-around ring moves to another
  resonance order. (708) Re-check and repeat.
- The needed ΔT is about N times the temperature for a one-channel shift, but it is not known in advance, so a
  feedback loop confirms the shift.
- FIGs. 5–6 (eight MRMs) show the trade-off. If the chip warms and then cools quickly, rings whose ΔT exceeds
  the heater range drop out. A smaller ΔT range per ring widens the usable chip temperature range.
- No numbers are given: no bits, no FSR value, no power and no time constants. Sensor options are thermopile,
  IR sensor, thermistor, ammeter or thermocouple.
- [verified by OCR of the USPTO publication: ¶[0050] "A heater may only increase temperature, not decrease
  it"; ¶[0048] "thermal crosstalk is negligible". Added by checker: ¶[0050] also says it "may be desirable to
  bias the MRMs at local temperatures that are above the maximum allowed chip temperature". For the plant
  model this means the locked ring set-point sits above the hottest expected chip temperature, so the heater
  always has headroom to back off]

**PAT-2, US11409045B2 and continuation US11686902B2** (priority 2019-07-02, provisional titled "Ring/Disk
Resonator Stabilization"; Dorta-Quinones, Ramey, Yildirim, Ravi, Gupta, Harris).
- A photodiode on an "observation port" of the ring or disk sees almost no light when the ring is tuned and
  more light when it is detuned. This is a 2019-era minimum-seeking monitor. Its sign handling is not
  described.
- **ADC.** Current-mode dual-slope integrating ADC with analog baseline (offset) subtraction. Output
  D = ceil[N·(i_det − i_os,det)/(i_ref − i_os,ref)] with N = 400 clocks. C_int 10–20 pF (16 pF example),
  T_int 1–10 µs (2 µs = 400 × 5 ns), Vref ≈ 500 mV, nA sensitivity, photocurrent "on the order of micro amperes".
- **Heater drive.** A low-precision 8-bit DAC plus 8-bit PWM: 1/256 steps over a 2.56 µs period, about
  16 bit total. Fast code dithering between N and N+1 is averaged by the thermal low-pass. This requires the
  thermal time constant to exceed the PWM period.
- **Targets.** 0.001 °C and ~1 part in 50 000 of applied power. Calibration vectors every 10 µs, 100 µs or 1 ms.
- [verified by OCR of the USPTO grant: "1/256 increments over a 2.560 us period", "provided that the thermal
  time constant of the system exceeds the total PWM period", "N=400 5-ns clock cycles, or … 2 us",
  "Cint … (e.g., 16 pF)", "10-20 pF", "1-10 us", "DC reference voltage (e.g., 500 mV)", "every 10 us, every
  100 us, every 1 ms", "one part per 50000", "within 0.001 degrees Celsius", and "When the resonator is
  perfectly tuned, no light will be detected by the photodiode". Grant date 2022-08-09]
- *Note (inferred, not confirmed):* the G26 DAC codes reach about 49 k, which needs ≥ 16 bit. That is
  consistent with this 8 + 8 bit scheme, but nothing public says the G26 hardware uses it.

**PAT-3, US20230308188A1 → US12470304B2** (priority 2022-03-28; granted 2025-11-11; Dorta-Quinones, Braid,
Kopa, Gould, Bowman, Buckenmaier, Stadolnik, Gupta, Carr, Harris, Bunandar). Passage platform circuits.
- **TX lock.** (FIG. 3-4A) A linear ramp drives the ring heater; the controller monitors detectors 412 + 414
  (their sum) and parks the heater at the value that **maximizes** them, i.e. a peak search on the dropped light.
  (FIG. 3-4B) A second ramp sets an MZI so that nearly 100 % of the modulator output goes the right way
  (minimize detector 414). (FIG. 3-4C) A third ramp tunes a second-order **add filter**, minimizing
  detector 413 or 411. These steps are claimed (claims 7–10).
- **RX lock.** (FIG. 3-5A/B) A ramp drives the drop-filter heater to maximize detector 425 and/or minimize
  detector 424, then an MZI ramp follows.
- **TX–RX pairing.** (FIG. 3-6A/B, claims 1–4) Dither a TX component and the matching RX drop filter at the
  same frequency, **1 kHz to 1000 kHz**, with a different frequency for each pair. The frequencies must not be
  rational fractions of one another. The dither is supported by the thermo-optic heater. Detection is by
  homodyne or heterodyne circuits or a programmable FIR band-pass filter.
- [verified by OCR of US12470304B2: claim 4 "the first frequency is between 1 KHz and 1000 KHz"; claims 7–9
  are the heater, MZI and add-filter ramps with maximize/minimize; claim 10 "second order filter"; earliest
  provisional 63/324,598 filed 2022-03-28; granted 2025-11-11. Note claim 7 maximizes "the first monitoring
  detector"; the sum 412 + 414 is from the description]
- **Other content.** Lasers and rings drift thermally; two-photon absorption is named as an
  intensity-dependent shift. A ring modulator can be under 5 µm diameter with Q > 10^5 (a generic statement).
  The reticle-tile interposer has a 3 µm waveguide pitch.

**PAT-4, US12355492B2** (priority 2023-09-27; granted 2025-07-08). Spectrally interleaved transceivers.
- TX and RX each have their own bus, joined by a multi-stage asymmetric MZI interleaver. The interleaver is
  locked by monitor PD → ADC → controller → DAC → phase shifters, keeping the monitored leakage below about
  10 %.
- Each TX modulator **may have a drop port and photodetector**; the controller uses that signal to decide
  whether to recalibrate the modulator's resonance.
- RX filters are second-order coupled rings (flatter top, less sensitive to thermal drift).
- **FSR of the modulators < 11.44 nm (~2 THz).** Same-set spacing 1.72–2.86 nm (300–500 GHz). Example grid
  1301.47–1309.43 nm at ~1.14 nm. [verified by OCR of US12355492B2: claim text "free spectral ranges (FSR)
  that are less than 11.44 nm"; the description frames it as a practical limit ("the maximum FSR that can be
  achieved in practice is less than about 11.44 nm, corresponding to about 2 THz in the O-band"). 11.44 nm =
  1.998 THz at 1310 nm; 1.72 / 2.86 nm = 300 / 500 GHz; 1.145 nm = 200 GHz. Granted 2025-07-08; first
  provisional 63/585,923 filed 2023-09-27]
- An optional polarization controller (PBS, TM→TE converter, delay, coupler) sits in front of the
  interleaver.

**PAT-5, WO2025122797A1 / US20250192912A1** (priority 2023-12-07; Kumar, Srinivasan, Bunandar). An on-chip
log2(N)-stage butterfly of directional couplers (N = 16 example) remaps wavelength sets to a different spatial
order, for example AAAA…BBBB → ABAB…. It does not describe ring locking.

**PAT-6, US12147078B2** (priority 2022-11-11; Bunandar). A polarization locker at each end of a non-PM fiber.
The controller sets one locker active and the other passive, so the two ends never need to exchange phase
information. This bears on the textbook's "possibly polarization controllers" extension. Only the front page
was read; the full text is a scanned image.

**PAT-7** (priority 2024-09-27): "Laser system with active wavelength control" and "… integrated wavelength
control". This is the Guide second servo. **Titles only**: Google Patents returned HTTP 503 on every attempt.

### 3.3 Synthesis: the loop as public data constrains it

```
 laser comb (±20 GHz spec, ±15 demonstrated)        on-die / package / laser-sweep aggressors
        │                                                       │ (≤2200 °C/s peak, 7.66 Hz; 50 nm/s; 0.12 °C/s)
        ▼                                                       ▼
 [bus] ─► MRM_k (add-drop, PN + in-ring resistive heater) ─► thru (data)
                    │ drop (~few % coupling)                     ▲
                    ▼                                            │ heater P(code), heat-only, ≥16-bit code
             Ge monitor PD ≈ 15 µA ─► ADC ─► controller (per ring, 8 rings/bus) ─┘
                                            │  init: ramp-and-maximize (PAT-3), ring↔λ assignment,
                                            │  remap by one channel if power excessive (PAT-1)
                                            │  run: hold drop-port current constant (LM26-P Fig. 3b)
                                            └─ pairing/diagnostics: per-pair dither 1–1000 kHz (PAT-3)
```

What the plant model therefore needs:
- A **unipolar heater input** in DAC codes. Use a 16-bit code with a power law to be decided; the
  slide-10 shape does not settle P ∝ code against P ∝ code². [verified: fitting the bottom-lane code to
  a + b·|sin| (P ∝ code) gives 393 codes rms; fitting it to sqrt(a + b·|sin|) (P ∝ code²) gives 306 codes rms.
  The square law is slightly better, but both residuals are < 2 % of the swing, so the shape alone does not
  decide it]
- A **drop-port photocurrent output** through an ADC with offset.
- Per-ring **fabrication offsets** large enough to give the 7.5–9.5 k-code lane spread.
- **Resonance-order wrap**, so that a barrel shift by one channel is possible.
- Three aggressor generators:
  - a parabolic-segment on-die waveform at 7.66 Hz, 25 → 113 °C; [corrected: use a rectified-sine waveform,
    T(t) = T_max − ΔT·|sin(πt/P)| with T_max ≈ 113.6 °C, ΔT ≈ 88.5 °C, P ≈ 130.7 ms (cusp at T_max, rounded
    minimum). It fits the slide-10 trace to 0.34 °C rms and reproduces both the ≈ ±2130 °C/s peak and the
    ≈ 1350 °C/s mean. A parabola with the same swing would peak at ≈ 2780 °C/s]
  - a laser-wavelength trapezoid at 50 nm/s over 4 nm, period ≈ 0.44 s;
  - a slow triangle of about 33–108 °C with a 20 min period.
- A dither path at kHz to MHz.

### 3.4 Public loops on the same platform family (not Lightmatter; for context only)

- **GF, ECIO 2023 (45 nm GF PDK MRM).**
  - Ge PD at the drop port, then a TIA (5 kΩ, 1 GHz BW), then a difference amplifier against Vref, then an
    integrator whose time constant matches the heater's thermal response, then a current mirror into the
    heater.
  - It locks anywhere on the rising (blue-side) edge of the drop peak and cannot lock above the peak
    (Vref > 0.65 V).
  - Laser at 1308.5 nm; heater currents 11.5–15.1 mA. [verified against the ECIO 2023 text: TIA "5kOhms gain
    and 1GHz bandwidth"; "lock the resonance at any point of the left-hand side of MRM resonance (the rising edge
    of the MRM drop port)"; "heater current reducing after Vref > 0.65V"; points A/B/C = 15.1/13.7/11.5 mA]
- **Xilinx/AMD, US20240405882A1** (priority 2023-06-05; not Lightmatter, despite the adjacent publication
  number).
  - Uses a drop-port PD and a TIA IDAC as a DC-power proxy. The calibration sweeps the heater over about
    200 GHz with a DC-balanced 1010 pattern, finds IDAC_MAX, then locks at K·IDAC_MAX with K = 3/4 (the
    maximum-slope point of a Lorentzian). [math verified: d²/dx²[1/(1+x²)] = 0 at x = 1/√3, where
    1/(1+x²) = 3/4; patent text not re-checked]
  - In mission mode it tracks IDAC continuously. Carriers are 100 GHz apart (200 GHz after de-interleaving).
- **Kramnik et al., arXiv:2411.05921 (45RFSOI, not 45SPCLO).**
  - The in-cavity p-i-n photocurrent comes from defect and surface absorption, about 1 mA/W. A 10-bit heater
    DAC gives an LSB of about 0.6 pm.
  - A PI loop locks at 95 % of the way from dark current to peak photocurrent after a hot-to-cold sweep.
  - The loop rate is about 10 Hz in Python, with about 2 kHz projected on an FPGA and about 60 kHz with custom
    logic.

---

## 4. Process stack: GF 45SPCLO (GF Fotonix)

### 4.1 Identification

| Evidence | Prov. | Source |
|---|---|---|
| "The designs were fabricated in the GF 45SPCLO platform" (G25) | printed | LM-P §3.1 |
| "45nm CMOS SiPh process", monolithic driver and AFE | printed | LM-P §1 |
| GF selected to mass-produce Passage on Fotonix (Nov 2024), building on about 7 years of collaboration | printed (press) | Converge Digest 2024-11-19 |
| G26 MRMs "independently driven by monolithically integrated CMOS drivers" | printed; platform inferred | LM26-P §2 |
| Fiber-attach options list "FAB: GF & TSMC" | printed | SMTAI slide 29 |

Conclusion: G25 is GF 45SPCLO (printed). G26 is very probably the same monolithic platform (inferred). Future
products may also use TSMC; nothing public says which PIC is built where.

Naming: 45CLO (OFC 2020) and 45SPCLO / GF Fotonix are the same lineage. GF's own documents use both names for
the 45 nm RF-SOI + photonics platform (inferred from GF-IMAPS, GF-BIAN and the CMC page).

### 4.2 Public stack data

| Item | Value | Prov. | Source |
|---|---|---|---|
| Wafer / fab | 300 mm, Fab 8, Malta NY | printed | Europractice; GF-BIAN |
| SOI thickness, photonics | **160 nm** | printed | GF-IMAPS ("Dual SOI thickness: 160nm photonics, 88nm CMOS") |
| SOI thickness, CMOS | 88 nm | printed | GF-IMAPS |
| BOX | **2 µm** | printed | GF-IMAPS |
| Waveguide types | Si "ridge" and Si "rib" (partial etch exists), SiN waveguide in middle-of-line | printed | GF-IMAPS slide 4; BU-45CLO; GF-BIAN |
| Slab / partial-etch depth | **not public** | — | — |
| Losses | SM SOI ridge −1.0 dB/cm; SOI rib 0.73 dB/cm; SM SiN 0.39 dB/cm | printed | GF-BIAN |
| Doping | multiple implant layers, including a shallow implant that allows vertical PN (BU) | printed | BU-45CLO |
| Ge PD | Ge epi coupled into the SOI waveguide; > 55 GHz; angled design > 0.95 A/W, < 30 nA, ~60 GHz, ORL < −36 dB | printed | GF-BIAN; Europractice |
| Modulators | ring and MZI modulators > 35 GHz BW; 56 Gbaud class; 112 Gbaud in development | printed | GF-BIAN |
| Heaters | "thermal phase shifters standard & undercut"; ring filters with "sealed undercut (airgap) heaters"; PDK MRM heater = four silicide resistors with Si undercut | printed | GF-IMAPS; GF-BIAN; GF-ECIO |
| High-value resistor | 1.5 kΩ/sq | printed | Europractice |
| BEOL | 9 metal levels; 7 Cu global levels with 1×, 2×, 8× thick wires; low-k and TEOS/FTEOS ILDs; planarized passivation | printed | Europractice; CMC |
| BEOL total thickness | **not public** | — | — |
| Edge couplers | V-groove; BEOL replaced by thick TEOS cladding over the SiN spot-size converter, with undercut | printed | GF-BIAN |
| Substrate resistivity | 9–18 Ω·cm | printed | Europractice |
| Voltages | core 0.9 / 1.0 V; IO 1.5 / 1.8 V; gate oxide 1.12 / 2.5 nm | printed | Europractice |
| Junction temperature (CMOS) | −40 to 125 °C | printed | Europractice |
| PDK model features | hardware-verified compact models; 105 °C; self-heating; optical power; "EO thermal time constant is modeled in all thermal phase shifters"; thermal coupling; "optical power self-heating in MRM"; bidirectional | printed | GF-BIAN slide 39 (no values given) |
| MPW | 5 × 5 mm block; 5–6 months; 50 dies | printed | Europractice |

**Contrast: GF 45RFSOI** ("zero-change", used by Berkeley/MIT/Ayar Labs before 45CLO): sub-100 nm Si body,
BOX < 200 nm (substrate removal needed for optics), no Ge epi (Kramnik et al. 2024; BU-45CLO intro).

### 4.3 Ring modulators published on this platform family

| Device | Platform | Radius | Junction | Q | FSR | Shift | BW | Heater | Source |
|---|---|---|---|---|---|---|---|---|---|
| Lightmatter G25 MRM | 45SPCLO | not printed (D ~15 µm on later slides) | depletion PN | 3500 (talk, not re-checked) | design 1.8 THz | 50 pm/V (talk) | 30.9 ± 2 GHz at −1 V | 0.44 nm/mW, in-ring resistive | LM-P; LM26-S |
| GF PDK MRM | GF 45 nm SiPh (45SPCLO) | 7.5 µm | not stated | not stated | not stated | not stated | 56 Gbaud-optimized | 4 silicide resistors, undercut | GF-ECIO |
| BU rib ring | 45CLO | 8 µm, w = 410 nm | vertical PN (counter-doped) | 5000 loaded | 8.6 nm (1.5 THz) | 3.1 GHz/V (≈ 17 pm/V) | 20 GHz at 0 V | — | BU-45CLO |
| BU spoked disk | 45CLO | 4.36 µm outer / 3.05 µm inner | interdigitated lateral | 4210 | 18 nm (3.2 THz) | 2.8 GHz/V | 17 GHz | — | BU-45CLO |
| MOSCAP ring | "45 nm SOI CMOS" | 1.5 µm | MOS capacitor | — | 8.5 THz | 30 GHz/V | — | — | Gevorgyan et al., OFC 2021 Th5A.3 (title and abstract only) |
| Kramnik pair source | 45RFSOI | ≈ 19–20 µm, 2.8 µm wide | in-cavity p-i-n (sensing) | Q_i ~1e5 | ≈ 901 GHz | — | — | center resistive heater; 0.62 nm over 5.1 mW (≈ 0.12 nm/mW, inferred) | arXiv:2411.05921 |
| Ayar Labs 8λ × 32 G macro | "45 nm SOI" (exact process not confirmed) | — | — | — | — | — | — | integrated thermal control; error-free over a 50–110 °C ramp | Bhargava et al., VLSI 2024 (abstract only) |

No public source gives a measured heater thermal time constant on 45SPCLO. GF models it in the PDK but
publishes no value.

### 4.4 Derived cross-checks for the plant model

- **Strip waveguide.** A 400 × 160 nm strip at 1310 nm gives n_eff ≈ 2.36 and n_g ≈ 4.04 (derived, crude
  EIM, straight guide). At 400 × 200 nm it gives 2.53 and 4.16. The only measured ring on the family (BU rib)
  implies n_g ≈ 3.91 (inferred). The textbook's assumed n_g = 4.2 is at the top of this range.
- **Radius for 1.8 THz FSR.** R = c/(2π n_g FSR) gives 6.80 µm (n_g = 3.9), 6.56 µm (4.04) and 6.31 µm (4.2).
  [verified]
- **FSR of the GF PDK size.** A ring with R = 7.5 µm has FSR ≈ 9.3 nm (1.63 THz) at n_g = 3.9 and 9.0 nm at
  4.04. A 15 µm diameter ring and a 1.8 THz FSR are therefore consistent only if "diameter ~15 µm" is an outer
  dimension, or if n_g < 3.6 (inferred). [corrected: the exact threshold at R = 7.5 µm is n_g < 3.53. If 15 µm is
  the outer edge of a 400 nm waveguide (centreline R = 7.3 µm), 1.8 THz still needs n_g ≈ 3.63. With the strip
  n_g ≈ 4.07, 1.8 THz needs R ≈ 6.51 µm, so "~15 µm" fits only if it measures the device ~1 µm beyond the
  waveguide centreline (for example the doping, contacts or heater), or if it is a loose round number]
- **Thermal resistance.** 0.405 nm/mW ÷ 85.9 pm/K = 4.72 K/mW. With the printed 0.44 nm/mW it is
  5.12 K/mW (inferred). [verified arithmetic; 0.405 = 10.30 nm / 25.45 mW from the textbook's talk values]
- **Laser sweep in temperature units.** At 85.9 pm/K, 50 nm/s ≡ 582 °C/s and 4 nm ≡ 46.6 K. At the bench's
  own 62.5 pm/K the same numbers give 800 °C/s and 64 K (inferred). [verified arithmetic. Checker's note:
  62.5 pm/K is close to the derived strip-ring estimate (≈ 61–64 pm/K at 300 K, see §2.5), so the two
  Lightmatter coefficients (86 and 62.5 pm/K) cannot both be the ring's own dλ/dT]

---

## 5. REF check, entry by entry

Verdicts: **confirmed** (the corpus agrees), **confirmed, refine** (agrees, with a nuance to add), **conflict**
(the corpus disagrees), **not re-checked** (the source was not reachable), **no source** (still assumed).

| REF key | REF value | Verdict | Evidence |
|---|---|---|---|
| lambdaNm | 1310 | confirmed | LM26-P "centered at 1310 nm"; comb 1301.5–1318.9 nm read-off (HC slide 17) |
| neff | 2.5 | no source | EIM: 2.36 (400 × 160) / 2.53 (400 × 200), derived |
| ng | 4.2 | no source; leans high | 45CLO rib ring 3.91 (inferred); EIM 4.04 for 160 nm SOI |
| confinement | 0.85 | no source; likely high for 160 nm SOI | thinner Si lowers confinement (qualitative; compute in the mode-solver study) |
| radiusUm | 6.3 | derived; consistent | corpus diameters 5–15 µm (SIB), ~10 µm (PECC), ~15 µm (LM26-S, HC); GF PDK R = 7.5 µm |
| waveguideLmNm | [400, 200] | **conflict (D1)** | GF 45SPCLO photonics SOI = 160 nm |
| fsrTHz | 1.8 | confirmed | LM-P §2 |
| fsrMeasuredBoundNm | 14.1 | **conflict (D2)** | traces are windowed to ~1 nm; no FSR information [verified on the 1000 × 700 px embedded image] |
| fsrRequiredG26THz | 2.8 | confirmed arithmetic; **refine (D10)** | an FSR below the span can work if aliases fall off-grid; PAT-4 FSR < 11.44 nm |
| compositeGridGHz / perDirectionGridGHz / nChannels / nChannelsPerFiber | 200 / 400 / 8 / 16 | confirmed | LM26-P §2, Fig. 1 |
| l20LambdasNm | [1311, 1331] | confirmed | BIDI; HOTI26 |
| Q, qFromLinewidth | 3500, [3900, 4200] | not re-checked | talk not reached |
| erVsHeaterV | table | confirmed (±1 dB) | read-off: 27.9, 29.8, 28.8 (0.6 V), 24.4, 22.1, 19.8, 18.0, 16.9 dB |
| modEffPmPerV, modEffMeasuredAvgPmPerV | 50, 52.8 | not re-checked | talk |
| swingVpp | 1.3 (simulated) | confirmed | LM-P §2 |
| swingMeasuredVpp | 0.9 "on the bench" | **conflict (D3)** | 0.9 Vpp is the PPG swing into a 50 Ω-terminated CMOS driver; the MRM sees the driver's rail-to-rail output [verified against LM-P §2 and §3.2 text] |
| biasRangeV | [1.0, 3.75] | confirmed | LM26-S slide 9 axes |
| eoBwGhz ± tol | 30.9 ± 2 | confirmed | LM-P §3.1 |
| eoBwAt2VGhz | 35.4 | confirmed (legend); caption 34.1 | LM-P Fig. 1(f) |
| eoBwG26Ghz | 66.8 | **conflict in tagging (D4)** | the slide cites Tu3J.6; −2 V, "detuning 4 dB", ~2 dB peaking |
| modIlDb | 4.9 | confirmed | LM-P §3.2 |
| dLambdaDT_pmK | 86 | confirmed (inferred from ~15 GHz/°C → 85.9) | LM26-P §1 |
| dLambdaDT_benchPmK | 62.5 | confirmed | SIB (50 nm/s, 800 °C/s) |
| dLambdaDT_ocpPmK, derived | 80, 50 | not Lightmatter; not checked | — |
| heaterPFsrMw, heaterVFsr, heaterOhm, heaterImA (+ ranges) | 25.45, 1.195, 55.9, 21.4 | not re-checked | talk wafer maps not reached |
| heaterNmPerMw | 0.44 | confirmed | LM-P §3.1 |
| heaterTuningRangeNm | 13.0 | confirmed | read-off 1301.89 → 1314.89 |
| HEATER_V_LAMBDA (16 points) | table | confirmed (±0.05 nm) | read-off; see 2.2 |
| heaterDacBits | 16 | confirmed as a lower bound (inferred) | codes up to ~49.3 k |
| heaterDacCodeRange | [19500, 49300] | confirmed | read-off 19.2 k – 49.3 k |
| heaterDacLaneSpread | 8000 | confirmed | read-off 7.5–9.5 k |
| heaterDacSlewCodesPerMs | 460 | **refine (D11)** | per lane: mean ≈ 340, peak ≈ 700; 460 = all-lane envelope ÷ 65 ms |
| RthKPerMw | 4.7 | arithmetic confirmed (4.72) | with printed 0.44 nm/mW: 5.12 |
| tauThUs, tauSlowUs, slowShare | 10, 300, 0.25 | no source | PAT-2 premise τ_th > 2.56 µs; GF models τ but publishes none |
| crosstalkNearest/Next | 0.1 / 0.03 | no source | PAT-1 description treats crosstalk as negligible (paraphrase) [verified verbatim: ¶[0048] "thermal crosstalk is negligible" (design assumption)] |
| dieGradientK, picPeakC, coldPlateC | 63, 100, 25 | not re-checked | ECTC paper not reached |
| photonicSupplyStabilityC | 1 | not Lightmatter; not checked | — |
| serdesToMrmUm | 100 | confirmed | ARXIV §III-A |
| transientCPerS | 2000 | confirmed (headline "> 2000") | read-off peak ±2200, mean 1350 |
| transientPeerReviewedCPerS | 808 | confirmed; **refine (D7)** | 808 × 0.1 s = 80.8 °C → probably ΔT/period, not the instantaneous slope [verified arithmetic; hedge added in D7] |
| transientOpticalEmulatedCPerS | 800 | confirmed for the laser sweep; **conflict in labelling (D6)** | SMTAI/PECC use "800 °C/s … on-die heater aggressor" |
| soakHours, soakCycles | 30.5, [85, 95] | confirmed | read-off ≈ 30.4 h, ≈ 90 cycles |
| slowRampCPerS | [0.12, 0.14] | confirmed | read-off 0.124 °C/s |
| fastCycleRampCPerS | [1.3, 1.4] | not re-checked | source not identified |
| xpuPowerStepCPerS | [10, 120] | not Lightmatter; not checked | — |
| lmJunctionC | [0, 105] | confirmed | LM26-S slide 9; HC slide 16 |
| demonstratedCycleC vs tecSweepC | [25, 105] vs [35, 105] | **conflict (D5)** | same experiment; trace 33–108 °C |
| monitorCurrentUa | 15 | confirmed | read-off 15.5 ± 0.5 µA |
| afeCornerCurrentUa | 80 (simulated) | confirmed | LM-P §2 |
| dataPathCurrentUa | 637 | not re-checked | — |
| monitorAdcBits | 16 | **refine** | inferred only from RX-path polarization-monitor codes (≤ 54 k); MRM-monitor ADC width not published |
| monitorPolSwingDb (codes 7000 → 41500) | 7.7 | **conflict (D8)** | paper: current swings max → zero, so ~7 k is an offset; also IN0 gives 5.6 dB |
| responsivity (0.9, 0.88–1.04) | — | not re-checked | talk; GF angled Ge PD > 0.95 A/W (GF-BIAN) |
| pdBwGhz | [46.4, 56] | 56 confirmed (body); 46.4 not found in the submission | LM-P |
| afeBwGhz, afeNoisePaRtHz | [16, 18], 6.4 | 16 and 6.4 confirmed (simulated) | LM-P §2 |
| inputPowerDbm | 4 | confirmed | LM-P §3.2 |
| relOmaRefDbm | 0 (1 mW) | confirmed | HC slide 16 note |
| ringPowerOcpMidDbm | 9 | not Lightmatter | G26 budget: laser 10 dBm/λ into a 12 dB TX path (QDSOA) |
| laserPerLambdaDbmM1000 | 14 | **not found** | HC "16λ, 100 mW/fiber" → 8.0 dBm/λ (11.0 for 8 λ); QDSOA 10 dBm/λ; OSA peaks +10…+12 dBm |
| rxSensG25DbmOma | {−11, −9} | not re-checked | talk |
| rxSensG26DbmOma | best −14, worst −12.8, floor −12.2 | **refine (D12)** | read-off at 1e-9: best ≈ −14.7, worst ≈ −12.5; error-free floor reached at −13.4 … −11.3 |
| relOmaStabilizedDb | [−3.0, −2.1] | confirmed | contours −2.1 … −3.0 |
| relOmaStabilizedWorstBiasV | 3.1 | confirmed (≈ 3.15 at 0 °C; the contour rises to ≈ 3.4 at 105 °C) | read-off |
| relOmaUnstabilizedPeakC | 70 | confirmed (≈ 69.6) | read-off |
| relOma8dbHalfWidthC | 2.85 | confirmed (≈ 2.8) | read-off |
| omaPenaltyDbPerC | 0.67 | inferred; fine. Map asymmetric: −24 dB at −12.7 / +9.3 °C from peak | read-off |
| eyeMv, eyeUi | 4.7, 0.3 | consistent (≈ 5 mV, ≈ 0.3 UI at 1e-10) | LM26-P Fig. 2(c, d), read-off |
| laserAccuracyGhz {spec, demo, roadmap} | {20, 15, 3} | confirmed | LM26-P §4; YOLE; HOTI26 |
| laserLinewidthMhz.msaReq, laserSmsrDb.msaReq | 1.0, 30 | confirmed | WPW |
| blueBiasMeasuredNm | 7.6 | consistent | 0 V resonance 1301.89 nm (read-off) |
| bumpPitchUm | 120 | confirmed | HC slide 14 |

---

## 6. Discrepancies with the textbook

Ranked by how much each one changes a plant model or a controller.

- **D2. `fsrMeasuredBoundNm = 14.1` is unsupported.**
  - In the high-resolution embedded image of the G25 heater-sweep spectra (LM26-S slide 8; the same data as
    LM-P Fig. 1(e)), each bias trace spans only about 1–1.5 nm around its own dip. Each trace begins abruptly
    at about −0.5 to −1 dB, roughly 0.5 nm to the left of its own dip, and ends shortly after recovering to
    about 0 dB. The 0 dB level itself lies under the black axis frame, so traces cannot be followed there.
  - So the figure cannot show "one resonance across a 15 nm window". The **measured FSR is unpublished**, and
    only the 1.8 THz design value exists. [verified: in the embedded image each trace starts at about −0.5 dB
    and spans ≈ 1–1.5 nm; e.g. the 1.5 V trace runs from ≈ 1314.4 to ≈ 1316 nm]
  - Model consequence: keep the FSR as a design parameter. Do not bound it at > 14.1 nm. Note that a bound of
    > 14.1 nm would also have needed a ring of D < ~10 µm, which conflicts with "diameter ~15 µm".

- **D3. `swingMeasuredVpp = 0.9` is not the MRM swing.**
  - LM-P §3.2 says a pulse-pattern generator delivered 0.9 Vpp through ~6 dB of electrical loss into the chip.
  - LM-P §2 says the monolithic driver has 50 Ω input termination and a CMOS rail-to-rail output equal to its
    supply (1.3 V simulated).
  - So the junction swing during the bench test is not printed. By design it is about the supply voltage,
    which was also not printed for the bench. [verified: §3.2 "A pulse pattern generator (PPG) provided … PRBS
    patterns with output swing of 0.9 Vpp … electrical insertion loss between the PPG and the chip at the
    Nyquist frequency was about -6 dB"; §2 "CMOS rail-to-rail output swing equal to the supply voltage", "simulated
    output voltage swing is 1.3 Vpp … supply voltage of 1.3V". About 0.45 Vpp therefore reached the driver input
    at Nyquist (inferred)] Several chapters (Ch00, Ch06, Ch09, Ch17, Ch19) treat 0.9 Vpp
    as the swing applied to the modulator.

- **D1. Waveguide thickness 200 nm vs 160 nm.**
  - The SMTAI render (Oct 2025) labels a Lightmatter waveguide 400 × 200 nm.
  - GF's own 45SPCLO overview prints a 160 nm photonics SOI layer, and G25 is printed as 45SPCLO.
  - Either the render is schematic, or the product it depicts is not on 45SPCLO. In the model, treat
    thickness as uncertain in {160, 200} nm and default to 160 nm for G25/G26. This changes neff, ng and
    confinement, and so dλ/dT and the FSR-to-radius mapping.

- **D5. Two "envelopes" are one experiment.**
  - `demonstratedCycleC` [25, 105] and `tecSweepC` [35, 105] come from the same 30.5 h, 8-lane run. The
    lane-by-lane BER traces are identical in LM26-P Fig. 3(a), LM26-S slide 10, SMTAI slide 19 and PECC
    slide 9. The labels differ: 35→105 in the paper; 25→105→25 in SMTAI, PECC, SIB and HOTI25.
  - The plotted temperature spans about 33–108 °C (read-off).
  - Use one experiment with the label disagreement printed, not two envelopes.
  - [verified: the slide-10 temperature axis (30–110 °C) shows troughs at ≈ 33 °C and peaks at ≈ 107.5 °C, and
    the lane BER traces (two lowest lanes crossing mid-run, end at ≈ 30.4 h) look the same as in LM26-P
    Fig. 3(a); paper caption "between 35oC and 105oC using a TEC over 31 hours"]

- **D6. "800 °C/s" is attached to two methods.**
  - `transientOpticalEmulatedCPerS` says the 800 °C/s figure is optical and that the die is never heated.
    That matches SIB (Sep 2025: 50 nm/s laser sweep over 4 nm).
  - SMTAI and PECC (Oct 2025) print "800 °C/s … using an on-die heater aggressor" over the on-die plot, which
    is the same plot as the paper's 808 °C/s Fig. 3(b).
  - So "800 °C/s" alone does not identify the method. Keep both rows and cite the method with the number.

- **D4. The 66.8 GHz EO bandwidth is tagged G26, but its slide cites Tu3J.6.**
  - LM26-S slide 8 carries the footer "R. Baghdadi et al., OFC 2025. Paper Tu3J.6".
  - The measurement conditions are reverse bias −2 V, "detuning 4 dB", RF 0 dBm, "Reticle 0, device 25".
  - The S21 curve peaks by about +2 dB near 35–40 GHz (read-off), so 66.8 GHz is a peaked, detuning-dependent
    response.
  - The Hot Chips copy of the same slide (Aug 2025) showed 31.8 / 35.4 GHz instead. Treat its generation as
    unknown, and report it as a single device at a specific detuning.

- **D8. The 7.7 dB "polarization swing" ignores the ADC offset.**
  - LM26-P §3.1 says the photocurrent goes between the maximum and **zero**. The lowest code, about 7–7.5 k
    on IN7, is therefore an offset or dark level, not a light level. The optical swing is much larger than
    7.7 dB (in principle unbounded, limited by the polarization splitter).
  - The other channel (IN0, ≈ 15 k – 54 k) would give 5.6 dB by the same arithmetic.
  - The textbook does correctly say that this is a receive-path monitor, not the TX lock sensor (Ch18).

- **D7. The 808 °C/s definition.**
  - Printed: 25 → > 105 °C, "ramped up and down" with a 100 ms period. A symmetric up-and-down ramp of
    ≥ 80 °C twice per 100 ms needs ≥ 1600 °C/s mean slope.
  - 808 °C/s × 100 ms = 80.8 °C. So 808 is most likely ΔT/period, and the instantaneous slope was ≥ 1600 °C/s.
    [verified with a hedge: 808 × 0.1 = 80.8 holds. The ≥ 1600 °C/s conclusion needs the up and down ramps to
    take about half the period each. If the waveform were a one-sided sawtooth (one ramp spanning almost the
    whole 100 ms, then a fast return), 808 °C/s would be the true ramp slope. The words "ramped up and down"
    and the symmetric slide-10 waveform favour the report's reading; the Fig. 3(b) data are under-sampled and
    cannot decide it]
  - The Fig. 3(b) inset reaches about 113 °C, like the slide-10 aggressor, and looks under-sampled.
  - It is plausible, but unconfirmed, that the 808 and > 2000 °C/s results are the same on-die heater
    described with two rate definitions (mean vs peak). The slide-10 waveform is parabolic in T
    (dT/dt is a linear sawtooth), with mean ≈ 1350 and peak ≈ 2200 °C/s. Model aggressors by waveform, not by
    one rate number. [corrected: the waveform is not parabolic. Integrating a linear 0 → 2200 °C/s ramp over
    65 ms gives only 72 °C, not the plotted 88 °C. The re-digitized red trace is concave (≈ 27 %, 45 %, 66 %,
    78 %, 92 % of peak at 15 %, 31 %, 46 %, 62 %, 77 % of the rise), close to a quarter-cosine. T fits
    T_max − ΔT·|sin(πt/P)| to 0.34 °C rms (parabola: 1.6 °C rms), so peak/mean = π/2. If the 808 °C/s test used
    the same shape with ΔT = 80 °C and P = 100 ms, its mean would be 1600 °C/s and its peak ≈ 2510 °C/s (derived)]

- **D10. "The free spectral range check fails" (correction 9) is too strong as stated.**
  - The 2.8 THz requirement assumes no alias resonance may fall inside the TX comb. With FSR = 1.8 THz
    = 4.5 × 400 GHz, a ring's ±1 FSR aliases land exactly half-way between TX channels, 200 GHz from the
    nearest one. The next alias (3.6 THz) is outside the 2.8 THz span.
  - So the comb is still unambiguous. The cost is an off-resonance penalty on two neighbours at 200 GHz
    detuning, about the same as a 200 GHz-grid neighbour. [verified by enumeration over the 8-channel,
    400 GHz grid: rings on channels 0–2 and 5–7 each put one ±1-FSR alias inside the TX span, always 200 GHz
    from the nearest TX channel; rings on channels 3–4 put none; ±2 FSR (3.6 THz) is outside the 2.8 THz span]
  - Lightmatter's own patent (PAT-4) claims modulator FSR < 11.44 nm (< 2 THz) for this kind of interleaved
    architecture.
  - The alias positions coincide with the counter-propagating channels, so backscatter and interleaver leakage
    land on them (derived). The textbook's arithmetic (2.8 > 1.8) is right; the conclusion should be "aliases
    must be managed", not "fails".

- **D11. `heaterDacSlewCodesPerMs = 460` is an envelope figure.**
  - Per lane, the code swings about 20–22 k in 65 ms: mean ≈ 310–340 codes/ms.
  - The peak slew right after the temperature cusp is ≈ 700 codes/ms (read-off, ±15 %).
  - The actuator must support the peak value.

- **D12. G26 RX sensitivity.** The read-off at BER 1e-9 gives best ≈ −14.7 and worst ≈ −12.5 dBm OMA,
  against REF's −14 / −12.8 (±0.3 dB read-off uncertainty). The difference is small but systematic on the
  best lane.

- **D13. The aggressor minimum temperature.** The slide-10 black trace bottoms at about 25 °C (read-off
  25.3), not about 30 °C as in the REF comment. Peak ≈ 113 °C (agrees). dT/dt is ≈ −2200 → +2200 (REF
  comment: −2300 → +2200).

- **D14. G25 transmit topology is ambiguous.**
  - REF and Ch06 say the G25 transmit ring was all-pass with no monitor, based on the LM-P Fig. 1(b) symbol.
  - The two later slides that present the G25 results (HC slide 15; LM26-S slide 8, with the Tu3J.6 footer)
    show a micrograph labelled Input / Thru / Drop next to the MRM heater-sweep spectra.
  - It may be the receive filter ring rather than the MRM. The claim should be hedged: "drawn as all-pass in
    the paper schematic".

- **D15. G26 energy and line rate.**
  - The paper gives 1.7 pJ/bit for the SiPh die (3.8 total). The deck, Yole and SIB give 2.6 pJ/bit
    "photonics" (4.6 wall-plug in SIB). arXiv gives 2.3 pJ/bit for PIC + laser.
  - The line rate is 50 Gb/s NRZ (paper) against 53.125 Gbaud NRZ (WPW).
  - The textbook generation map gives "50 Gb/s NRZ"; add the 53.125 Gbaud figure as a second source.
  - [corrected: these are most likely scope differences, not conflicts. SIB's full sentence is "wall-plug
    efficiency of 4.6 pJ/bit, inclusive of the 2.6 pJ/bit for the photonics and external light source (Guide),
    and ~2 pJ/bit for an off-the-shelf XSR SerDes" (printed). The paper's 3 W / 3.8 pJ/bit covers "the SerDes and
    SiPh chip" with the lasers external. So 2.6 − 1.7 ≈ 0.9 pJ/bit and 4.6 − 3.8 ≈ 0.8 pJ/bit are both the
    laser share (inferred). For the rate, 53.125 = 50 × 17/16, the standard line rate of a nominal 50 Gb/s
    lane (inferred); WPW calls the same demo "800 Gbps on a single fiber". The paper's measured PRBS9 rates were
    50 Gb/s (Fig. 3) and 56 Gb/s (Fig. 2). For the EO plant, use the symbol rate actually driven in the test you
    model]

- **D16. `laserPerLambdaDbmM1000 = 14` was not found.** Every public figure is lower: 100 mW/fiber for 16 λ
  (→ 8 dBm/λ; HC, SMTAI); 10 dBm/λ in the G26-hardware budget (QDSOA); OSA peaks of +10…+12 dBm (HC
  slide 17).

- **D17. `monitorAdcBits = 16` rests on the RX polarization-monitor codes.** The MRM drop-port monitor is
  plotted only in µA, and its ADC width is not published.

- **D18. Smaller notes.**
  - The paper's text says Fig. 1(e) is "vs. applied bias voltage on the modulation junction", while its caption
    says "heater voltages". The 13 nm shift settles it in favour of the heater (inferred). The textbook's
    reading is right, but the source is internally inconsistent.
  - The fast-test BER text says "10^-11", while the plotted line sits near 8e-13 (read-off).

---

## 7. Open questions (for Lightmatter or for further search)

1. The measured FSR of the G25/G26 MRM, the ring radius at the waveguide centre, and whether "diameter
   ~15 µm" is an outer dimension.
2. Whether the G26 MRM is the GF PDK 7.5 µm add-drop ring (≈ 5 % drop, silicide undercut heater) or a custom
   design. This sets the heater τ, efficiency and drop-port monitor level.
3. The heater power law versus DAC code (voltage DAC, current DAC or PWM; PAT-2 suggests DAC + PWM), and the
   DAC full-scale.
4. Lock set-point: peak, fraction of peak (as in GF's 3/4 or edge lock) or dither-derived. The drop current
   stays at 15 µA, but the peak current is unknown.
5. What temperature the 15 GHz/°C refers to: ring, on-die sensor or TEC. A reference sensor that sees only
   ~0.6 of the ring's ΔT would explain the 86 vs ~50–54 pm/K gap (hypothesis, derived). [corrected: the
   checker's derived strip-ring value is ≈ 61 pm/K (dn_Si/dT = 1.86e-4 /K) to ≈ 64 pm/K (1.94e-4 /K, the
   1310 nm value in 02_materials.md), not 50–54 pm/K. The ratio in the hypothesis would then be ≈ 0.71–0.75,
   not ~0.6. The SIB bench equivalence (62.5 pm/K) sits inside this derived range]
6. The rate definition behind 808 °C/s, and whether it is the same heater as the > 2000 °C/s slide.
7. The content of PAT-7 (Guide wavelength-control patents) and the coupling between laser and ring loops.
8. Ring-to-ring thermal crosstalk on an 8-ring bus, and the heater thermal time constant(s). No public number
   exists anywhere in this corpus.
9. Whether L20 MRMs (two colours 20 nm apart, each port programmable to either λ; BIDI) are on 45SPCLO.

---

## 8. Read-off methods (so the numbers can be redone)

- **PDFs.** Rendered with poppler `pdftoppm` at 300–600 dpi. Embedded raster figures were pulled with
  `pdfimages` at native resolution. LM26-S slide 8 spectra are 1000 × 700 px.
- **Axis calibration.** Pixel positions of gridlines and ticks. Traces were found by matplotlib "tab10"
  colours (distance < 40–45 in RGB), with the deepest pixel per colour cluster taken as the dip.
- **Slide-10 heater DAC.** The lowest saturated-colour pixel per column (the bottom lane), smoothed over 9 px
  (≈ 3.5 ms). The slope is from finite differences. Smoothing lowers the peak slope, so ≈ 700 codes/ms is a
  lower estimate.
- **Slide-10 30 h cycling.** Peaks were counted on a fixed row near 107 °C; the legend hides x > 1580 px. The
  rest was extrapolated with the measured 17.55 px period.
- **Uncertainties** are quoted beside each value. Read-offs are not printed numbers. Before any number here is
  used as fact, it should be checked against Lightmatter's underlying data.

## 9. Sources

Lightmatter:
- OFC 2025 Tu3J.6 submission: https://lightmatter.co/wp-content/uploads/2025/04/OFC-2025-submission.pdf
- OFC 2026 M4B.3 paper (IEEE Xplore, local copy): `/Users/kenny/Downloads/16-Wavelength_800-Gbps_Bidirectional_Link_over_Single-Mode_Fiber_Using_Microring_Transceivers (1).pdf`
- OFC 2026 deck: https://lightmatter.co/wp-content/uploads/2026/04/OFC-2026-16-lambda-Bidi_PAM4-Nikhil-Kumar.pdf
- Yole briefing: https://lightmatter.com/wp-content/uploads/2026/04/OFC-2026-Yole-Nikhil-Kumar.pdf
- QD-SOA OFC 2026 M1B.1: https://lightmatter.com/wp-content/uploads/2026/04/OFC-2026-presentation-9dB-link-margin-gain-using-a-Quantum-Dot-SOA-in-an-8-wavelength-DWDM-50Gbps-NRZ-CPO-link-Alex-Sludds.pdf
- Hot Chips 2025: https://lightmatter.com/wp-content/uploads/2025/08/57-lightmatter_hotchips_2025_final.pdf
- SMTAI 2025: https://lightmatter.com/wp-content/uploads/2025/10/2025-SMTAI-Womens-Leadership-Session-DFX.pdf
- PECC 2025: https://lightmatter.com/wp-content/uploads/2025/10/PECC-Darius-Bunandar.pdf
- HotI 2025 deck: https://lightmatter.com/wp-content/uploads/2025/08/Nick-Harris-Lightmatter-HOTI-Deck-2025.pdf
- ISES Korea 2025: https://lightmatter.co/wp-content/uploads/2025/08/ISES-Korea-08-2025_Ritesh-Jain.pdf
- HotI 2026 deck: https://lightmatter.co/wp-content/uploads/2026/08/Nick-Harris_Lightmatter_The-Interconnect-Foundation-for-High-Performance-AI-Training-and-Inference_HotI2026.pdf
- Seeing is Believing (5 Sep 2025): https://lightmatter.co/blog/seeing-is-believing-a-technical-deep-dive-into-lightmatters-hardware/ (Fig. 1: https://lightmatter.com/wp-content/uploads/2025/09/Copy-of-Temperature-ramp-plots.png)
- Wide and Parallel Wins (15 Mar 2026): https://lightmatter.co/blog/wide-and-parallel-wins/
- BiDi technical brief (Sep 2026): https://lightmatter.com/wp-content/uploads/2026/09/BiDi-ScaleUp-Tech-Brief_Sept-26.pdf
- arXiv:2609.01821: https://arxiv.org/abs/2609.01821
- Resources index: https://lightmatter.co/resources/

Patents (Google Patents pages):
- https://patents.google.com/patent/US20240405908A1/en
- https://patents.google.com/patent/US11409045B2/en (and US11686902B2, WO2021003221A1)
- https://patents.google.com/patent/US20230308188A1/en (granted as US12470304B2)
- https://patents.google.com/patent/US12355492B2/en
- https://patents.google.com/patent/WO2025122797A1/en
- US12147078B2 (USPTO image PDF, front page read)
- US20260095014A1, US20260095015A1, WO2026072846A1 (titles from a Google Patents assignee query; text not reached)
- Non-Lightmatter context: https://patents.google.com/patent/US20240405882A1 (Xilinx)

Platform:
- GF IMAPS 2022: https://imapsource.org/api/v1/articles/91148-establishing-a-packaging-ecosystem-for-globalfoundries-leadership-silicon-photonics-wafers-2022-view.pdf
- Y. Bian, GF, Optica webinar Oct 2025: https://opticaorg-dev-cac7d2csctagc8bm.z01.azurefd.net/$web/optica/media/files/getinvolved/tech-groups/pdfs/2025/optica_webinar_globalfoundries_yushengbian_oct2025.pdf
- GF ECIO 2023 (Nandi et al.): https://www.ecio-conference.org/wp-content/uploads/2023/05/MRM_thermal_stabilization_ECIO23.pdf
- Al Qubaisi et al., CLEO 2021 (45CLO rings): https://par.nsf.gov/servlets/purl/10284518
- Europractice GF 45SPCLO: https://europractice-ic.com/technologies/photonics/globalfoundries/
- CMC GF Fotonix 45SPCLO: https://www.cmc.ca/globalfoundries-fotonix-45spclo/
- Rakowski et al., OFC 2020 T3H.3 (abstract only): https://opg.optica.org/abstract.cfm?uri=ofc-2020-T3H.3
- Kramnik et al., arXiv:2411.05921 (45RFSOI): https://arxiv.org/abs/2411.05921
- Converge Digest, GF to produce Passage (19 Nov 2024): https://convergedigest.com/lightmatter-picks-globalfoundries-to-produce-its-ai-interconnect/
- Bhargava et al., VLSI 2024 (abstract only): https://ieeexplore.ieee.org/iel8/10631290/10631310/10631547.pdf

---

## Verification log

Independent check, 2026-10-09, by a second agent that did not write this note. Method: re-read every cited
primary source that could be reached; recomputed every formula; re-digitized the key figures with my own
scripts. Sources used: LM26-P (local IEEE copy in `~/Downloads`); LM-P, LM26-S, GF-IMAPS, GF-ECIO, BU-45CLO,
GF-BIAN, HC, SMTAI, PECC, YOLE, QDSOA, HOTI26 (PDF copies cached in the session scratchpad, text extracted with
`pdftotext`); SIB and WPW (live pages, fetched 2026-10-09); patents US11409045B2, US12470304B2, US12355492B2
and US20240405908A1 (USPTO `image-ppubs` image PDFs, OCR with the macOS Vision framework, because Google
Patents returned HTTP 503 and Justia, FreePatentsOnline and Patentscope returned HTTP 403). Not re-checked:
WO2025122797, US12147078, PAT-7, the Xilinx patent text, arXiv:2609.01821, Europractice/CMC pages, HC slide 17
comb read-off, relOMA map read-offs (slide 9), RX-sensitivity read-offs (D12).

Verdicts: **confirmed** = the source or the math agrees; **corrected** = edited in place above;
**refined** = correct, with a hedge or nuance added in place.

| # | Claim (where) | Verdict | Evidence |
|---|---|---|---|
| 1 | ~15 GHz/°C = 85.9 pm/K (§2.5, LM26-P §1) | confirmed (arithmetic) + note | Printed verbatim in §1 as an aside. 15e9 × λ²/c = 85.86 pm at 1310 nm. Derived cross-check: semi-vectorial FD of a 400 × 160 nm oxide-clad strip gives dλ/dT ≈ 61 pm/K (dn_Si/dT 1.86e-4) to ≈ 64 pm/K (1.94e-4); printed value is ~1.35× higher. Open question 5's "50–54 pm/K" corrected to ≈ 61–64 pm/K. |
| 2 | 808 °C/s aggressor: 25 → > 105 °C, 100 ms period, local heater, drop-port monitor flat (§2.5, §3.1) | confirmed | LM26-P §3.2 text and Fig. 3 caption re-read; Fig. 3(b) viewed at 300 dpi: PD current ≈ 15.5 µA, T 25–113 °C, BER line ≈ 7e-13 on the inset axis. |
| 3 | D7 inference "808 = ΔT/period, true slope ≥ 1600 °C/s" | refined | Arithmetic right (808 × 0.1 = 80.8). Hedge added: a one-sided sawtooth would make 808 the true slope; the "ramped up and down" wording favours the report. |
| 4 | Slide-10 aggressor numbers (T 25 → 113 °C, ±2200 °C/s, P 130.5 ms, mean 1350 °C/s) | confirmed | Re-digitized: T 24.6 → 112.6 °C; peaks at 65.1, 195.3, 325.5, 456.6 ms (mean spacing 130.5 ms); red trace −2130 … +2170 °C/s; integral of red over a rise 84.6 °C vs plotted ΔT 87.4 °C (3 % agreement). |
| 5 | Slide-10 waveform "parabolic in T, dT/dt a linear sawtooth" (§3.3, D7) | **corrected** | A linear 0 → 2200 °C/s ramp over 65 ms gives 72 °C, not 88 °C. Fit T_max − ΔT·\|sin(πt/P)\|: 0.34 °C rms (T_max 113.6, ΔT 88.5, P 130.7 ms); parabola: 1.6 °C rms and an implied 2780 °C/s peak. Rectified sine gives peak πΔT/P ≈ 2130 and mean 2ΔT/P ≈ 1355 °C/s. |
| 6 | Heater DAC codes, lane offsets, swing, slew (§2.4, finding 7, D11) | confirmed | Re-digitized: 19.6 k – 49.2 k overall; bottom lane 22.2 k swing (341 codes/ms mean); offsets 7.4 k (maxima), 9.4–10 k (minima); 580–620 codes/ms averaged 1–6 ms after each cusp. P ∝ code vs P ∝ code² fits: 393 vs 306 codes rms, not decisive (report's statement stands). |
| 7 | Added by checker: hidden "MRM Drop-Port PD Error Signal" panel (slide-10 image object 184) | new (added to §2.5, §3.1) | Signed % error per lane: ≈ −0.3 % heating, ≈ +0.4 % cooling, spikes ≈ +3.0 / −1.9 % at cusps (read-off). Supports the report's set-point-regulator reading. |
| 8 | G25 fabricated on GF 45SPCLO (§2.1, §4.1) | confirmed | LM-P §3.1: "The designs were fabricated in the GF 45SPCLO platform." |
| 9 | 45SPCLO stack: 160 nm photonics SOI, 88 nm CMOS SOI, 2 µm BOX (§2.1, §4.2) | confirmed | GF-IMAPS PDF page 4, verbatim. A web search found no second public source for 160 nm; treat it as single-source. |
| 10 | GF PDK MRM: R 7.5 µm, ~5 % drop, 4 silicide undercut heaters, TIA 5 kΩ / 1 GHz, 11.5–15.1 mA, blue-edge lock, Vref > 0.65 V (§2.4, §3.4) | confirmed | ECIO 2023 text re-read; all values verbatim. |
| 11 | LM-P values: 1.8 THz design FSR, 0.44 nm/mW, ER > 18 dB, 30.9 ± 2 GHz at −1 V, 31.8 / 35.4 / "near 34.1" GHz, 4 dBm in, 4.9 dB IL, 1.3 Vpp simulated at 1.3 V, PPG 0.9 Vpp, −6 dB, 50 Ω input, 84.4 mW → 1.51 pJ/bit, areas, AFE specs | confirmed | All re-read in the LM-P text or by OCR of Fig. 1; 84.4 mW / 56 Gb/s = 1.507 pJ/bit; 1.8 THz = 10.30 nm. D3 conclusion stands (≈ 0.45 Vpp at the driver input at Nyquist, inferred). |
| 12 | D2: G25 heater-sweep traces are windowed, no FSR bound (§6) | confirmed | Viewed the 1000 × 700 px embedded image (LM26-S slide 8): each trace spans ≈ 1–1.5 nm starting near −0.5 dB. Dip positions (≈ 1301.9 → 1314.9 nm) and notch depths agree with §2.2 by eye. Extra observation: the read-off gives Δλ ≈ 8.96 nm at 1.2 V and ≈ 10.3 nm at ≈ 1.30 V, vs the textbook's (talk) V_FSR = 1.195 V; different device or definition, not re-checked. |
| 13 | D4: 66.8 GHz slide carries the Tu3J.6 footer, −2 V, detuning 4 dB, ≈ +2 dB peaking | confirmed | LM26-S slide 8 text and image. |
| 14 | PAT-2 (US11409045B2): 8-bit DAC + 8-bit PWM, 1/256 over 2.56 µs, ≈ 16 bit, τ_th > PWM period, N = 400 × 5 ns = 2 µs, 16 pF (10–20 pF), 1–10 µs, 500 mV, nA, 10 µs / 100 µs / 1 ms, 0.001 °C, 1/50 000, monitor dark when tuned | confirmed | OCR of the USPTO grant (13 pages); all verbatim. Inventors and 2019-07-02 provisional confirmed; grant date 2022-08-09. |
| 15 | PAT-3 (US12470304B2): ramp-and-maximize lock (claims 7–10), dither 1–1000 kHz (claim 4), non-rational dither ratios, homodyne/heterodyne/FIR, Q > 10⁵ for D < 5 µm, 3 µm pitch, TPA, priority 2022-03-28, grant 2025-11-11 | confirmed | OCR of the 85-page grant. |
| 16 | PAT-4 (US12355492B2): modulator FSR < 11.44 nm (claimed), same-set spacing 1.72–2.86 nm, example grid, optional drop port + PD per TX modulator, < 10 % interleaver leakage, dates, inventors | confirmed | OCR of the grant. 11.44 nm = 1.998 THz; 1.72 / 2.86 nm = 300 / 500 GHz. |
| 17 | PAT-1 (US20240405908A1): heaters only add heat; crosstalk treated as negligible ("paraphrase, not re-verified") | confirmed, upgraded to verbatim | ¶[0048] "thermal crosstalk is negligible"; ¶[0050] "A heater may only increase temperature, not decrease it". Added: ¶[0050] suggests biasing MRMs above the maximum allowed chip temperature. |
| 18 | G26 energy 1.7 (paper) vs 2.6 pJ/bit (slides/blog) "not reconciled" (finding 8, §2.7, D15) | **corrected** | SIB: "4.6 pJ/bit, inclusive of the 2.6 pJ/bit for the photonics and external light source (Guide), and ~2 pJ/bit for an off-the-shelf XSR SerDes". 2.6 includes the laser; 1.7 does not. Laser share ≈ 0.8–0.9 pJ/bit (inferred). |
| 19 | G26 line rate 50 Gb/s vs 53.125 Gbaud "not reconciled" (finding 8, §2.3, D15) | **corrected** | 53.125 = 50 × 17/16, the standard line rate of a nominal 50 Gb/s lane (inferred); WPW calls the demo "800 Gbps on a single fiber" with "16-wavelength, 53.125 Gbaud NRZ". Paper measured 50 Gb/s (Fig. 3) and 56 Gb/s (Fig. 2) PRBS9. |
| 20 | Strip neff / ng (EIM): 2.36 / 4.04 (400 × 160), 2.53 / 4.16 (400 × 200) (§2.2, §4.4) | confirmed | Checker's semi-vectorial FD (10 nm grid, Li-1980 Si, Malitson SiO2): 2.360 / 4.07 and 2.531 / 4.18. |
| 21 | BU-45CLO rib ring: R 8 µm, w 410 nm, Q 5000, FSR 8.6 nm (1.5 THz), 3.1 GHz/V ≈ 17 pm/V, n_g 3.91 | confirmed | CLEO 2021 text re-read ("O-band near 1300 nm"). n_g = 3.909 at 1300 nm, 3.97 at 1310 nm or from 1.5 THz; 3.1 GHz/V = 17.5 pm/V. |
| 22 | Radius / FSR geometry: R(1.8 THz) = 6.80 / 6.56 / 6.31 µm; R = 7.5 µm → 9.34 nm (1.63 THz) / 9.01 nm; "consistent only if outer dimension or n_g < 3.6" (§4.4) | refined | Numbers confirmed. Threshold is n_g < 3.53 at R = 7.5 µm; an outer-edge reading (R = 7.3 µm) still needs n_g ≈ 3.63. With n_g ≈ 4.07 the centreline radius is 6.51 µm, so "~15 µm" must include ~1 µm beyond the waveguide centreline or be loose. |
| 23 | D10 alias argument (FSR 1.8 THz = 4.5 × 400 GHz; aliases 200 GHz from TX channels and on RX channels; 2 × FSR outside span) | confirmed | Enumerated all 8 rings; rings 3–4 have no in-span alias. |
| 24 | Derived arithmetic: R_th 4.72 / 5.12 K/mW; 50 nm/s ≡ 582 °C/s and 4 nm ≡ 46.6 K at 85.9 pm/K; 800 °C/s and 64 K at 62.5 pm/K; 3.8 pJ/bit × 800 Gb/s = 3.04 W; 100 mW/16 λ = 7.96 dBm; 90 × 20.3 min = 30.45 h, 75 °C / 609 s = 0.123 °C/s; 7.7 dB and 5.6 dB monitor ratios; Lorentzian max slope at 3/4 | confirmed | Recomputed in Python. |
| 25 | D5: one 30-h experiment, plotted 33–108 °C | confirmed | Slide-10 temperature panel viewed: troughs ≈ 33 °C, peaks ≈ 107.5 °C; BER trace features match LM26-P Fig. 3(a). |
| 26 | D6: SMTAI/PECC label "800°C/s … using an on-die heater aggressor"; SIB "50 nm/s … over a 4 nm range—representing an 800℃/s temperature aggressor" | confirmed | SMTAI PDF page 19, PECC text, SIB live page. |
| 27 | Laser accuracy ±20 GHz needed / ±15 GHz demonstrated / 3 GHz | refined | LM26-P §4 also says ±20 GHz was "achieved in our laser system"; YOLE prints "ACCURACY OF +/- 15 GHZ" as a VLSP performance claim; HOTI26 "3 GHz Frequency Accuracy". |

Net effect on the plant model: the coefficient table (§2) survives almost intact. The changes that matter
are (a) the on-die aggressor generator should be a rectified sine, not a parabola; (b) the two G26 energy
figures and the two line rates are scope differences, not conflicts; (c) PAT-1's negligible-crosstalk
statement and heat-only/bias-above-max-chip-temperature premise are verbatim; (d) the printed ~15 GHz/°C is
about 1.35× above a derived strip-ring dλ/dT, which keeps open question 5 open but with a smaller gap.
