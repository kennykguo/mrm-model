# 02 · Material properties at 1310 nm, 0 to 125 °C (273 to 400 K)

Scope: the materials of an O-band (1260 to 1360 nm, centre 1310 nm) silicon-on-insulator microring
modulator (MRM) with an oxide cladding, a resistive heater, a depletion PN junction and a Ge monitor
photodiode. For each property this file gives a model with coefficients, its validity range, an
uncertainty and the source. The machine-readable twin is `02_materials.json` (same numbers).

**Provenance words** (same vocabulary as the textbook, `docs/SOURCES_AND_GENERATIONS.md`):
**printed** = verbatim in the source; **read-off** = digitized from a figure or a tabulated dataset;
**inferred** = arithmetic on printed values; **derived** = computed by us (a model or a fit);
**assumed** = no source, or a handbook value we did not re-open in this session.

**Access notes, stated once.** These were read in full: Li 1980 (NIST reprint), Frey et al. 2006
(arXiv), Leviton and Frey (arXiv 0805.0091), Bristow et al. 2007, Borghi et al. 2021 (arXiv), the
Corning 7980 data sheet, the Lightmatter OFC 2026 M4B.3 paper, and the refractiveindex.info database
files (Li, Franta, Wang, Green, Malitson, Luke, Rodríguez-de Marcos). These were seen only through
abstracts, publisher pages or another tool's transcription: Nedeljkovic et al. 2011 (coefficients
taken from the Tidy3D source code), Masetti 1983 and Arora 1982 (taken from TCAD and nextnano
documentation), Cocorullo 1999, Lin 2007, Dinu 2003, Sinclair 2019, Arbabi and Goddard 2013, Lee and
Cahill 1997, Ju and Goodson 1999, Asheghi 1998 and 2002, Misiakos and Tsamakis 1993, Couderc 2014,
Colace 2008 and Koester 2006. These we could not open at all: Della Corte et al. 2000 (JAP 88, 7115),
Komma et al. 2012 and Turner-Foster et al. 2010.

---

## 0. Headline findings

1. **The textbook's `dnSi_dT = 1.86e-4 /K` is a 1523 nm, 300 K value** (Cocorullo et al. 1999). At
   1310 nm silicon's thermo-optic coefficient is larger, and it grows with temperature:
   **1.94e-4 /K at 300 K and 2.13e-4 /K at 375 K** (Li 1980 formula). The sources span 1.87e-4 to
   2.04e-4 at 300 K and 2.00e-4 to 2.13e-4 at 375 K. Over the 0 to 125 °C envelope, dn/dT changes by
   **+17 %** (from 1.86e-4 at 273 K to 2.18e-4 at 400 K). A constant dn/dT is not adequate for this
   envelope. [verified: Li formula re-run and checked against Li's printed Table 2; Frey Table 4 prints
   1.94e-4 at 1.3 µm, 295 K. The +17 % is Li's (and Cocorullo's, +18 % at 1523 nm); Franta 2017 gives
   only +12 % (1.82e-4 → 2.04e-4) over the same envelope, so the gain-schedule slope itself is uncertain
   by about ±3 %.]
2. **The thermal-shift ceiling for any Si-core, oxide-clad ring at 1310 nm** (all electric energy in
   Si, silicon's own dispersion included, substrate expansion added) is **71 pm/K (12.4 GHz/K) at
   300 K, 78 pm/K (13.6 GHz/K) at 375 K and 80 pm/K (13.9 GHz/K) at 400 K** (derived).
   Lightmatter's "~15 GHz/°C" (86 pm/K) is still above this ceiling at every temperature up to 400 K.
   It would need a silicon dn/dT of 2.35e-4 /K, which Li's formula reaches only near 500 K. The gap
   is smaller than the book's 1.48× (about 1.08× at 400 K to 1.26× at 273 K), but it does not close. The paper's own
   wording is an approximate aside in the introduction ("intrinsic thermal sensitivity of MRMs
   (~15 GHz/°C)", M4B.3 §1), not a measurement. [verified: wording is in M4B.3 §1 Introduction;
   15 GHz = 85.9 pm at 1310 nm; the first-order perturbation bound is exact for a lossless mode.
   Qualified: the table uses n_eff/n_g = 0.6 for the expansion term while taking Γ = 1 for the
   thermo-optic term. In the strict all-silicon limit (n_eff/n_g = n_Si/n_g,Si = 0.953) the ceiling is
   72.3 / 79.2 / 81.1 pm/K at 300 / 375 / 400 K, still 1.06× short of 86 pm/K at 400 K. With the
   Frey/Franta material group index (3.69 instead of Li's 3.674) the thermo-optic part drops by
   about 0.3 pm/K. The conclusion stands.]
3. **The absolute index of Si at 1310 nm differs by 4.1e-3 between Li 1980 (3.5016 at 300 K) and Frey
   2006 / Franta 2017 (3.5057 at 300 K).** That is twice Li's stated ±2e-3. Frey's measurement is the
   more recent one and has ±1e-4 uncertainty. A 4e-3 index error moves a ring's design resonance by
   about 1.2 to 1.3 nm (derived), so the FSR, n_g and resonance-order logic must use one model consistently.
   [verified, with two qualifiers: (a) Frey's index is absolute (in vacuum); Li's compilation appears
   to be air-referenced (Li converts Rank et al.'s Ge data "from vacuum values to air values"; his Si
   section does not state the reference). If so, about 0.9e-3 of the 4.1e-3 is the air/vacuum
   convention, and the like-for-like gap is about 3.2e-3, still 1.6× Li's ±2e-3. (b) The sources
   split two against two: Green 2008 (300 K) tabulates n(1.31 µm) = 3.502, at Li's level; Franta 2017
   (300 K) gives 3.5056, at Frey's level (Franta is built on Frey's data, and Green's sub-gap n is
   probably not independent of Li). The resonance shift is 1.2–1.4 nm for 80–95 % of the mode weight in Si.]
4. **The Frey 2006 Sellmeier must not be used above 300 K.** Its fourth-order polynomials diverge near
   325 K, where dn/dT jumps to about 1.6e-4 and d²n/dT² to about 1e-5. Below 300 K its derivative
   (1.82e-4 at 293 K) also undershoots its own measured Table 4 (1.94e-4 at 295 K, 1.3 µm).
   [verified: Table 5 coefficients re-typed from the arXiv PDF reproduce Table 2 exactly (3.49818 at
   250 K, 3.50616 at 295 K, 1.3 µm); the Sellmeier derivative is 1.827e-4 at 295 K, 1.3 µm, against
   the printed 1.94e-4; the fit becomes singular near 320 K.]
5. **Free-carrier dispersion at 1.3 µm**: the 2011 Nedeljkovic-Soref-Mashanovich fit gives a 10 %
   smaller |Δn| and a 13 % smaller Δα than the 1987 Soref-Bennett fit at ΔN = ΔP = 1e18 cm⁻³.
   The textbook implements Soref-Bennett correctly. Both models should be available behind one switch.
   [verified: arithmetic re-done; NSM 1.3 µm row matches the Tidy3D 2.12.0 source table (first row of
   `default_nedeljkovic_soref_mashanovich_coeffs`, wvl = 1.3 µm); the NSM paper itself was not
   opened by the checker either. SB 1.3 µm coefficients match two independent secondary quotations (a US patent on silicon phase modulators and a 2022 lecture deck, seen via search).]
6. **Thin-film silicon conducts heat well below the bulk value** [corrected: the original said "about
   half, 69 W/(m·K)", anchored to a mis-cited data point; see §5.2]: about **84 W/(m·K)** for a 220 nm
   film at 300 K (range about 73–95), against 148 for bulk (derived fit anchored to the measured
   90 ± 10 W/(m·K) of a 260 nm SOI device layer, Aubain & Bandaru 2010). Its temperature dependence is
   also weaker than bulk: −13 % from 300 to 375 K, against −25 % for bulk. Interfaces are negligible
   next to a 2 µm BOX: the Si/SiO₂ boundary resistance of about 2e-8 m²·K/W equals about 20 nm of oxide
   [verified: Lee & Cahill 1997 abstract].
7. Silicon is transparent across the O-band over the whole range. Band-to-band absorption is
   2.7e-5 cm⁻¹ at 1310 nm and 3.6e-4 cm⁻¹ at 1260 nm (Green 2008, 300 K) [verified: k = 2.8147e-10
   and 3.6096e-9 in the refractiveindex.info Green-2008 file]. The ring's absorption, and
   therefore its self-heating, come from free carriers, defects and sidewall scattering, not from the
   silicon lattice.

---

## 1. Summary table

Values at 1310 nm. "300 K" and "375 K" (about 102 °C) bracket the closed-loop junction range.

| Property | 300 K | 375 K | Model (section) | Valid range | Uncertainty | Source | Provenance |
|---|---|---|---|---|---|---|---|
| n_Si (Li 1980) | 3.5016 | 3.5169 | Li eq. 22 (§2) | 1.2–14 µm, 100–750 K (fit 20–1600 K) | ±2e-3 | Li, JPCRD 9, 561 (1980) | printed formula |
| n_Si (Frey level) | 3.5057 | 3.5210 | Li + 4.1e-3 offset (§2) | 1.2–1.4 µm | ±5e-4 | Frey et al. 2006; Franta et al. 2017 | derived |
| dn_Si/dT | 1.94e-4 /K | 2.13e-4 /K | ∂/∂T of Li (§2) | as Li | ±0.07e-4 (300 K), ±0.1e-4 (375 K) | Li 1980; spread over 5 sources | derived |
| d²n_Si/dT² | 2.78e-7 /K² | 2.22e-7 /K² | ∂²/∂T² of Li | as Li | 1.7–2.8e-7 across sources | Li 1980, Franta 2017, Cocorullo 1999 | derived |
| n_g,Si (material) | 3.674 | 3.694 | n − λ dn/dλ | as Li | ±0.02 [corrected from ±0.01: Frey 2006 Sellmeier gives 3.689 at 295 K and Franta 2017 3.690 at 300 K, both 0.016 above Li] | Li 1980 | derived |
| Ring shift ceiling | 71.1 pm/K | 77.8 pm/K | λ(dn/dT)/n_g,Si + λ(n_eff/n_g)α (§3) | Si-core rings | −4 to +3 pm/K (source spread) [verified; +1.2 to +1.4 pm/K more if the expansion term also takes the all-Si limit: 72.3 / 79.2 pm/K] | this file | derived |
| n_SiO₂ (fused) | 1.4468 | ≈1.4475 | Malitson Sellmeier (§4) | 0.21–3.71 µm, 20 °C | ±1e-4 bulk; films ±0.01 | Malitson 1965 | printed |
| dn_SiO₂/dT | 0.86e-5 /K | 1.04e-5 /K | linear in T (§4) | 30–310 K measured | 0.85–1.0e-5 at RT | Leviton & Frey; Corning; Arbabi 2013 | inferred above 300 K |
| n_Si₃N₄ (LPCVD) | 2.0031 | — | Luke Sellmeier (§4) | 0.31–5.5 µm, RT | ±3e-3 | Luke et al. 2015 | printed |
| dn_Si₃N₄/dT | 2.45e-5 /K | — | constant | RT, 1550 nm | ±0.09e-5 | Arbabi & Goddard 2013 | printed |
| k_Si bulk | 148 W/(m·K) | 111 W/(m·K) | 148 (T/300)^−1.3 (§5) | ~250–500 K | ±5 % | Palankovski 2000; CRC/TPRC | printed |
| k_Si film 220 nm | 84 W/(m·K) [corrected from 69] | 73 W/(m·K) [corrected from 62] | Matthiessen fit, d₀ = 170 nm (§5) | 150–300 nm | ±15 % (73–95 at 300 K) | fit to Aubain & Bandaru 2010 (260 nm: 90 ± 10) | derived |
| k_SiO₂ | 1.30–1.38 W/(m·K) | 1.49 W/(m·K) | 1.38 (T/300)^0.33 | 250–500 K | ±10 %; PECVD −10 % | Palankovski; Corning (prints both 1.30 and 1.38); Lee & Cahill 1997 | printed |
| ρc_Si | 1.656 MJ/(m³·K) | 1.818 MJ/(m³·K) | Palankovski c_p (§5) | 200–400 K | ±1.5 % | Palankovski; NIST-JANAF | derived |
| ρc_SiO₂ | 1.56 MJ/(m³·K) | 1.82 MJ/(m³·K) | Palankovski c_p | 250–500 K | ±5 % | Palankovski | derived |
| α_Si (CTE) | 2.57e-6 /K | 3.08e-6 /K | Okada–Tokumaru (§6) | 120–1500 K | ±2e-7 /K | Okada & Tokumaru 1984 | printed |
| α_SiO₂ (CTE) | 0.52e-6 /K | 0.57e-6 /K (0–200 °C avg) | constant | — | ±0.05e-6 | Corning 7980 data sheet | printed |
| Δn, Δα free carriers | see §7 | — | NSM 2011 1.3 µm row | 1e17–1e20 cm⁻³ | ~±10–20 % | Nedeljkovic et al. 2011 | printed |
| β_TPA | 0.9 cm/GW | — | constant (§8) | 1.3 µm, 300 K | 0.5–2.0 cm/GW | Lin 2007; Bristow 2007; Dinu 2003 | inferred central |
| n₂ (Kerr) | 4e-14 cm²/W | — | constant | 1.2–1.55 µm | 3–6e-14 | Bristow 2007; Dinu 2003 | inferred |
| ρ heater, n 1e19 | 5.5e-3 Ω·cm | 6.0e-3 Ω·cm | Arora (§9) | 250–500 K | ±10 % | Arora 1982; Masetti 1983 | derived |
| TCR doped Si | 1.2–1.8e-3 /K | 1.0–1.8e-3 /K | Arora, full ionization | 1e18–1e20 | ±50 % | Arora 1982 | derived |
| Ge PD responsivity limit | 1.057 A/W | 1.057 A/W | ηλ/1.2398 µm (§10) | — | — | physics | derived |
| Ge PD dark current | I₀ | ≈20 × I₀ (vs 298 K) | Arrhenius, E_a ≈0.37 eV | 25–85 °C data | E_a 0.35–0.38 eV | Colace 2008; Koester 2006 | inferred |
| E_g,Si | 1.1245 eV | 1.1042 eV | Varshni (§11) | 0–1000 K | ±2 meV | Thurmond 1975 | printed |
| n_i,Si | 9.70e9 cm⁻³ | 1.51e12 cm⁻³ | Misiakos–Tsamakis | 78–340 K | ±1 % at 300 K | Misiakos & Tsamakis 1993 | printed (extrapolated above 340 K) |
| V_bi (1e18/1e18, no BGN) | 0.954 V | 0.866 V | kT ln(N_A N_D/n_i²) | — | BGN lowers it by ~0.05 V | this file | derived |

---

## 2. Crystalline silicon refractive index n(λ, T)

### 2.1 Models and their ranges

| Model | Form | Range | Absolute uncertainty | Use here |
|---|---|---|---|---|
| **Li 1980**, JPCRD 9, 561, eq. (22) | n² = ε(T) + (L(T)/λ²)(A₀ + A₁T + A₂T²) | 1.2–14 µm; 100–750 K recommended (L(T) branches cover 20–1600 K) | n ±2e-3 (λ < 2.55 µm); dn/dT ±0.15e-4 /K | **the T dependence** |
| **Frey, Leviton, Madison 2006**, SPIE 6273, 62732J | 3-term Sellmeier, 4th-order T polynomials | 1.1–5.6 µm, **20–300 K only** | ±1e-4 (Table 1: 8.7e-5 at 1.5 µm, 295 K) | **the absolute level at 295 K** |
| Franta et al. 2017, Appl. Surf. Sci. 421, 405 | physics-based dispersion model built on Frey's data | 10–500 K (tabulated on refractiveindex.info at 250/300/350/400 K) | not stated in the file | cross-check |
| Green 2008, Sol. Energy Mater. Sol. Cells 92, 1305 | KK-consistent table | 300 K | — | absorption k |
| Cocorullo, Della Corte, Rendina 1999, APL 74, 3338 | dn/dT = 9.48e-5 + 3.47e-7 T − 1.49e-10 T² | **1523 nm**, 300–550 K | — | cross-check |
| Lukes 1959/1960 (tabulated in Li's Tables A-3 and A-4) | measured dn/dT | 1.13–5.1 µm, 118–522 K | not given | cross-check |
| Wang et al. 2025, APL 127, 061103 | ellipsometry of a 224 nm SOI film | 0.40–1.31 µm, 298–338 K | noisy dn/dT | thin-film cross-check |

**Li's coefficients (printed, eq. 22):**
ε(T) = 11.4445 + 2.7739e-4·T + 1.7050e-6·T² − 8.1347e-10·T³;
A₀ = 0.8948, A₁ = 4.3977e-4, A₂ = 7.3835e-8; λ in µm, T in K;
L(T) = exp(−3·ΔL(T)/L₂₉₃), with ΔL/L₂₉₃ = −0.071 + 1.887e-6·T + 1.934e-9·T² − 4.544e-13·T³ (293–1600 K)
and ΔL/L₂₉₃ = −0.021 − 4.149e-7·T − 4.620e-10·T² + 1.482e-11·T³ (20–293 K).

**A reading trap in Li's printed equation (derived check).** As printed, ΔL/L₂₉₃ does not vanish at
293 K. The formula reproduces Li's own Table 1 only if the constant term is read in percent, that
is as −7.1e-4 for the 293–1600 K branch: n(1.30 µm, 293 K) = 3.5016 and n(1.31 µm, 400 K) = 3.5223,
against tabulated 3.5016 and 3.5222. The literal reading gives 3.520, which is wrong by 0.019. The
code in §12 uses −7.1e-4. The 20–293 K branch has the same issue; it is outside our range and is not
coded. [verified: the NIST reprint prints "−0.071 + 1.887×10⁻⁶ T + …" (Li's ref. [34], the TPRC
expansion data, which tabulates ΔL/L in percent with coefficients 100× larger). Li divided the T terms
by 100 but not the constant. With −7.1e-4 the formula reproduces Li's Table 1 to ≤1e-4 (1.20 µm:
3.5084 / 3.5167 / 3.5284 / 3.5394 at 250 / 293 / 350 / 400 K against printed 3.5084 / 3.5167 / 3.5284 /
3.5393) and Table 2 to ≤0.003e-4 /K (1.30 µm, 400 K: 2.185 against 2.183).]

**Frey's coefficients (printed, Table 5):** the code in §12 carries them in full. Check: n(1.3 µm,
295 K) = 3.50616, which equals the measured Table 2 value.

### 2.2 Values at 1310 nm

| T | n (Li) | n (Frey/Franta level) | dn/dT (Li) | d²n/dT² (Li) | n_g,Si (Li) | dn/dT (Franta 2017, read-off + derived) |
|---|---|---|---|---|---|---|
| 273.15 K (0 °C) | 3.49652 | 3.50060 | 1.861e-4 | 2.98e-7 | 3.6666 | 1.82e-4 |
| 300 K | 3.50162 | 3.50570 | 1.938e-4 | 2.78e-7 | 3.6736 | 1.87e-4 |
| 325 K | 3.50655 | 3.51064 | 2.006e-4 | 2.59e-7 | 3.6802 | 1.92e-4 |
| 350 K | 3.51164 | 3.51573 | 2.068e-4 | 2.41e-7 | 3.6871 | 1.96e-4 |
| 375 K | 3.51689 | 3.52097 | 2.126e-4 | 2.22e-7 | 3.6941 | 2.00e-4 |
| 400 K (127 °C) | 3.52227 | 3.52635 | 2.179e-4 | 2.04e-7 | 3.7012 | 2.05e-4 |

The Franta dn/dT column comes from a cubic fit through the four tabulated Franta indices at 1.31 µm
(250, 300, 350, 400 K). Its d²n/dT² is 1.88e-7 at 300 K and 1.70e-7 at 375 K. [verified: refit of the
refractiveindex.info Franta-250K/300K/350K/400K rows at 1.31012 µm gives dn/dT = 1.871e-4 and 2.003e-4
and d²n/dT² = 1.87e-7 and 1.67e-7 at 300 and 375 K. Franta is also tabulated at 10–200 K, 450 K and
500 K, so the fit can be extended.]

**Material group index disagreement [added by checker].** The n_g,Si column is Li's. Frey's
Sellmeier gives n_g,Si(1.31 µm, 295 K) = 3.689 (its Table 3 dn/dλ = −0.148 /µm at 1.3 µm implies
3.699), and Franta's 300 K table gives 3.690 (dn/dλ ≈ −0.141 /µm). Li's dn/dλ is −0.134 /µm, so his
n_g is about 0.016 lower. The recommended `n_si()` (Li + λ-dependent Frey offset) inherits most of
Frey's dispersion (n_g ≈ 3.686), but `dlambda_dT_ceiling_pm()` uses pure Li; a ring model must take
n_g from the same index model it uses for n.

### 2.3 dn/dT at 1310 nm: every source side by side

| Source | Wavelength | dn/dT at ~300 K | dn/dT at ~375 K | Provenance |
|---|---|---|---|---|
| Li 1980 formula | 1.31 µm | 1.94e-4 | 2.13e-4 | derived from printed formula |
| Frey 2006, measured Table 4 | 1.3 µm, 295 K | 1.94e-4 | — (≤300 K) | printed [verified] |
| Frey 2006, Sellmeier derivative | 1.31 µm, 293 K | 1.82e-4 | invalid | derived; undershoots its own table |
| Franta 2017 tables | 1.31 µm | 1.87e-4 | 2.00e-4 | read-off + derived |
| Lukes 1959/60, in Li Table A-3/A-4 | 1.327–1.328 µm, 300 K; 1.259 µm, 316 K and 415 K | 2.04e-4 (1.327 µm) | 2.09e-4 at 316 K, 2.22e-4 at 415 K (1.259 µm) | printed [verified in the NIST reprint; interpolated to 375 K and scaled to 1.31 µm this is 2.14e-4; Li notes Lukes's values were "read from a figure"] |
| Cocorullo 1999 | 1.523 µm | 1.855e-4 | 2.04e-4 | printed formula [checker could not open the paper; the formula's 300 K value matches the independently quoted "1.85e-4 at 1523 nm, 300 K"] |
| Cocorullo scaled to 1310 nm by Li's wavelength ratio | 1.31 µm | 1.95e-4 | 2.13e-4 | derived |
| Wang 2025, 224 nm SOI film, ellipsometry | 1.305 µm | ≈2.3e-4 (298–338 K mean, noisy) | — | derived from tabulated n |

**Recommendation.** Use Li's formula for the temperature dependence, with dn/dT(1310 nm) =
**1.94e-4 ± 0.07e-4 /K at 300 K** and **2.13e-4 ± 0.1e-4 /K at 375 K**. Use Frey's 295 K level for the
absolute index (add +4.1e-3 to Li at 1.31 µm; +4.6e-3 at 1.26 µm; +3.7e-3 at 1.36 µm). For a
sensitivity study, the low corner is Franta (1.87e-4 / 2.00e-4) and the high corner is Lukes (2.04e-4
at 300 K). The thin-film value from Wang 2025 is higher still. It comes from a single ellipsometric
sample with step-to-step noise, so treat it as a flag, not as a datum.

**Doping.** The Della Corte group reported, for measurements at 1.5 µm from 30 to 300 °C, no
dependence of silicon's dn/dT on doping type, doping level or crystal orientation (Photonics West
2000; abstract only). We know of no O-band measurement.

**Absorption.** k(1.31 µm) = 2.81e-10 and k(1.26 µm) = 3.61e-9 (Green 2008, 300 K, printed), so
α = 4πk/λ = 2.7e-5 cm⁻¹ and 3.6e-4 cm⁻¹ (inferred). The band edge moves about 27 meV to the red from
300 to 400 K (Thurmond, §11). That is not enough to reach the O-band, so intrinsic absorption stays
below about 0.02 dB/cm at 1260 nm and 400 K (derived, order of magnitude).

---

## 3. What the material data imply for the ring's thermal shift (derived)

First-order perturbation of a resonant mode in dispersive media gives
Δλ/λ = Σᵢ ∫ nᵢ Δnᵢ |E|² dV / Σᵢ ∫ nᵢ n_g,i |E|² dV, where n_g,i is each material's own group index.
A weighted average cannot exceed its largest term, so a Si-core, SiO₂-clad ring obeys

dλ/dT ≤ λ·(dn_Si/dT)/n_g,Si + λ·(n_eff/n_g)·α_sub

The second term is the substrate-locked expansion of the ring's length (the textbook's Padmaraju form;
we take n_eff/n_g = 0.6). The oxide's own term, (dn/dT)/n_g ≈ 6e-6 /K, is 9× smaller than silicon's,
so the all-silicon case is the ceiling. A constant dn/dT also misses the 17 % rise in silicon's
dn/dT across 0 to 125 °C, which is why the table below is per temperature.

| T | thermo-optic ceiling | expansion term | total ceiling | in GHz/K | vs 86 pm/K (15 GHz/°C) |
|---|---|---|---|---|---|
| 273 K | 66.5 pm/K | 1.8 pm/K | 68.3 pm/K | 11.9 | 1.26× short |
| 300 K | 69.1 pm/K | 2.0 pm/K | 71.1 pm/K | 12.4 | 1.21× short |
| 350 K | 73.5 pm/K | 2.3 pm/K | 75.8 pm/K | 13.2 | 1.13× short |
| 375 K | 75.4 pm/K | 2.4 pm/K | 77.8 pm/K | 13.6 | 1.11× short |
| 400 K | 77.1 pm/K | 2.5 pm/K | 79.7 pm/K | 13.9 | 1.08× short |

With the Franta values the ceiling is 2 to 4 pm/K lower; with Lukes's it is about 3 pm/K higher at
300 K. [verified: formula re-derived. For a lossless mode, Δω/ω = −∫Δε|E|²/∫[∂(ωε)/∂ω|E|² + μ|H|²],
and ∫ε|E|² = ∫μ|H|² gives the n·n_g weights used here, so the "weighted average ≤ largest term" bound
is exact to first order. Table values re-computed. Qualified: the expansion column uses n_eff/n_g = 0.6;
the strict all-silicon limit (0.953) adds 1.2–1.5 pm/K, giving 72.3 / 79.2 / 81.1 pm/K at 300 / 375 /
400 K. Still 1.06× short of 86 pm/K at 400 K, and the dn/dT that 86 pm/K needs falls only to about
2.31e-4, which Li reaches near 480 K.] The textbook's Chapter 9 already has the right form of this bound (66 pm/K with silicon's
dispersion). It computes it with the 1523 nm value of dn/dT and at one temperature.

Consequences for the plant model:

- A real ring with 80 to 95 % of its electric energy in silicon should show about 58 to 76 pm/K
  between 300 and 400 K, rising roughly linearly with T (derived). The controller sees a plant gain
  that grows about 9 % from 300 to 375 K at a fixed confinement.
- To reach 86 pm/K at full confinement, silicon would need dn/dT ≈ 2.35e-4 /K. Li's formula reaches
  that near 500 K. Lightmatter's figure needs another mechanism or another reference temperature.
  Two candidates: the "°C" is read on a sensor that moves less than the ring does, or the figure is a
  rounded rule of thumb (the paper writes "~15 GHz/°C" in its introduction and gives no measurement).
  This stays an open question for Lightmatter.

---

## 4. Silicon dioxide and silicon nitride

**Fused silica index (Malitson 1965, printed):** n² − 1 = 0.6961663 λ²/(λ² − 0.0684043²) + 0.4079426
λ²/(λ² − 0.1162414²) + 0.8974794 λ²/(λ² − 9.896161²), λ in µm, 20 °C, 0.21–3.71 µm.
n(1.31 µm) = 1.44680, material n_g = 1.4616 (derived). Leviton and Frey's CHARMS index for Corning 7980
at 295 K is 1.44845 at 1.2 µm and 1.44502 at 1.5 µm (printed; ±1e-5). It agrees with Malitson within
1e-5 once Malitson's air-referenced index is multiplied by n_air ≈ 1.00027 (derived check). [verified:
Leviton & Frey Table 4; Malitson × 1.00027 gives 1.44844 and 1.44501.]

**Film oxides** differ from fused silica by up to about 0.01. Reactive e-beam SiO₂ deposited at
300 °C gives n(1.31 µm) = 1.4579 (Rodríguez-de Marcos et al. 2016, Opt. Mater. Express 6, 3622;
read-off from the tabulated data). An e-beam film deposited at 25 °C gives 1.4628, and one deposited at
300 °C gives 1.4598 (Franta et al. 2016, SPIE 9890). Arbabi and Goddard measured PECVD oxide at
1.428 ± 0.011 at 1550 nm. Thermally grown oxide is within a few 1e-3 of fused silica. **Use 1.447 for
thermal BOX and 1.45 ± 0.01 for PECVD or BEOL cladding (assumed spread).**

**dn/dT of SiO₂, every source:**

| Source | Material, λ | Value | Provenance |
|---|---|---|---|
| Leviton & Frey (arXiv 0805.0091), Table 6 | Corning 7980, 1.2–1.5 µm | 8.40–8.45e-6 at 295 K; 8.52–8.57e-6 at 300 K; 7.94–7.97e-6 at 275 K | printed [verified; slope 2.32–2.40e-8 /K² from the 275–300 K rows] |
| Corning 7980 data sheet | 1128.64 nm, near 22 °C | 9.6e-6 (and 10.0e-6 at 633 nm) | printed |
| Corning data-sheet polynomial evaluated at 1310 nm | C0 + C1/λ² + C2/λ⁴ + C3/λ⁶, ppm/K | 9.53e-6 | inferred (form checked against two printed rows) |
| Arbabi & Goddard 2013, Opt. Lett. 38, 3878 | PECVD SiOx, 1550 nm, RT | (0.95 ± 0.10)e-5 | printed [verified, abstract; also n_Si₃N₄ = 1.977 ± 0.003, dn/dT_Si₃N₄ = (2.45 ± 0.09)e-5, n_SiOx = 1.428 ± 0.011] |
| Elshaari et al. 2016, IEEE Photonics J. | PECVD oxide, RT | (0.96 ± 0.09)e-5 | printed (abstract) |
| Textbook REF `dnSiO2_dT` | — | 1e-5 | as used |

Model: dn/dT_SiO₂(T) = 8.45e-6 + 2.4e-8·(T − 295) /K. The slope is read from Leviton and Frey's
240 to 300 K rows. Above 300 K it is an extrapolation (assumed), and it reaches 1.04e-5 at 375 K. For
PECVD or BEOL claddings use 0.95e-5 at room temperature with the same slope. The oxide carries 5 to
20 % of the mode energy, and its (dn/dT)/n is about 9× smaller than silicon's, so a ±15 % error
here moves the ring sensitivity by less than about 1 pm/K.

**Si₃N₄ (only if a nitride layer sits in the mode or is used for the interleaver):** LPCVD Sellmeier
n² − 1 = 3.0249 λ²/(λ² − 0.1353406²) + 40314 λ²/(λ² − 1239.842²) (Luke et al. 2015, Opt. Lett. 40,
4823; printed). That gives n(1.31 µm) = 2.0031 and n_g = 2.042 (derived). Arbabi and Goddard measured
n(1550) = 1.977 ± 0.003 and dn/dT = (2.45 ± 0.09)e-5 /K (printed). Elshaari et al. give PECVD SiN
(2.51 ± 0.08)e-5 /K (printed, abstract). Corning lists a stress-optic coefficient of 35.0 nm/(cm·MPa),
that is 3.5e-12 /Pa, for fused silica (printed); §6 uses it.

---

## 5. Thermal conductivity, heat capacity and interfaces

### 5.1 Bulk silicon
k_Si(T) = 148·(T/300)^−1.3 W/(m·K), with κ₃₀₀ = 148 and exponent −1.3 (printed in Palankovski, PhD thesis,
TU Wien 2000, §3.2.3; literature κ₃₀₀ given as 150). It gives 167 W/(m·K) at 273 K, 121 at 350 K and
102 at 400 K. That matches the commonly quoted CRC/TPRC recommended set (168 / 148 / 98.9 at 273 / 300 /
400 K) within about 3 %. **The CRC set is quoted from memory; we did not re-open it (assumed).**
[verified: Palankovski thesis Table 3.5 (iue.tuwien.ac.at/phd/palankovski/node34.html) prints Si
κ₃₀₀ = 148, α = −1.3, reported 150; SiO₂ 1.38, +0.33, reported 1.4. The thesis compares the law with
data over 300–800 K, so below 300 K it is an extension.]
Glassbrenner and Slack (Phys. Rev. 134, A1058, 1964) is the classic data set; our recollection of
about 156 W/(m·K) at 300 K from it was not re-checked (assumed). A first-principles study finds
their data reliable from 100 to 500 K. Handbook values for silicon
vary widely (Electronics Cooling 1998). Uncertainty ±5 %.

### 5.2 The 100 to 250 nm silicon film (this is what heats in a ring)
Phonon-boundary scattering cuts in-plane conductivity well below bulk.

- Ju and Goodson, APL 74, 3005 (1999): films down to 74 nm, **up to 50 % reduction at room
  temperature**, effective dominant-phonon mean free path ~300 nm (printed, abstract) [verified in the
  paper text: "As much as 50% reduction from the bulk value is observed", i.e. the 74 nm film is
  still about 74 W/(m·K)].
- Asheghi et al., J. Heat Transfer 120, 30 (1998): **about 40 % below bulk at 0.42 µm** (printed,
  abstract via search) [verified, abstract: "nearly 40 percent less than that of bulk silicon for
  ds = 0.42 microns", i.e. about 89 W/(m·K)].
- [corrected: the original bullet read "A thermoreflectance study of 50 and 80 nm SOI films (MRS
  Online Proc. Library, 'In-plane thermal conductivity determination in SOI structures through
  thermoreflectance measurements'): 32 and 38 W/(m·K) at 293 K". That paper (M. S. Aubain and
  P. Bandaru, MRS Proc. 1267, DD12-01, 2010, doi:10.1557/PROC-1267-DD12-01) measured **one 260 ± 15 nm
  device layer** (lightly p-doped, 1 µm BOX) and reports **90 ± 10 W/(m·K)** at room temperature
  (printed, abstract and full text). It contains no 50 or 80 nm data. No source for "32 and 38 W/(m·K)
  at 50 and 80 nm" was found.]
- The 100 nm film of Zhu et al. (J. Appl. Phys. 2020, arXiv 2004.07428), 40–400 K, agrees with Liu
  and Asheghi (J. Heat Transfer 128, 75, 2006). Its values are only in a figure, which we did not
  digitize. [verified: the text states agreement with refs. 23 and 39 over 40–400 K; no number is printed.]

**Derived fit (Matthiessen form)** [corrected: d₀ changed from 250 nm to 170 nm]: 1/k = 1/k_bulk(T) +
1/k_b, with k_b = 148·(d/d₀)·c_p(T)/c_p(300) and **d₀ = 170 nm**, chosen so the fit returns the
measured 90 W/(m·K) at 260 nm, the measurement closest to the device thickness. Aubain's ±10 W/(m·K)
maps to d₀ = 125–221 nm and to 73–95 W/(m·K) for a 220 nm film. The verified data do not fit a single
d₀: the fit gives 45 W/(m·K) at 74 nm (Ju & Goodson: about 74) and 105 at 420 nm (Asheghi 1998:
about 89). Use it only for 150–300 nm films. The old d₀ = 250 nm fit (69 W/(m·K) at 220 nm) sits at
the bottom of the measured range and was anchored to the mis-cited 38 W/(m·K) at 80 nm.

| Film thickness | 273 K | 300 K | 375 K | 400 K |
|---|---|---|---|---|
| 100 nm | 55 | 55 | 51 | 50 |
| 150 nm | 71 | 69 | 63 | 60 |
| 200 nm | 83 | 80 | 70 | 67 |
| 220 nm | 87 | 84 | 73 | 69 |
| 250 nm | 93 | 88 | 76 | 72 |

All values in W/(m·K), ±15 % (derived; the 100 nm row is outside the anchored range and is probably
low, see Ju & Goodson). Lightmatter's printed strip is 400 × 200 nm (`[SMTAI]`), so the
200 nm row is the closest. The weak temperature dependence is itself a modelling result: a lumped
thermal resistance built on bulk silicon would overstate how fast R_th rises with temperature
(film: −13 % from 300 to 375 K at 220 nm; bulk: −25 %).

**Doping.** Asheghi et al., J. Appl. Phys. 91, 5079 (2002), measured B- and P-doped 3 µm films from
1e17 to 3e19 cm⁻³ at 15–300 K. The room-temperature reductions are not in the abstract. At the
modulator's ~1e18 cm⁻³, boundary scattering dominates in a 200 nm film, and doping adds little
(assumed: <10 %). For 1e19–1e20 contact and heater regions, a common engineering estimate is about
80 % of the undoped value (Electronics Cooling 1998; assumed).

### 5.3 Oxides, low-k, metals, interfaces

| Material | k at 300 K, W/(m·K) | T dependence | Source | Provenance |
|---|---|---|---|---|
| Fused silica | 1.38 (Palankovski); 1.30 (Corning 7980 data sheet, 25 °C) [qualified: other Corning HPFS 7980 documents (Corning germicidal-irradiation sheet 2020; Edmund Optics' Corning sheet) print 1.38; both numbers come from Corning] | 1.38·(T/300)^0.33, giving 1.49 at 375 K | Palankovski 2000; Corning | printed |
| Thermal oxide (BOX) | 1.2–1.4 | as above | Lee & Cahill 1997; bulk silica | inferred |
| PECVD SiO₂ (>100 nm) | ≈10 % below thermal oxide, ≈1.1–1.3 | as above | Lee & Cahill, J. Appl. Phys. 81, 2590 (1997), 78–400 K | printed (ratio) |
| PECVD SiO₂, apparent, 50–500 nm | 0.83–1.23 | — | SiC/SiO₂ TBR study, arXiv 1905.08570 | printed |
| Low-k SiOC:H | ~0.7 dense, falling to ~0.1 at 30 % porosity | — | 2014 microfin study (via search) | printed (abstract) |
| Cu (bulk) | ≈400 | weak | handbook | assumed |
| Cu (BEOL lines) | ≈250–380 | — | Wiedemann–Franz with ρ 2–2.5 µΩ·cm | derived |
| W (bulk) | ≈173 | weak | handbook | assumed |
| W (film, ρ 10–15 µΩ·cm) | ≈60–100 | — | WF plus a large phonon part (W: 62 % phonon, arXiv 2505.07613) | derived/assumed |
| TiN (film) | ≈10–30 | — | WF with ρ 50–200 µΩ·cm plus phonons | assumed |

**Thermal boundary resistance.** PECVD SiO₂ on Si: R_I ≈ 2e-8 m²·K/W at room temperature, the same as
about 20 nm of SiO₂ (Lee & Cahill 1997, printed) [verified, abstract: PECVD films on Si, 78–400 K,
3ω method]. Against a 2 µm BOX (about 1.5e-6 m²·K/W) it is about
1.4 %, so it is negligible. Metal/dielectric interfaces are about 10× larger: Cr/PECVD SiO₂ versus
PECVD SiO₂/Si in a 2008 study (printed ratio, abstract). That is about 2e-7 m²·K/W, or about 0.25 µm of
oxide, and it is **not** negligible for a metal heater separated from the ring by ≲1 µm of oxide
(derived).

### 5.4 Heat capacity and density

c_p(T) = c₃₀₀ + c₁·((T/300)^β − 1)/((T/300)^β + c₁/c₃₀₀) (Palankovski form; coefficients printed):

| Material | c₃₀₀ J/(kg·K) | c₁ | β | ρ kg/m³ | ρc at 273 / 300 / 375 / 400 K, MJ/(m³·K) |
|---|---|---|---|---|---|
| Si | 711 | 255 | 1.85 | 2329 | 1.577 / 1.656 / 1.818 / 1.858 |
| SiO₂ | 709 | 696 | 1.5 | 2200 (Corning 7980: 2.201 g/cm³) | 1.451 / 1.560 / 1.816 / 1.888 |

Check against NIST-JANAF for crystalline Si: C_p = 20.000 J/(mol·K) at 298.15 K and 22.142 at
400 K, that is 0.7121 and 0.7884 J/(g·K) (printed). The model gives 0.7088 and 0.7979, so it is within
0.5 % and 1.2 %. [verified: JANAF Si-001 table; Palankovski Table 3.7 prints Si 711 / 255 / 1.85 and
SiO₂ 709 / 696 / 1.5. Note that Palankovski lists 1000 J/(kg·K) as the reported c₃₀₀ for SiO₂ next to
the model's 709.] Film densities vary: thermal oxide is 2.20–2.27 g/cm³ and PECVD oxide 2.1–2.3
(assumed spread). Metals (assumed handbook values): ρc of Cu is about 3.45, W about 2.55 and TiN
about 3.1 MJ/(m³·K).

Thermal diffusivity (derived): Si 8.9e-5 m²/s, SiO₂ 8.8e-7 m²/s; Corning prints 0.0075 cm²/s =
7.5e-7 m²/s. **The thermal time constants of the ring are not in this file.** One relevant data point:
Borghi et al., Opt. Express 29, 4363 (2021), fit a single-pole τ_th = 270 ± 10 ns to a buried SiO₂-clad
Si microring (1550 nm). They also show that Newton cooling with one pole is inadequate and that a slow
and a fast thermal pole both appear (printed). This supports the textbook's two-pole network.
[verified in the arXiv full text: "τth = 270(10) ns" from the self-pulsing fit; FEM shows a fast
(~100 ns) and a slow (~1 µs) relaxation. The 270 ns comes from a one-pole fit with σ_FCD raised 2–3×,
so it is a lumped fit value, not a measured pole.]

---

## 6. Thermal expansion and the thermo-mechanical contribution

**Si** (Okada & Tokumaru, J. Appl. Phys. 56, 314, 1984; printed; 120–1500 K; FZ and CZ samples within
±2e-7 /K): α(T) = [3.725·(1 − exp(−5.88e-3·(T − 124))) + 5.548e-4·T]·1e-6 /K. That gives 2.33e-6 at
273 K, 2.57e-6 at 300 K, 3.08e-6 at 375 K and 3.21e-6 at 400 K. The textbook's 2.6e-6 is the 300 K
value. [verified: formula, 120–1500 K range and ±2e-7 /K spread match the J. Appl. Phys. 56, 314
abstract; values re-computed.] For 7–293 K, use the PTB interferometric data (Phys. Rev. B 92, 174113, 2015).

**SiO₂** (Corning 7980 data sheet, printed): mean CTE 0.52 ppm/K (5–35 °C), 0.57 ppm/K (0–200 °C),
0.48 ppm/K (−100 to 200 °C). Assume the same for thermal oxide; PECVD oxide is process dependent.

**Contribution to the ring shift (derived).** The silicon substrate is hundreds of µm thick, so it sets
the in-plane strain of the whole stack. (1) The ring's circumference grows with α_Si. This adds
λ·(n_eff/n_g)·α_Si = 2.0 pm/K at 300 K and 2.4 pm/K at 375 K. It is already in §3 and in the
textbook's eq. 9.3. (2) The Si core has the same CTE as the substrate, so in-plane mismatch stress in
the core is small. The oxide is stretched by (α_Si − α_SiO₂)·ΔT ≈ 2.05e-6 /K. That gives a biaxial
stress of E/(1 − ν)·Δα ≈ 72.7 GPa/0.84 × 2.05e-6 ≈ 0.18 MPa/K. With stress-optic coefficients of order
1e-12 to 4e-12 /Pa, the oxide index changes by about 1e-7 to 7e-7 /K, which is 1 to 8 % of its dn/dT.
Weighted by the oxide's share of the mode, this is ≲0.1 pm/K. Stress coupled into the core through
the cladding is estimated at ≲1 pm/K: the Si photoelastic coefficients p₁₁ ≈ −0.1 and p₁₂ ≈ 0.01 act on
a strain below about 1e-6 /K. **This is an order-of-magnitude estimate with no source behind it
(derived).** It cannot supply the 8 to 15 pm/K missing in §3.

---

## 7. Free-carrier plasma dispersion at 1.3 µm

**Nedeljkovic, Soref & Mashanovich, IEEE Photonics J. 3(6), 1171 (2011), 1.3 µm row (printed in the
paper's table; transcribed here from the Tidy3D implementation, flexcompute/tidy3d
`default_nedeljkovic_soref_mashanovich_coeffs`, rows ordered 1.3, 1.55, 2.0 … 14 µm):**

Δα [cm⁻¹] = a·ΔN_e^b + c·ΔN_h^d    Δn = −(p·ΔN_e^q + r·ΔN_h^s)    (ΔN in cm⁻³)

| λ | a | b | c | d | p | q | r | s |
|---|---|---|---|---|---|---|---|---|
| 1.3 µm | 3.48e-22 | 1.229 | 1.02e-19 | 1.089 | 2.98e-22 | 1.016 | 1.25e-18 | 0.835 |
| 1.55 µm | 8.88e-21 | 1.167 | 5.84e-20 | 1.109 | 5.40e-22 | 1.011 | 1.53e-18 | 0.838 |

[verified against the installed Tidy3D 2.12.0 source (coords wvl = [1.3, 1.55, 2.0 …], coeff =
a…s). This is the same transcription path the author used; the checker also could not open the IEEE
paper, so a typo shared by Tidy3D would not be caught.]

**Soref & Bennett, IEEE JQE 23, 123 (1987), 1.3 µm (printed; also in the textbook):**
Δn = −(6.2e-22·ΔN_e + 6.0e-18·ΔN_h^0.8); Δα = 6.0e-18·ΔN_e + 4.0e-18·ΔN_h. [verified against an
independent secondary quotations of the 1.3 µm relations (a US patent and a lecture deck, via search); original paper not opened.]

Side by side (derived):

| ΔN_e = ΔN_h | Δn NSM | Δn SB | Δα NSM [cm⁻¹] | Δα SB [cm⁻¹] |
|---|---|---|---|---|
| 1e17 | −2.52e-4 | −3.01e-4 | 0.60 | 1.0 |
| 1e18 | −1.92e-3 | −2.13e-3 | 8.69 | 10.0 |
| 5e18 | −8.10e-3 | −8.56e-3 | 56.9 | 50.0 |

Effective small-signal coefficients from NSM (derived):

| | 1e16 | 1e17 | 1e18 cm⁻³ |
|---|---|---|---|
| σ_e (dα/dN_e) | 2.0e-18 cm² | 3.3e-18 cm² | 5.7e-18 cm² |
| σ_h (dα/dN_h) | 3.0e-18 cm² | 3.6e-18 cm² | 4.4e-18 cm² |
| \|dn/dN_e\| | 5.5e-22 cm³ | 5.7e-22 cm³ | 5.9e-22 cm³ |
| \|dn/dN_h\| | 2.4e-21 cm³ | 1.6e-21 cm³ | 1.1e-21 cm³ |

Soref-Bennett's linear cross sections at 1.3 µm are σ_e = 6.0e-18 and σ_h = 4.0e-18 cm².

**Which to use.** NSM is newer and fits a wider data set. Its authors describe it as an update of the
commonly used 1.3 and 1.55 µm equations. The two differ by about 10 % in Δn at modulator doping, which
is within the spread of the underlying data. Keep both behind a switch. For a depletion modulator,
ΔN is the change in carriers in the depletion edge, so Δn is positive (the index rises) on reverse
bias. The textbook's `signedPow` handles this.

**Temperature dependence (not established at 1.3 µm in any source we found).** In the Drude picture Δn
∝ N/(n·m*_opt), so |Δn| per carrier falls slowly with T as the optical effective mass rises (Riffe,
JOSA B 19, 1092, 2002, cited for the trend only). Δα ∝ N/(n·m*²·μ_opt) rises as mobility falls. We
estimate (assumed) |Δn per carrier| −0.02 to −0.05 %/K and Δα +0.1 to +0.3 %/K. The larger
temperature effect on a depletion MRM comes from the junction instead: V_bi falls about 1.1 to
1.2 mV/K (§11), so the depletion width and the pm/V efficiency at fixed bias shift by a few percent
over 100 K (derived). Incomplete ionization at 1e17–1e18 cm⁻³ is another route (Liu et al. 2013 model
it; not quantified here). Kim et al., IEEE PTL 2021 ("A temperature-aware large-signal SPICE model for
depletion-type silicon ring modulators") is the closest published model; we could not open it.
[verified citation: M. Kim, Y. Jo, S. Lischke, C. Mai, L. Zimmermann, W.-Y. Choi, IEEE Photon.
Technol. Lett. 33(17), 947–950 (2021), doi:10.1109/LPT.2021.3098760.]

---

## 8. Nonlinear absorption, Kerr effect and carrier lifetime

| Quantity | Value at ~1.3 µm, 300 K | Source | Provenance |
|---|---|---|---|
| β_TPA | **0.57 ± 0.17 cm/GW at 1315 nm** | Q. Lin et al., APL 91, 021111 (2007), z-scan | printed (abstract) [verified; 0.48 ± 0.14 at 1501 nm, nearly flat to 1.7 µm] |
| β_TPA | **2.1 ± 0.4 cm/GW at 1220 nm**; peak 2 ± 0.5 cm/GW near the indirect gap | Bristow, Rotenberg, van Driel, APL 90, 191104 (2007) | printed [verified in the full text] |
| β_TPA | "~0.8 cm/GW" reported at 1300 and 1500 nm by earlier work | as summarised by Bristow 2007 | printed |
| β_TPA | 0.8 cm/GW at 1.54 µm | Dinu, Quochi, Garcia, APL 82, 2954 (2003) | printed (abstract) |
| β_TPA vs T | 0.76 cm/GW at 300 K → 0.42 at 5.5 K (1.55 µm, waveguide) | Sinclair et al., Phys. Rev. Applied 11, 044084 (2019) | printed (abstract) [verified] |
| n₂ | (4.7 ± 2.0)e-14 cm²/W at 1220 nm | Bristow 2007 | printed [verified: printed as (4.7 ± 2.0)e-5 cm²/GW] |
| n₂ | ~3e-14 cm²/W at 1300 and 1550 nm (earlier work); 4e-14 at 1.54 µm | Bristow 2007 summary; Dinu 2003 | printed |
| n₂ vs T | 5.2e-18 m²/W at 300 K → 3.9e-18 at 5.5 K (1.55 µm) | Sinclair 2019 | printed (abstract) |

**Model values:** β_TPA = **0.9 cm/GW** (9e-12 m/W), with a range of 0.5 to 2.0 and a factor-of-3
disagreement between z-scan groups near 1.3 µm. **We do not round this away.** n₂ = **4e-14 cm²/W**
(4e-18 m²/W), range 3 to 6e-14. Both are inferred central values. Above 300 K, β is expected to keep
rising with T (phonon-assisted, and the gap shrinks). No source we found quantifies this; assume
+0.1 to +0.3 %/K.

**Is TPA relevant to the ring? (derived, order of magnitude).** Take 2.5 mW on the bus (+4 dBm, G25),
a build-up of ~30 and an effective area of 0.08 µm². Then I ≈ 0.09 GW/cm², and α_TPA = βI ≈ 0.08 cm⁻¹
(0.4 dB/cm). That is small next to the doped junction's free-carrier loss: NSM gives 8.7 cm⁻¹ for full
overlap with 1e18/1e18. The generated carrier density is N ≈ βI²τ/(2ħω) ≈ 2.6e16 cm⁻³ for τ = 1 ns,
which gives Δn ≈ −8e-5 with NSM (equivalent to about −0.4 K). Under reverse bias the lifetime falls to tens of
ps, and this becomes negligible.

**Free-carrier lifetime.**
- Without a junction: SOI waveguide lifetimes "traditionally reported" as ~1 to 10 ns. Borghi et al.
  2021 measured 20 to 60 ns in a spiral waveguide, falling with carrier density (Shockley-Read-Hall fit:
  τ_n = 35 ns, τ_p = 22.5 ns, trap at E_i + 0.10 eV) (printed) [verified: Borghi Table 2, E_t = 0.66 eV,
  E_i = 0.56 eV]. Dimitropoulos et al., APL 86, 071115
  (2005) show the lifetime depends on rib geometry (citation confirmed; numbers not retrieved).
- With a reverse-biased PIN or PN junction: lifetimes fall to the tens-of-ps range. Turner-Foster et
  al., Opt. Express 18, 3582 (2010) is the usual reference. **We could not open it; the often-quoted
  "12.2 ps" is unverified here.** Model value: 10 to 100 ps (assumed).
- Free-carrier dispersion at the 1310 nm operating point: use the NSM derivatives above. Borghi had to
  fit σ_FCD = −9.5e-27 m³ at 1550 nm, which they note is 2 to 3× the literature value. That is a
  warning that lumped FCD coefficients absorb model error.

---

## 9. Heater materials: resistivity and temperature coefficient of resistance

**Doped single-crystal silicon (resistivity at 23 °C, printed reference table via search; Thurber/ASTM-type
data):** n (P): 2.3e-2, 5.5e-3 and 7.8e-4 Ω·cm at 1e18, 1e19 and 1e20 cm⁻³. p (B): 4.5e-2, 8.5e-3 and
1.2e-3 Ω·cm.

**Models (coefficients printed in the TCAD and nextnano documentation of the original papers):**
- Masetti, Severi, Solmi, IEEE TED 30, 764 (1983), 300 K. P: μ = 68.5 + (1414 − 68.5)/(1 + (N/9.20e16)^0.711)
  − 56.1/(1 + (3.41e20/N)^1.98). B: μ = 44.9·exp(−9.23e16/N) + 470.5/(1 + (N/2.23e17)^0.719) −
  29.0/(1 + (6.10e20/N)^2.0). The arsenic C_s is printed as 3.43e20 in one implementation and 3.34e20 in
  another (unresolved).
- Arora, Hauser, Roulston, IEEE TED 29, 292 (1982), 250–500 K, 1e13–1e20 cm⁻³, Tn = T/300.
  Electrons: μ = 88·Tn^−0.57 + 7.4e8·T^−2.33/(1 + (N/(1.26e17·Tn^2.4))^(0.88·Tn^−0.146)).
  Holes: μ = 54.3·Tn^−0.57 + 1.36e8·T^−2.23/(1 + (N/(2.35e17·Tn^2.4))^(0.88·Tn^−0.146)).
  Two implementation differences are unresolved: the electron N_ref is 1.26e17 (COMSOL) or 1.25e17
  (nextnano), and one course page prints the hole exponent as −2.33. We use −2.23, which reproduces
  407 cm²/(V·s) at 300 K.

**Derived (full ionization, ρ = 1/(qNμ)):**

| N, type | ρ Arora, 300 K | ρ Masetti, 300 K | TCR 300 K | TCR 375 K | ρ(375)/ρ(300) |
|---|---|---|---|---|---|
| 1e18 n | 2.38e-2 Ω·cm | 2.25e-2 | 1.26e-3 /K | 1.52e-3 /K | 1.11 |
| 1e18 p | 4.36e-2 | 3.89e-2 | 1.62e-3 | 1.82e-3 | 1.14 |
| 1e19 n | 5.47e-3 | 5.44e-3 | 1.25e-3 | 1.06e-3 | 1.09 |
| 1e19 p | 9.08e-3 | 8.53e-3 | 1.31e-3 | 1.12e-3 | 1.09 |
| 1e20 n | 6.82e-4 | 8.52e-4 | 1.75e-3 | 1.39e-3 | 1.12 |
| 1e20 p | 1.11e-3 | 1.25e-3 | 1.76e-3 | 1.40e-3 | 1.12 |

Caveats. Arora is weak above about 1e19 cm⁻³; at 1e20 its ρ is 13 % below the reference table, while
Masetti is within 9 %. At 1e17–1e18 cm⁻³ incomplete ionization makes the carrier density rise with T.
That lowers the TCR below these mobility-only values. **Uncertainty of the TCR: ±50 %.** No
side-by-side measured TCR table for 1e18–1e20 single-crystal Si turned up in this search. A
measurement on the actual heater is worth more than any of these numbers.

**Consistency with Lightmatter's heater (G25):** the textbook infers 55.9 Ω hot and fits a TCR of
1e-3 /K from the 20 to 30 % gain droop (derived, `REF.heaterTcrPerK`). That is at the low end of
doped-Si estimates (1.0–1.8e-3 /K) and about 2× TiN. Heater material is not published.

**Metal and nitride heaters:**

| Material | TCR | Source | Provenance |
|---|---|---|---|
| TiN (MEMS microheater) | 4.87e-4 /K | AIP Advances 2022 comparison (Ti 4.15e-3, Pt 2.64e-3) | printed (abstract) |
| TiN ("conventional" heater) | ≈5.5e-4 /K | TaN heater patent (US) | printed, patent |
| TiN, low-stress sputtered | nonlinear, can change sign | Sensors & Actuators A 2008 | printed (abstract) |
| W thin film | sign and size set by deposition (α-phase films positive) | Micromachines 2024 | printed (abstract) |
| W thick film / bulk | ≈4.6e-3 /K | thick-film patent; bulk handbook ≈4.5e-3 | printed / assumed |
| Cu (BEOL) | ≈3.9e-3 /K bulk | handbook | assumed |
| Silicides (NiSi, CoSi₂, TiSi₂) | metallic, positive, ~2–4e-3 /K; sheet ~2–10 Ω/sq | — | **assumed, no source found** |

---

## 10. Germanium photodiode at 1310 nm

- **Responsivity ceiling** (derived): R = ηλ/1.23984 µm = **1.0566 A/W** at η = 1 and 1310 nm [verified].
  Lightmatter's G25 receiver shows 0.88 to 1.04 A/W over 500 devices, with 0.9 A/W printed as the
  worst case (`[LM-T]`, textbook REF). The 1.04 A/W top end is 98 % of the quantum limit, so it needs
  near-unity internal efficiency or a calibration offset.
- **Responsivity against temperature at 1310 nm:** 1310 nm sits far above Ge's direct-gap edge (about
  0.80 eV at 300 K, near 1550 nm), so absorption is strong and the change should be small (inferred).
  Colace et al. (J. Lightwave Technol., 2008, "Temperature-dependence of Ge on Si p-i-n
  photodetectors") report that the trend depends on wavelength and Ge quality: defective films lose
  responsivity as T rises (printed, abstract). We found no 1310 nm data from 25 to 105 °C. **Model:
  constant, with an assumed ±3 % over 0 to 105 °C.**
- **Dark current against temperature:** about 1.6× per 10 °C (Colace 2008, printed, abstract), and about
  10× from room temperature to 85 °C (Koester et al. 2006, "Temperature-dependent analysis of Ge-on-SOI
  photodetectors and receivers", printed, abstract). An Arrhenius fit gives E_a ≈ 0.38 eV and 0.35 eV
  (inferred); we use **E_a = 0.37 eV**. Compared with 25 °C, that is ×11 at 85 °C and ×21 at 105 °C
  (derived). [verified: Colace et al., JLT 26, 2211 (2008) abstract prints "nearly a factor 1.6 every
  10°C"; that is E_a = 0.36–0.37 eV if read at 20–35 °C. Qualified: the same group (Balbi et al., abstract via search,
  Physica E 2009) reports 1.6–1.9× per 10 °C, which is E_a up to about 0.5 eV read near room
  temperature and up to about ×65 at 105 °C. Then a 5 nA room-temperature dark current reaches about
  0.3 µA, which is about 2 % of a 15 µA monitor signal, so the "below 1 %" statement below holds only
  for the 1.6× case.] Germanium-on-insulator vertical p-i-n diodes show 0.15 to 0.36 eV
  (SRH plus trap-assisted tunnelling; printed, abstract). A room-temperature dark current of a few nA
  at −1 to −3 V is typical (e.g. 4.62 nA at −3 V and 0.87 A/W at 1550 nm in "Low temperature characteristics of Ge-on-Si
  waveguide photodetectors", PMC12113715; printed). For a 15 µA monitor, even
  ×21 of a few nA stays below 1 % of signal (derived).

---

## 11. Junction physics: band gap, intrinsic density and built-in voltage

- **Band gap** (Varshni form, Thurmond, J. Electrochem. Soc. 122, 1133, 1975; as reproduced in device
  texts): E_g(T) = 1.170 − 4.73e-4·T²/(T + 636) eV. That gives 1.1312, 1.1245, 1.1042 and 1.0969 eV at
  273, 300, 375 and 400 K. Uncertainty about ±2 meV.
- **Intrinsic density** (Misiakos & Tsamakis, J. Appl. Phys. 74, 3293, 1993; printed; 78–340 K):
  n_i = 5.29e19·(T/300)^2.54·exp(−6726/T) cm⁻³, giving 9.70e9 at 300 K (±0.1e9). Values: 8.43e8 (273 K),
  6.67e10 (325 K), 3.53e11 (350 K), 1.51e12 (375 K) and 5.47e12 (400 K). **Above 340 K this is an
  extrapolation.** [verified: 78–340 K range and (9.7 ± 0.1)e9 at 300 K match the J. Appl. Phys. 74,
  3293 abstract; the fit constants were confirmed only from a secondary quotation.] An alternative is Couderc et al. 2014 (J. Appl. Phys. 115, 093705; abstract):
  n_i = 1.541e15·T^1.712·exp(−E_g0/(2kT)), with 9.65e9 at 300 K, the Altermatt 2003 value. We could
  not read which E_g0(T) it uses, so it is not coded.
- **Built-in voltage** (derived, step junction, full ionization, no band-gap narrowing):
  V_bi = (kT/q)·ln(N_A N_D/n_i²). At 1e18/1e18 cm⁻³ it is 0.984, 0.954, 0.866 and 0.835 V at 273,
  300, 375 and 400 K. dV_bi/dT = −1.12 mV/K at 300 K and −1.22 mV/K at 375 K. At 5e17/5e17 it is
  0.918 V at 300 K. Band-gap narrowing at ~1e18 lowers V_bi by a few tens of mV (assumed). The
  depletion width, W ∝ √(V_bi + V_R), therefore grows slightly with T at fixed reverse bias.

---

## 12. Code-ready section

Pure Python, no dependencies. Every function names its source and validity. The self-test reproduces
every derived number in this file.

```python
"""
O-band (1260-1360 nm) material models for a silicon microring plant, 273-400 K.

Units: wavelength lam in um, temperature T in K, carrier densities in cm^-3,
lengths in m unless a name says otherwise. Every function names its source and
its validity range. Outside the stated range the result is an EXTRAPOLATION.
Provenance words: printed / read-off / inferred / derived / assumed.
"""
import math

K_B_EV = 8.617333262e-5      # eV/K
Q_E = 1.602176634e-19        # C
H_C_EV_UM = 1.239841984      # eV*um

# ---------------------------------------------------------------- silicon index
def n_si_li1980(lam, T):
    """H. H. Li, J. Phys. Chem. Ref. Data 9, 561 (1980), eq. (22). 1.2-14 um, 20-1600 K
    (recommended 100-750 K). Stated uncertainty: n +/-2e-3 (lam < 2.55 um), dn/dT +/-0.15e-4 /K.
    NOTE: the printed constant of dL/L293 is '-0.071'; it must be read as -0.071 % (= -7.1e-4)
    for the formula to reproduce Li's own Table 1 (derived check: 3.5016 at 1.30 um, 293 K)."""
    eps = 11.4445 + 2.7739e-4 * T + 1.7050e-6 * T**2 - 8.1347e-10 * T**3
    dLL = -7.1e-4 + 1.887e-6 * T + 1.934e-9 * T**2 - 4.544e-13 * T**3   # 293-1600 K branch
    L = math.exp(-3.0 * dLL)
    return math.sqrt(eps + (L / lam**2) * (0.8948 + 4.3977e-4 * T + 7.3835e-8 * T**2))

_FREY_S = [[10.4907, -2.08020e-4, 4.21694e-6, -5.82298e-9, 3.44688e-12],
           [-1346.61, 29.1664, -0.278724, 1.05939e-3, -1.35089e-6],
           [4.42827e7, -1.76213e6, -7.61575e4, 678.414, 103.243]]
_FREY_L = [[0.299713, -1.14234e-5, 1.67134e-7, -2.51049e-10, 2.32484e-14],
           [-3.51710e3, 42.3892, -0.357957, 1.17504e-3, -1.13212e-6],
           [1.71400e6, -1.44984e5, -6.90744e3, -39.3699, 23.5770]]

def n_si_frey2006(lam, T):
    """Frey, Leviton, Madison, Proc. SPIE 6273, 62732J (2006), Table 5. 1.1-5.6 um, 20-300 K ONLY.
    Absolute uncertainty ~1e-4. The polynomial diverges above ~310 K: never call it above 300 K."""
    if T > 300.0:
        raise ValueError("Frey 2006 Sellmeier is valid only to 300 K")
    s = 0.0
    for Si, Li in zip(_FREY_S, _FREY_L):
        S = sum(c * T**j for j, c in enumerate(Si))
        l0 = sum(c * T**j for j, c in enumerate(Li))
        s += S * lam**2 / (lam**2 - l0**2)
    return math.sqrt(1.0 + s)

def n_si(lam, T, offset_frey=True):
    """Recommended: Li 1980 temperature dependence, shifted to Frey's absolute level at 295 K.
    offset = n_Frey(lam,295) - n_Li(lam,295) = +4.1e-3 at 1.31 um (derived)."""
    n = n_si_li1980(lam, T)
    if offset_frey:
        n += n_si_frey2006(lam, 295.0) - n_si_li1980(lam, 295.0)
    return n

def dn_dT(f, lam, T, h=0.5):
    return (f(lam, T + h) - f(lam, T - h)) / (2 * h)

def d2n_dT2(f, lam, T, h=2.0):
    return (f(lam, T + h) - 2 * f(lam, T) + f(lam, T - h)) / h**2

def n_group_material(f, lam, T, h=1e-3):
    return f(lam, T) - lam * (f(lam + h, T) - f(lam - h, T)) / (2 * h)

def dn_dT_si_cocorullo1523(T):
    """Cocorullo, Della Corte, Rendina, APL 74, 3338 (1999), 1523 nm, 300-550 K."""
    return 9.48e-5 + 3.47e-7 * T - 1.49e-10 * T**2

# ---------------------------------------------------------------- oxide, nitride
def n_sio2_malitson(lam):
    """Malitson, JOSA 55, 1205 (1965): fused silica, 20 C, 0.21-3.71 um."""
    l2 = lam * lam
    return math.sqrt(1 + 0.6961663 * l2 / (l2 - 0.0684043**2) + 0.4079426 * l2 / (l2 - 0.1162414**2)
                     + 0.8974794 * l2 / (l2 - 9.896161**2))

def dn_dT_sio2(T):
    """Fused silica (Corning 7980), 1.2-1.5 um: 8.45e-6 /K at 295 K (Leviton & Frey 2006/2008,
    printed, CHARMS, 30-310 K). Slope 2.4e-8 /K^2 read from their 275-300 K rows (inferred);
    above 300 K the linear extension is an EXTRAPOLATION (assumed). Film oxides: 0.95e-5 at RT
    (Arbabi & Goddard 2013, PECVD) - use that for BEOL/PECVD claddings."""
    return 8.45e-6 + 2.4e-8 * (T - 295.0)

def n_si3n4_luke(lam):
    """Luke et al., Opt. Lett. 40, 4823 (2015): LPCVD Si3N4, 0.31-5.5 um, room temperature."""
    l2 = lam * lam
    return math.sqrt(1 + 3.0249 * l2 / (l2 - 0.1353406**2) + 40314 * l2 / (l2 - 1239.842**2))

DN_DT_SI3N4 = 2.45e-5   # /K, LPCVD, Arbabi & Goddard 2013 (+/-0.09e-5), 1550 nm, RT

# ---------------------------------------------------------------- ring thermal-shift ceiling
def alpha_si(T):
    """Linear CTE of Si, Okada & Tokumaru, J. Appl. Phys. 56, 314 (1984), 120-1500 K, +/-2e-7 /K."""
    return (3.725 * (1 - math.exp(-5.88e-3 * (T - 124.0))) + 5.548e-4 * T) * 1e-6

def dlambda_dT_ceiling_pm(lam_um, T, neff_over_ng=0.6):
    """Largest thermo-optic + expansion shift any Si-core / SiO2-clad ring can have (derived):
    lam*(dn_Si/dT)/n_g,Si (all electric energy in Si) + lam*(neff/ng)*alpha_Si. pm/K."""
    f = n_si_li1980
    to = lam_um * 1e6 * dn_dT(f, lam_um, T) / n_group_material(f, lam_um, T)
    ex = lam_um * 1e6 * neff_over_ng * alpha_si(T)
    return to + ex, to, ex

# ---------------------------------------------------------------- thermal transport and storage
def k_si_bulk(T):
    """W/(m K). Power law used by Palankovski (TU Wien thesis 2000): 148*(T/300)^-1.3; reproduces the
    CRC/TPRC recommended bulk values within ~3 % over 250-400 K."""
    return 148.0 * (T / 300.0) ** -1.3

def k_si_film(d_nm, T, d0_nm=170.0):
    """In-plane k of a single-crystal SOI film (derived Matthiessen fit, +/-15 %, 150-300 nm only):
    1/k = 1/k_bulk(T) + 1/k_b,  k_b = k_bulk(300)*(d/d0)*(c_p(T)/c_p(300)).
    d0 = 170 nm reproduces 90 W/mK at 260 nm (Aubain & Bandaru, MRS Proc. 1267, DD12-01, 2010:
    90 +/- 10 W/mK, which maps to d0 = 125-221 nm). It gives 84 W/mK at 220 nm, 300 K.
    [corrected by checker: was d0 = 250 nm, anchored to a mis-cited '38 W/mK at 80 nm'.]
    Outside 150-300 nm it disagrees with data: 45 vs ~74 W/mK at 74 nm (Ju & Goodson 1999) and
    105 vs ~89 W/mK at 420 nm (Asheghi et al. 1998)."""
    kb = k_si_bulk(300.0) * (d_nm / d0_nm) * cp_si(T) / cp_si(300.0)
    return 1.0 / (1.0 / k_si_bulk(T) + 1.0 / kb)

def k_sio2(T, kind="thermal"):
    """W/(m K). Bulk fused silica 1.38*(T/300)^0.33 (Palankovski). PECVD ~10 % below thermal oxide
    for films >100 nm (Lee & Cahill 1997)."""
    k = 1.38 * (T / 300.0) ** 0.33
    return {"thermal": k, "fused": k, "pecvd": 0.9 * k}[kind]

def _cp_palankovski(T, c300, c1, beta):
    x = (T / 300.0) ** beta
    return c300 + c1 * (x - 1.0) / (x + c1 / c300)

def cp_si(T):
    """J/(kg K). Palankovski form, c300=711, c1=255, beta=1.85; within ~1.5 % of NIST-JANAF 200-400 K."""
    return _cp_palankovski(T, 711.0, 255.0, 1.85)

def cp_sio2(T):
    """J/(kg K). Palankovski form, c300=709, c1=696, beta=1.5 (amorphous SiO2)."""
    return _cp_palankovski(T, 709.0, 696.0, 1.5)

RHO_SI = 2329.0      # kg/m^3
RHO_SIO2 = 2200.0    # kg/m^3, fused silica; thermal oxide 2200-2270, PECVD 2100-2300

# ---------------------------------------------------------------- free carriers at 1.3 um
def plasma_nsm_1300(dNe, dNh):
    """Nedeljkovic, Soref, Mashanovich, IEEE Photon. J. 3, 1171 (2011), 1.3 um row.
    Returns (delta_n, delta_alpha [1/cm]); dNe, dNh >= 0 in cm^-3 (carriers ADDED)."""
    dn = -(2.98e-22 * dNe**1.016 + 1.25e-18 * dNh**0.835)
    da = 3.48e-22 * dNe**1.229 + 1.02e-19 * dNh**1.089
    return dn, da

def plasma_sb_1300(dNe, dNh):
    """Soref & Bennett, IEEE JQE 23, 123 (1987), 1.3 um."""
    return -(6.2e-22 * dNe + 6.0e-18 * dNh**0.8), 6.0e-18 * dNe + 4.0e-18 * dNh

# ---------------------------------------------------------------- doped-silicon heater
def mobility_arora(N, T, carrier="n"):
    """Arora, Hauser, Roulston, IEEE TED 29, 292 (1982); 250-500 K, N 1e13-1e20 cm^-3. cm^2/(V s)."""
    Tn = T / 300.0
    a = 0.88 * Tn ** -0.146
    if carrier == "n":
        return 88.0 * Tn ** -0.57 + 7.4e8 * T ** -2.33 / (1 + (N / (1.26e17 * Tn ** 2.4)) ** a)
    return 54.3 * Tn ** -0.57 + 1.36e8 * T ** -2.23 / (1 + (N / (2.35e17 * Tn ** 2.4)) ** a)

def mobility_masetti(N, carrier="n"):
    """Masetti, Severi, Solmi, IEEE TED 30, 764 (1983), 300 K; n = phosphorus, p = boron."""
    if carrier == "n":
        return 68.5 + (1414 - 68.5) / (1 + (N / 9.20e16) ** 0.711) - 56.1 / (1 + (3.41e20 / N) ** 1.98)
    return 44.9 * math.exp(-9.23e16 / N) + 470.5 / (1 + (N / 2.23e17) ** 0.719) - 29.0 / (1 + (6.10e20 / N) ** 2.0)

def resistivity_ohm_cm(N, T=300.0, carrier="n", model="arora"):
    mu = mobility_arora(N, T, carrier) if model == "arora" else mobility_masetti(N, carrier)
    return 1.0 / (Q_E * N * mu)

def tcr_doped_si(N, T=300.0, carrier="n", h=1.0):
    """TCR (1/K) assuming full ionization, mobility-only (Arora). Upper-bound-ish; see notes."""
    r = lambda t: resistivity_ohm_cm(N, t, carrier)
    return (r(T + h) - r(T - h)) / (2 * h) / r(T)

# ---------------------------------------------------------------- junction physics
def eg_si(T):
    """Varshni form, Thurmond, J. Electrochem. Soc. 122, 1133 (1975): 1.170 - 4.73e-4 T^2/(T+636) eV."""
    return 1.170 - 4.73e-4 * T**2 / (T + 636.0)

def ni_si(T):
    """Misiakos & Tsamakis, J. Appl. Phys. 74, 3293 (1993): 5.29e19 (T/300)^2.54 exp(-6726/T) cm^-3,
    78-340 K, 9.7e9 +/-0.1e9 at 300 K. Above 340 K: extrapolation."""
    return 5.29e19 * (T / 300.0) ** 2.54 * math.exp(-6726.0 / T)

def vbi(NA, ND, T):
    """Ideal step-junction built-in voltage (V), no band-gap narrowing, full ionization."""
    return K_B_EV * T * math.log(NA * ND / ni_si(T) ** 2)

# ---------------------------------------------------------------- germanium photodiode
def responsivity_limit(lam_um, eta=1.0):
    """A/W: eta*lam/1.23984. At 1.31 um and eta = 1: 1.0566 A/W."""
    return eta * lam_um / H_C_EV_UM

def ge_dark_current(I0, T, T0=298.15, Ea_eV=0.37):
    """Arrhenius scaling; Ea 0.35-0.38 eV inferred from x1.6 per 10 K (Colace 2008) and
    x10 from RT to 85 C (Koester 2006)."""
    return I0 * math.exp(-Ea_eV / K_B_EV * (1.0 / T - 1.0 / T0))


if __name__ == "__main__":
    lam = 1.31
    print("Li check 1.30um/293K (Table 1: 3.5016):", round(n_si_li1980(1.30, 293.0), 4))
    print("Frey check 1.3um/295K (Table 2: 3.50616):", round(n_si_frey2006(1.3, 295.0), 5))
    for T in (273.15, 300.0, 325.0, 350.0, 375.0, 400.0):
        tot, to, ex = dlambda_dT_ceiling_pm(lam, T)
        print(f"T={T:6.2f} n_Li={n_si_li1980(lam,T):.5f} n_rec={n_si(lam,T):.5f} "
              f"dn/dT={dn_dT(n_si_li1980,lam,T):.4e} d2n/dT2={d2n_dT2(n_si_li1980,lam,T):.3e} "
              f"ng_Si={n_group_material(n_si_li1980,lam,T):.4f} ceiling={tot:.1f} pm/K (TO {to:.1f} + exp {ex:.2f}) "
              f"alpha_Si={alpha_si(T):.3e} k_bulk={k_si_bulk(T):.1f} k_220nm={k_si_film(220,T):.1f} k_100nm={k_si_film(100,T):.1f} "
              f"k_SiO2={k_sio2(T):.3f} cp_Si={cp_si(T):.1f} cp_SiO2={cp_sio2(T):.1f} "
              f"rhoc_Si={RHO_SI*cp_si(T)/1e6:.3f} rhoc_SiO2={RHO_SIO2*cp_sio2(T)/1e6:.3f} "
              f"Eg={eg_si(T):.4f} ni={ni_si(T):.3e} dnSiO2/dT={dn_dT_sio2(T):.3e} Coc1523={dn_dT_si_cocorullo1523(T):.4e}")
    for T in (273.15, 300.0, 375.0, 400.0):
        print(f"T={T} Vbi(1e18,1e18)={vbi(1e18,1e18,T):.4f} V  Vbi(5e17,5e17)={vbi(5e17,5e17,T):.4f}")
    h = 0.5
    print("dVbi/dT (1e18/1e18) at 300 K: %.3e V/K" % ((vbi(1e18,1e18,300+h)-vbi(1e18,1e18,300-h))/(2*h)))
    for N in (1e18, 1e19, 1e20):
        for c in ("n", "p"):
            print(f"N={N:.0e} {c}: rho_arora={resistivity_ohm_cm(N,300,c):.3e} rho_masetti={resistivity_ohm_cm(N,300,c,'masetti'):.3e} "
                  f"TCR300={tcr_doped_si(N,300,c):.3e} TCR375={tcr_doped_si(N,375,c):.3e} rho375/rho300={resistivity_ohm_cm(N,375,c)/resistivity_ohm_cm(N,300,c):.3f}")
    for N in (1e17, 1e18, 5e18):
        print("NSM 1.3um N=P=%.0e:" % N, plasma_nsm_1300(N, N), " SB:", plasma_sb_1300(N, N))
    print("responsivity limit 1.31 um:", responsivity_limit(1.31))
    print("Ge dark x at 85C, 105C vs 25C:", ge_dark_current(1, 358.15), ge_dark_current(1, 378.15))
    print("Si3N4 n(1.31):", n_si3n4_luke(1.31), " SiO2 Malitson n(1.31):", n_sio2_malitson(1.31))
```

Self-test output (1310 nm):

```
T=300.00 n_Li=3.50162 n_rec=3.50570 dn/dT=1.9384e-04 d2n/dT2=2.779e-07 ng_Si=3.6736 ceiling=71.1 pm/K
T=375.00 n_Li=3.51689 n_rec=3.52097 dn/dT=2.1260e-04 d2n/dT2=2.224e-07 ng_Si=3.6941 ceiling=77.8 pm/K
k_bulk 148.0 / 110.7, k_220nm 83.5 / 72.5, cp_Si 711.0 / 780.7, Eg 1.1245 / 1.1042, ni 9.696e9 / 1.514e12
```

Implementation notes for the plant model:
- Make dn_Si/dT and d²n/dT² functions of T, not constants. A single constant from 1.86e-4 to 2.18e-4
  is a 17 % plant-gain error across the envelope.
- Carry two absolute-index options, Li and Frey-level, and pick one per study. Never mix them inside
  one ring calculation.
- Plasma dispersion: a switch between `nsm2011` and `soref1987`; default `nsm2011`.
- Conductivity of the device layer: use the film model, not bulk. The BOX and cladding should use
  1.3 to 1.4 W/(m·K) with the +0.33 power law.
- Keep every coefficient in one data file, `02_materials.json`, so a test can sweep them.

---

## 13. Disagreements with the textbook (`src/physics/index.ts`, Ch02, Ch03, Ch09)

| Textbook | Our view | Evidence |
|---|---|---|
| `REF.dnSi_dT = 1.86e-4` /K, used at 1310 nm and as a constant (Ch02 line ~1858, Ch09) | At 1310 nm: 1.94e-4 at 300 K, 2.13e-4 at 375 K, 1.86e-4 only at 273 K. 1.86e-4 is the 1523 nm, 300 K value | Li 1980; Frey 2006 Table 4 (1.94e-4 at 1.3 µm, 295 K); Cocorullo 1999 |
| Ch09: "no silicon-and-silica guide can pass about 70 pm/K", computed with 1.86e-4 (66 pm/K with dispersion) | The form is right. With the 1310 nm, T-dependent dn/dT the ceiling is 69 pm/K (300 K) to 77 pm/K (400 K) thermo-optic, or 71–80 pm/K with expansion. 86 pm/K is still out of reach | §3 |
| `src/physics` comment: "Γ = 1 caps at 58 pm/K" | That figure uses (λ/n_g)·dn/dT with n_g = 4.2 for the waveguide. The rigorous energy-weighted bound uses silicon's own n_g (3.67) and is about 71 pm/K at 300 K. Ch09 already corrects this; the comment in `index.ts` is stale | Ch09 lines 1140–1176; §3 |
| `REF.dnSiO2_dT = 1e-5` | 0.85e-5 (fused silica, 295 K) to 0.95e-5 (PECVD, RT), rising about 2.4e-8 /K² | Leviton & Frey; Arbabi & Goddard 2013; Corning |
| `nSilicon()` Sellmeier (ε = 11.6858 + …, no T) gives 3.5005 at 1.31 µm | Room-temperature only. It sits with Li (3.5003 at 293 K) and 4e-3 below Frey (3.5044 at 293 K). Fine for teaching, not for an FSR or n_g design | §2 |
| Ch09 `ALPHA_SUB = 2.6e-6` constant | Correct at 300 K; 3.1e-6 at 375 K (Okada–Tokumaru). Worth 0.4 pm/K | §6 |
| `plasmaDispersion1310` = Soref–Bennett 1987 | Correctly transcribed. The newer NSM 2011 fit gives about 10 % smaller Δn and 13 % smaller Δα at 1e18 | §7 |
| `REF.heaterTcrPerK = 1e-3` (fitted) | Plausible: doped Si 1.0–1.8e-3 /K (derived), TiN ~0.5e-3, W and silicides 2–4.5e-3 | §9 |

---

## 14. Open questions

1. What reference temperature and which thermometer define Lightmatter's "~15 GHz/°C"? The material
   ceiling at 1310 nm is 12.4 GHz/K at 300 K and 13.9 GHz/K at 400 K.
2. Is the 1310 nm absolute index of Lightmatter's device layer closer to Li or to Frey? A 4e-3
   difference moves the design resonance by about 1.2 to 1.3 nm.
3. Heater material and doping: doped Si, silicide or metal? The TCR ranges from 0.5e-3 to 4.5e-3 /K
   across the candidates.
4. Modulator doping profile (N_A, N_D). It sets V_bi(T), the depletion width and FCA loss, and
   therefore the self-heating fraction.
5. β_TPA at 1310 nm: z-scan groups disagree by 3× (0.57 vs ~2 cm/GW).
6. Temperature dependence of free-carrier dispersion and absorption at 1.3 µm above 300 K: no
   measurement found.
7. In-plane conductivity of a 200 nm, ~1e18-doped, partially etched rib at 300–400 K: no direct
   measurement found; ±15 % model anchored to one 260 nm measurement (Aubain & Bandaru 2010)
   [corrected from ±25 %, which was centred on a mis-cited value].
8. Ge PD responsivity at 1310 nm from 0 to 105 °C: no measurement found.

---

## 15. Sources (all cited above)

- H. H. Li, "Refractive index of silicon and germanium and its wavelength and temperature derivatives",
  J. Phys. Chem. Ref. Data 9, 561 (1980), doi:10.1063/1.555624 (NIST reprint read).
- B. J. Frey, D. B. Leviton, T. J. Madison, "Temperature-dependent refractive index of silicon and
  germanium", Proc. SPIE 6273, 62732J (2006), arXiv:physics/0606168.
- D. Franta et al., "Temperature-dependent dispersion model of float zone crystalline silicon",
  Appl. Surf. Sci. 421, 405 (2017) (tabulated data via refractiveindex.info).
- M. A. Green, Sol. Energy Mater. Sol. Cells 92, 1305 (2008) (via refractiveindex.info).
- G. Cocorullo, F. G. Della Corte, I. Rendina, APL 74, 3338 (1999); F. G. Della Corte et al., J. Appl.
  Phys. 88, 7115 (2000) (not opened).
- X. Wang et al., APL 127, 061103 (2025) (via refractiveindex.info).
- D. B. Leviton, B. J. Frey, "Temperature-dependent absolute refractive index measurements of synthetic
  fused silica", arXiv:0805.0091 (SPIE 6273, 2006).
- I. H. Malitson, JOSA 55, 1205 (1965). Corning 7980 data sheet (UQG Optics copy).
- L. V. Rodríguez-de Marcos et al., Opt. Mater. Express 6, 3622 (2016); D. Franta et al., Proc. SPIE
  9890, 989014 (2016).
- A. Arbabi, L. L. Goddard, Opt. Lett. 38, 3878 (2013); A. W. Elshaari et al., IEEE Photonics J. (2016).
- K. Luke et al., Opt. Lett. 40, 4823 (2015).
- V. Palankovski, PhD thesis, TU Wien (2000), §3.2.3–3.2.4 (iue.tuwien.ac.at/phd/palankovski).
- NIST-JANAF Thermochemical Tables, Si (cr).
- S.-M. Lee, D. G. Cahill, J. Appl. Phys. 81, 2590 (1997).
- W. Ju, K. E. Goodson, APL 74, 3005 (1999); M. Asheghi et al., J. Heat Transfer 120, 30 (1998);
  M. Asheghi et al., J. Appl. Phys. 91, 5079 (2002); W. Liu, M. Asheghi, J. Heat Transfer 128, 75 (2006);
  T. Zhu et al., arXiv:2004.07428; M. S. Aubain, P. Bandaru, MRS Proc. 1267, DD12-01 (2010),
  doi:10.1557/PROC-1267-DD12-01 [added by checker; replaces the mis-cited "MRS 50/80 nm" study].
- Y. Okada, Y. Tokumaru, J. Appl. Phys. 56, 314 (1984).
- M. Nedeljkovic, R. Soref, G. Z. Mashanovich, IEEE Photonics J. 3, 1171 (2011),
  doi:10.1109/JPHOT.2011.2171930; coefficients via flexcompute/tidy3d `parameter_perturbation.py`.
- R. A. Soref, B. R. Bennett, IEEE J. Quantum Electron. 23, 123 (1987).
- D. J. Riffe, JOSA B 19, 1092 (2002) (trend only).
- A. D. Bristow, N. Rotenberg, H. M. van Driel, APL 90, 191104 (2007); Q. Lin et al., APL 91, 021111
  (2007); M. Dinu, F. Quochi, H. Garcia, APL 82, 2954 (2003); G. F. Sinclair et al., Phys. Rev. Applied
  11, 044084 (2019).
- M. Borghi, D. Bazzanella, M. Mancinelli, L. Pavesi, Opt. Express 29, 4363 (2021), arXiv:2012.05989.
- A. C. Turner-Foster et al., Opt. Express 18, 3582 (2010) (not opened); D. Dimitropoulos et al., APL 86,
  071115 (2005).
- G. Masetti, M. Severi, S. Solmi, IEEE TED 30, 764 (1983); N. D. Arora, J. R. Hauser, D. J. Roulston,
  IEEE TED 29, 292 (1982).
- W. R. Thurmond, J. Electrochem. Soc. 122, 1133 (1975); K. Misiakos, D. Tsamakis, J. Appl. Phys. 74,
  3293 (1993); R. Couderc et al., J. Appl. Phys. 115, 093705 (2014) (abstract).
- L. Colace, M. Balbi, V. Sorianello, G. Assanto, J. Lightwave Technol. 26, 2211 (2008),
  doi:10.1109/JLT.2008.917080; S. J. Koester et al. (2006) (abstracts).
- R. Anand et al., OFC 2026 M4B.3 (Lightmatter), §1 and §3.2.
- Kenny's textbook: `src/physics/index.ts` (REF), `docs/SOURCES_AND_GENERATIONS.md`, Ch02, Ch03, Ch09.

---

## Verification log

Independent check, 2026-10-09. Method: re-run every formula in §12 and re-derive by hand; open the
cited source (or another primary source) and try to refute each claim. "Confirmed" means the number
was reproduced from the source or by independent arithmetic. "Corrected" means the report was edited
in place (marked `[corrected: …]`). "Qualified" means the claim stands but a disagreement or a caveat
was added (marked `[verified …]` with the qualifier). "Unverifiable" means the checker could not open
the primary source either.

| # | Claim (where) | Verdict | Evidence |
|---|---|---|---|
| 1 | Lightmatter prints "~15 GHz/°C" as an aside in M4B.3 §1; 15 GHz = 86 pm/K at 1310 nm (headline 2) | Confirmed | `pdftotext` of the M4B.3 PDF, §1 Introduction, lines 32–33; λ²Δf/c = 85.86 pm (derived). |
| 2 | Li 1980 eq. 22 coefficients; ΔL/L₂₉₃ constant must be read as −7.1e-4 (§2.1) | Confirmed | NIST reprint p. 567–568 prints −0.071 with T terms in units 100× smaller than TPRC's percent form. With −7.1e-4 the code reproduces Table 1 (≤1e-4) and Table 2 (≤0.003e-4 /K) at 1.20, 1.26 and 1.30 µm, 250–400 K. |
| 3 | dn_Si/dT(1310 nm) = 1.94e-4 (300 K) and 2.13e-4 (375 K); Frey Table 4 prints 1.94e-4 at 1.3 µm, 295 K (headline 1, §2.3) | Confirmed, qualified | Li formula re-run: 1.938e-4 / 2.126e-4 (derived). Frey arXiv PDF Table 4: 1.94E-04 (printed). Franta 2017 tables refit: 1.871e-4 / 2.003e-4 (read-off + derived). Qualifier added: the 0–125 °C rise is +17 % (Li) but +12 % (Franta). |
| 4 | Frey 2006 Table 5 coefficients; n(1.3 µm, 295 K) = 3.50616; Sellmeier derivative 1.82e-4 undershoots Table 4; diverges just above 300 K (headline 4) | Confirmed | Coefficients match the arXiv PDF exactly and reproduce Table 2 (3.49818 at 250 K, 3.50616 at 295 K). Derivative 1.827e-4 at 295 K; d²n/dT² goes singular near 320 K (derived). |
| 5 | Li and Frey differ by 4.1e-3 at 1.31 µm, twice Li's ±2e-3 (headline 3) | Confirmed, qualified | Offset 4.09e-3 (derived). Qualifiers added: Frey is a vacuum (absolute) index, and Li appears air-referenced (≈0.9e-3 of the gap); Green 2008 (3.502) sides with Li, Franta (3.5056) with Frey. |
| 6 | Material group index n_g,Si = 3.674 ± 0.01 at 300 K (summary table, §2.2) | Corrected | Frey Sellmeier gives 3.689 at 1.31 µm, 295 K; Frey Table 3 (dn/dλ = −0.148 /µm) implies 3.699 at 1.3 µm; Franta 300 K table gives 3.690. Li's dn/dλ (−0.134 /µm) is the outlier. Uncertainty widened to ±0.02. |
| 7 | Thermal-shift ceiling 71.1 / 77.8 / 79.7 pm/K at 300 / 375 / 400 K, and the perturbation formula behind it (headline 2, §3) | Confirmed, qualified | Weights n_i·n_g,i follow exactly from Δω/ω = −∫Δε\|E\|²/∫[∂(ωε)/∂ω\|E\|² + μ\|H\|²] with ∫ε\|E\|² = ∫μ\|H\|² (derived). Table re-computed. Qualifier: the expansion term used n_eff/n_g = 0.6; the strict all-Si limit gives 72.3 / 79.2 / 81.1 pm/K, still below 86 pm/K. 86 pm/K then needs dn/dT ≈ 2.31e-4 (Li: ≈480 K). |
| 8 | Cocorullo 1999: dn/dT = 9.48e-5 + 3.47e-7 T − 1.49e-10 T² at 1523 nm; 1.86e-4 is its 300 K value (headline 1, §13) | Partly confirmed | Formula gives 1.855e-4 at 300 K, matching an independent quotation of "1.85 × 10⁻⁴ K⁻¹ at 300 K, 1523 nm". The APL paper itself could not be opened, so the three coefficients are unverifiable here. |
| 9 | Lukes values in Li Tables A-3/A-4 (§2.3) | Confirmed | NIST reprint p. 623: 2.04 at 1.327/1.328 µm, 300 K; 2.09 at 315.9 K and 2.22 at 415.4 K, 1.259 µm (printed, Li notes "read from a figure"). |
| 10 | Okada–Tokumaru CTE formula and values (§6) | Confirmed | J. Appl. Phys. 56, 314 abstract (via search) prints the same formula, 120–1500 K, ±2e-7 /K; values re-computed. |
| 11 | NSM 2011 1.3 µm coefficients; SB 1987 1.3 µm coefficients; NSM gives 10 % smaller Δn and 13 % smaller Δα at 1e18 (headline 5, §7) | Confirmed (NSM: transcription only) | NSM row matches Tidy3D 2.12.0 source (wvl coordinate 1.3 µm). The IEEE paper could not be opened by the checker either, so a shared typo cannot be excluded. SB 1.3 µm matches two secondary quotations (a US patent, a lecture deck; via search). Side-by-side numbers re-computed. |
| 12 | Palankovski: k_Si = 148 (T/300)^−1.3; k_SiO₂ = 1.38 (T/300)^0.33; c_p parameters (§5) | Confirmed | Palankovski thesis Tables 3.5 and 3.7 (TU Wien web version). Note: the thesis validates the k law over 300–800 K. |
| 13 | NIST-JANAF C_p(Si) = 20.000 and 22.142 J/(mol·K) at 298.15 and 400 K (§5.4) | Confirmed | janaf.nist.gov Si-001 table. |
| 14 | Thin-film k: "MRS thermoreflectance study, 32 and 38 W/(m·K) at 50 and 80 nm"; 220 nm film ≈ 69 W/(m·K) ± 25 % (headline 6, §5.2, §12) | **Corrected** | The cited paper (Aubain & Bandaru, MRS Proc. 1267, DD12-01, 2010) measured only a 260 nm layer: 90 ± 10 W/(m·K) (printed, abstract and full text). No source for 32/38 was found. Ju & Goodson 1999 (printed): ≤50 % reduction down to 74 nm. Asheghi 1998 (printed): ~40 % below bulk at 0.42 µm. Fit re-anchored to d₀ = 170 nm: 84 W/(m·K) at 220 nm, 300 K (73–95), −13 % to 375 K. §5.2 table, §12 code, the self-test line and `02_materials.json` were updated. |
| 15 | Lee & Cahill 1997: R_I ≈ 2e-8 m²·K/W ≈ 20 nm SiO₂ (§5.3) | Confirmed | Abstract (via OSTI/search). |
| 16 | Corning 7980 k = 1.30 W/(m·K) at 25 °C (§5.3) | Confirmed, qualified | One Corning-derived sheet prints 1.30; Corning's 2020 HPFS 7980 sheet and Edmund's Corning sheet print 1.38. Both are now listed. |
| 17 | Leviton & Frey fused silica dn/dT 8.40–8.45e-6 (295 K), slope ≈2.4e-8 /K²; index 1.44845 / 1.44502; agreement with Malitson × n_air (§4) | Confirmed | arXiv 0805.0091 Tables 4 and 6 (printed); Malitson × 1.00027 = 1.44844 / 1.44501 (derived). |
| 18 | Arbabi & Goddard 2013 values (§4) | Confirmed | Abstract: 1.977 ± 0.003, (2.45 ± 0.09)e-5, 1.428 ± 0.011, (0.95 ± 0.10)e-5. |
| 19 | TPA and Kerr: Lin 0.57 ± 0.17 cm/GW at 1315 nm; Bristow 2.1 ± 0.4 cm/GW and n₂ (4.7 ± 2.0)e-14 cm²/W at 1220 nm, peak 2 ± 0.5, earlier ~0.8 cm/GW and ~3e-14; Sinclair 0.76 → 0.42 cm/GW (§8) | Confirmed | Lin abstract (via search); Bristow full text (APL 90, 191104); Sinclair arXiv abstract. |
| 20 | Borghi 2021: τ_th = 270 ± 10 ns, σ_FCD = −9.5e-27 m³ (2–3× literature), lifetimes 20–60 ns, SRH τ_n = 35 ns, τ_p = 22.5 ns, E_t − E_i = 0.10 eV (§5.4, §8) | Confirmed, qualified | arXiv 2012.05989 full text and Table 2. Qualifier: 270 ns is a one-pole fit made with an inflated σ_FCD, not a measured pole. |
| 21 | Green 2008 k(1.31 µm) = 2.81e-10, k(1.26 µm) = 3.61e-9; α = 2.7e-5 and 3.6e-4 cm⁻¹ (headline 7, §2.3) | Confirmed | refractiveindex.info Green-2008 file; α = 4πk/λ re-computed. |
| 22 | Misiakos & Tsamakis n_i fit; Thurmond E_g(T) (§11) | Confirmed (n_i via abstract + secondary); E_g not re-opened | n_i range 78–340 K and (9.7 ± 0.1)e9 at 300 K in the abstract; fit constants via a secondary quotation; values re-computed. Thurmond form is the standard textbook form (not re-opened). |
| 23 | Ge dark current ×1.6 per 10 °C (Colace 2008), E_a ≈ 0.37 eV, ×21 at 105 °C (§10) | Confirmed, qualified | Colace JLT 26, 2211 abstract (printed). Balbi et al., Physica E 2009 (same group; abstract seen via search only): 1.6–1.9× per 10 °C, which allows E_a ≈ 0.5 eV and ×65 at 105 °C. |
| 24 | Kim et al. IEEE PTL 2021 temperature-aware SPICE model exists (§7) | Confirmed | IEEE PTL 33(17), 947–950 (2021), doi:10.1109/LPT.2021.3098760 (imec repository record). |
| 25 | Doped-Si resistivity and TCR from Arora / Masetti (§9) | Arithmetic confirmed; coefficients unverifiable here | Code reproduces the table. The coefficient sets match the standard published forms as the checker knows them, but no primary source was re-opened. ±50 % TCR caveat stands. |

**Net effect on the plant model.** One material number was wrong: the in-plane conductivity of the
silicon device layer was about 20 % too low, because it rested on a mis-cited source. A lumped
heater/ring thermal model built on 69 W/(m·K) would understate lateral heat spreading in the silicon.
Everything else that drives the plant gain checks out: silicon dn/dT(T) at 1310 nm, the thermal-shift
ceiling and its gap to 86 pm/K, the CTE, the free-carrier models and the oxide properties. The
qualifiers say where sources disagree (the size of the dn/dT rise, n_g,Si, the absolute index level,
and the Ge dark-current activation energy).

**Lead not followed up.** Xu, Delâge, Verly, Janz, Wang, Vachon, Ma, Lapointe, Melati and Cheben,
"Empirical model for the temperature dependence of silicon refractive index from O to C band based on
waveguide measurements", Opt. Express 27, 27229 (2019), doi:10.1364/OE.27.027229, fits silicon's
n(λ, T) to SOI echelle filters at 22–80 °C, O band included. The abstract says a constant dn/dT per band
underestimates the channel shift by up to 10 %. The checker could not open the full text. It is the
only waveguide-based O-band dn/dT(T) model found, so it should be added to §2.3.
