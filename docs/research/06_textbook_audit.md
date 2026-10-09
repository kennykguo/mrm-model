# 06 — Textbook model audit: where our MRM plant agrees and disagrees

Audit of the microring-modulator (MRM) plant model in Kenny's interactive textbook
(`/Users/kenny/interactive-textbooks/lightmatter-capstone`, HEAD `02c9c54`). The textbook was read, not edited.

Files read in full: `src/physics/index.ts` (every function and every REF row) and `src/physics/physics.test.ts`.
Chapters read at the sections that define the plant and the control problem: Ch05 (ring, add-drop), Ch06 (modulator,
relOMA), Ch07 (aliasing and FSR), Ch09 (thermo-optics, sections 9.1 to 9.6 and 9.10), Ch10 (self-heating), Ch11
(heater, TCR, thermal network, crosstalk), Ch12 (sensing, polarization), Ch13 (plant gains, PI, lag, feed-forward),
Ch14 (section list), Ch15 (bus effects), Ch17 (the 12-step plant model, the parameter table and the questions), and
Ch18 (the disturbance). Widgets read: `PiLoopWidget`, `DitherLockWidget`, `ArrayLockWidget` and
`HeaterActuatorWidget`. `docs/SOURCES_AND_GENERATIONS.md` and the G26 paper PDF (OFC 2026 M4B.3) were also read.

New numerical evidence in this note was produced with free tools in `/Users/kenny/mrm-model/.venv`:

- `docs/research/scripts/06_thermo_optic_weights.py`: a full-vector FEM mode solver (femwell 0.1.12, order-2
  elements) that computes the thermo-optic weights. It ran under `tools/memguard.py` with a peak of 0.8 to 2.1 GB.
  Refining the mesh from 20 nm to 14 nm in the core changed the 4th decimal only.
- `docs/research/scripts/06_reloma_curvature.py`: the relOMA curvature against swing, linewidth and dλ/dT.
- `docs/research/scripts/06_dac_code_check.py`: what the G26 heater-code range implies for dλ/dT.

Provenance words follow `docs/SOURCES_AND_GENERATIONS.md`: **printed**, **read-off**, **inferred**, **derived**
(computed by us or by the textbook), **assumed**. Verdict words used in this note:

| Verdict | Meaning |
|---|---|
| **correct** | The physics and the algebra are right for what it claims. |
| **simplified-ok** | It leaves something out, but is good enough for control design at this stage. Our model may still add the missing part as an option. |
| **wrong** | The formula, the value or the inference is wrong. The table gives the correction and the evidence. |
| **unverifiable** | No source available to us can confirm or refute it. Our model should carry it as a swept parameter. |

---

## 1. Headline findings

1. **The textbook's "derived 50 pm/K" is wrong, and its "Γ = 1 ceiling of 58 pm/K" is not a ceiling.** The
   formula dλ/dT = (λ/n_g)·Γ·dn_Si/dT is exact only if Γ ≡ ∂n_eff/∂n_Si, the index-perturbation weight. That weight
   can exceed 1.
   - Our FEM gives ∂n_eff/∂n_Si = 1.077 for a 500 × 220 nm Si strip at 1310 nm, and 1.086 for Lightmatter's
     400 × 200 nm strip (derived). [verified with a second, independent solver: the tidy3d 2.12.0 local FDFD mode
     solver gives 1.080 and 1.092]
   - The textbook's 0.85 matches the Poynting **power** fraction in silicon (FEM 0.852 for 500 × 220 nm), which is
     the wrong Γ for this formula.
   - With the right weight, the same chain gives 61.7 to 63.8 pm/K at dn_Si/dT = 1.86 × 10⁻⁴ /K (derived).
     The textbook's formula with the power fraction gives 45.9 to 51.5 pm/K.
2. **The exact identity Σᵢ n_g,i·(∂n_eff/∂nᵢ) = n_g holds.** Here n_g,i is each material's group index. Our FEM
   reproduces it to 1 part in 10⁴ in every geometry. With this identity, dλ/dT|_TO = λ·Σᵢ wᵢ·(dnᵢ/dT)/n_g,i, where
   the weights wᵢ ≥ 0 sum to 1. The rigorous thermo-optic ceiling is therefore **λ·(dn_Si/dT)/n_g,Si**, not
   λ·(dn_Si/dT)/n_Si. It equals 66.3 pm/K at 1.86 × 10⁻⁴ /K. [verified: identity re-derived by the chain rule, and
   reproduced to 1 × 10⁻⁴ by the tidy3d solver. With Frey's n_g,Si = 3.70 in place of Li's 3.677, the ceiling is
   65.9 pm/K.] Chapter 9 sections 9.2 and 9.3 already state this
   ceiling (about 70 pm/K, or 66 pm/K with dispersion). But `REF`, `physics.test.ts`, `SOURCES_AND_GENERATIONS.md`
   §3.1 and Ch17's question Q1 still state the obsolete 58 pm/K "full-confinement" cap.
3. **dn_Si/dT = 1.86 × 10⁻⁴ /K is the wrong input for 1310 nm and 25 to 105 °C.**
   - Frey et al. (2006), Table 4, print 1.94 × 10⁻⁴ /K at 1.3 µm and 295 K (printed). [verified, arXiv:physics/0606168
     Table 4]
   - The coefficient rises by about 14 % per 100 K above 300 K (Souza et al. 2018, supplementary; printed). [verified:
     "close to 14%" is printed. The fitted first-order term is 2.4 × 10⁻⁷ /K² on 1.8 × 10⁻⁴ /K, which is 13.3 % per
     100 K, with a small negative second-order term]
   - That gives about 2.17 × 10⁻⁴ /K at 105 °C (derived). [corrected: Souza's full polynomial gives ×1.104 from 22 to
     105 °C, so 2.14 × 10⁻⁴ /K. The 2.15 × 10⁻⁴ used in the tables is the better value. The effect is −0.3 pm/K]
   - Thermo-optic dλ/dT for real strip and rib MRM cross-sections is then about **64 to 67 pm/K at 22 °C** and
     **71 to 74 pm/K at 105 °C**, plus about 2 to 3 pm/K from expansion (all derived).
   - The rigorous ceiling at 105 °C is about 77 pm/K before expansion (derived).
   - So 86 pm/K is still above the physics, but by about 10 to 30 %, not by a factor of 1.72.
   - The physics range matches Lightmatter's own bench equivalence of 62.5 pm/K (inferred) at room temperature and
     the OCP 80 pm/K (printed) at the hot end.
4. **The "~15 GHz/°C" in G26 is a generic sentence in the introduction, not a measurement.** The paper says:
   "Active stabilization overcomes the intrinsic thermal sensitivity of MRMs (~15 GHz/°C)". [verified against the
   PDF, §1. 15 GHz at 1310 nm is 85.9 pm] It gives no method, no
   reference temperature and no thermometer. Our model must carry dλ/dT as a parameter swept over about
   **60 to 90 pm/K**, temperature dependent, with a **separate sensor node**.
5. **The relOMA curvature does not point to about 50 pm/K, contrary to Ch09 section 9.10.** The published
   k ≈ 0.67 dB/°C² (inferred) constrains only the ratio FWHM/(dλ/dT), to about 6 to 8 K (derived).
   [corrected: the "constrains only the ratio" part is right (exact scaling), but the 6 to 8 K is not. k = 5.45 dB /
   (2.85 K)² is a secant over 2.85 K, about 0.65·FWHM at 86 pm/K and Q 3500, where relOMA is far from quadratic. The
   script matched it to the local second derivative at the optimum instead. With the same large-signal model, matching
   the published drops themselves on the monotonic flank (away from resonance) gives FWHM/(dλ/dT) ≈ 4.2 K from the
   "−8 dB at 2.85 °C" anchor and ≈ 3.5 K from the "−16 dB at 6 °C" anchor (derived). That is 89 to 109 pm/K at
   Q 3500, or Q ≈ 3600 to 4400 at 86 pm/K. So the map is consistent with 86 pm/K, not with 56. The map samples every
   5 °C, so this remains a weak constraint. See the verification log, item V5.]
   - The textbook's own Ch06 inversion says the G26 relOMA needs a 170 to 230 pm swing at Q = 3500 (we get
     169 to 227 pm). [verified: 168.7 to 227.3 pm]
   - With that swing, (Q ≈ 3500, 56 pm/K), (Q ≈ 2600, 74 pm/K) and (Q ≈ 2000, 97 pm/K) all reproduce k ≈ 0.67.
     [corrected: true only for k read as the local curvature at the optimum. Read as the secant it was taken from,
     the matching pairs are about (Q 3500, 89 pm/K), (Q 2620, 119 pm/K) and (Q 2015, 155 pm/K).]
6. **The plant descriptions disagree on how the ambient disturbance enters.** Ch13 and the PiLoop widget filter it
   through the heater's own poles. Ch17 eq. 17.2 and the ArrayLock widget add it with no lag. The physically
   correct choice is neither: each aggressor needs its own transfer function, so that a TEC, an on-die aggressor
   heater, a processor power event and a laser sweep can be told apart.
7. **Missing sensor physics.** The model has no photodiode dark current that rises with temperature. Ge PD dark
   current grows about 1.6 to 1.9 times per 10 °C (printed, Colace et al. 2008). [corrected: Colace et al., JLT 26,
   2211 (2008) print "nearly a factor 1.6 every 10 °C". The 1.6 to 1.9 range is from Balbi, Sorianello, Colace and
   Assanto, Physica E 41, 1086 (2009). IBM RC24490's own 12 → 138 nA over about 61 K is only ×1.49 per 10 K.] On a
   15 µA drop-port side-of-fringe lock at Q = 3500, that is an offset of up to about 6 pm at 105 °C (derived, PD size
   unknown), which is the whole 5 pm budget. [corrected: 3 to 7 pm, from 0.22 to 0.47 µA at 105 °C, depending on
   whether the IBM activation energy or the 1.9×/10 °C rate is used; see S4.] Neighbour-channel leakage into the
   drop-port monitor is also missing: about 0.7 % per neighbour at 400 GHz (derived). [verified: 0.66 % Lorentzian,
   0.78 % Airy at FSR 1.8 THz. But see S6: at FSR = 4.5 × 400 GHz, channels next to the ring's adjacent orders leak
   about 2.7 % each, four times more.]
8. **The notch depth depends on the wrong independent variable.** `notchDepthDbAtHeaterV` makes T_min a function of
   heater voltage. In locked operation the heater moves while λ_r stays at λ_L. The G25 sweep moved both together,
   so the dependence should be modelled on resonance wavelength (coupler dispersion) and ring temperature, not on
   heater drive.
9. **The heater electrical model has no ambient term.** R = R₀(1 + αR_th,H·P) leaves out α·(T_amb − T₀).
   - An 80 K die swing at α = 10⁻³ /K changes R by about 8 %, so P changes by about 8 % at a fixed voltage code.
   - The DAC law is assumed (voltage DAC, 1.6 V full scale). Through the G25 V→λ curve, the published G26 code range
     implies dλ/dT anywhere from 71 to 88 pm/K as the full scale goes from 1.4 to 1.6 V (derived).
   - So the code range cannot independently confirm 86 pm/K.
10. **Many textbook choices are correct and our model should keep them:**
    - the all-pass and add-drop transfer functions and the FWHM formula;
    - the Lorentzian approximation with its error bound;
    - the max-slope bias x = 1/√3 and the 9(Δδ/FWHM)² eye penalty;
    - the bit-averaged (p₁) quasi-static optics;
    - the exact zero-order-hold thermal update;
    - the noise bandwidth of a boxcar window, NEB = 1/(2T_w);
    - the lag accounting T_w/2 + T_d + T_s/2;
    - the drop-port sign flip;
    - the thermal-bistability threshold and the blue-flank recommendation;
    - the half-slot aliasing argument for FSR = 4.5 × 400 GHz.

---

## 2. Candidate (a): the thermo-optic derivation, done carefully

### 2.1 Exact statement

Resonance: n_eff(λ, T)·L(T) = m·λ_m. Differentiating at fixed order m and collecting the wavelength-dispersion term
(textbook eq. 9.2a–9.3, which is **correct**):

    dλ/dT = (λ / n_g) · [ ∂n_eff/∂T |_λ  +  n_eff·α_L  +  (∂n_eff/∂w)·w·α_w + (∂n_eff/∂h)·h·α_h + (stress-optic) ]

The bracket is dominated by the thermo-optic term. For a uniform temperature change in a non-magnetic waveguide whose
cross-section is fixed:

    ∂n_eff/∂T = Σ_i (∂n_eff/∂n_i) · dn_i/dT

### 2.2 The perturbation weight

First-order perturbation of the vector mode, with P = ½∫Re(E×H*)·ẑ dA, gives:

    ∂n_eff/∂n_i = n_i · c · ε₀ · ∫_i |E|² dA  /  (2P)        ≡ Γ_i   (the "index weight")

Γ_i is **not** bounded by 1. In the scalar, weak-guidance limit H ≈ (n_eff/Z₀)E, so 2P ≈ n_eff·c·ε₀·∫|E|²dA and
Γ_i ≈ (n_i/n_eff)·f_i, where f_i is the |E|² fraction (textbook eq. 9.3d). That scalar form is unreliable in a
high-contrast strip, where E_x jumps at the sidewalls and E_z is large.

### 2.3 Exact identity, valid with material dispersion

Only the products k₀·n_i enter the vector wave equation ∇×∇×E = k₀²n²(r)E, so β = F(k₀n₁, k₀n₂, …). The chain rule
then gives:

    n_g ≡ dβ/dk₀ = Σ_i (∂F/∂(k₀n_i))·d(k₀n_i)/dk₀ = Σ_i (∂n_eff/∂n_i)·n_g,i ,   n_g,i = n_i − λ·dn_i/dλ

For non-dispersive materials this reduces to Σ_i n_i·∂n_eff/∂n_i = n_g, the form stated in the task. It is also
equivalent to Γ_i = (n_g/n_i)·u_i, where u_i is the electric-energy fraction. [verified, with a caveat: here n_g
must be the waveguide-only group index Σ_i n_i·Γ_i (3.97 for 500 × 220 nm), not the measured dispersive n_g (4.16).
tidy3d gives u_Si = 0.954, so (3.97/3.50)·0.954 = 1.082 against Γ_Si = 1.080. With the dispersive 4.16 the same
product is 1.134, 5 % too high. Ch09's "(n_g/n_i)·u_i" with n_g = 4.2 has this error.] Our FEM satisfies the dispersive form
to 1 × 10⁻⁴ in all four geometries (table below). Define w_i = n_g,i·Γ_i/n_g. Then w_i ≥ 0, Σw_i = 1, and

    dλ/dT|_TO = λ · Σ_i w_i · (dn_i/dT) / n_g,i   ≤   λ · (dn_Si/dT) / n_g,Si

### 2.4 Which Γ makes each formula right

| Formula | Right if Γ is… | Wrong if Γ is… |
|---|---|---|
| dλ/dT = (λ/n_g)[Γ_Si·dn_Si/dT + Γ_ox·dn_ox/dT] | the **index weight** Γ_i = ∂n_eff/∂n_i (may exceed 1; Γ_Si + Γ_ox ≠ 1) | the Poynting power fraction, or any Γ forced to satisfy Γ_Si + Γ_ox = 1 |
| dλ/dT = λ·Σ u_i·(dn_i/dT)/n_i | u_i = **electric-energy fraction**, non-dispersive materials | any power fraction |
| dλ/dT = λ·Σ w_i·(dn_i/dT)/n_g,i | w_i = n_g,i·Γ_i/n_g (exact, dispersive) | — |
| Textbook eq. 9.4 with Γ = 0.85 and Γ_ox = 1 − Γ | only if 0.85 were the index weight **and** the two weights summed to 1, which cannot both hold for this guide | — |

The textbook (Ch09, "Which Γ is the 0.85?") states that "the 0.85 that this book uses is this rigorous weight ...
about 71 % of the energy in silicon". The FEM contradicts this. For 500 × 220 nm the rigorous weight is 1.077 and the
energy-like weight w_Si is 0.953. The value 0.85 coincides with the **power** fraction in silicon, 0.852.

### 2.5 FEM results at λ = 1310 nm, TE₀ (derived)

Materials: Si from Li's room-temperature Sellmeier (n_Si = 3.5005, n_g,Si = 3.677), the same fit as the textbook's
`nSilicon`. SiO₂ from Malitson (n = 1.4468, n_g = 1.4616). dn_ox/dT = 0.95 × 10⁻⁵ /K. Expansion is not included.

| Geometry | n_eff | n_g | ∂n_eff/∂n_Si | ∂n_eff/∂n_ox | Power fraction in Si | w_Si | dλ/dT at 1.86e-4 | at 1.94e-4 | at 2.15e-4 | Textbook formula with power Γ, at 1.86e-4 |
|---|---|---|---|---|---|---|---|---|---|---|
| Strip 500 × 220 nm | 2.712 | 4.157 | 1.077 | 0.133 | 0.852 | 0.953 | 63.6 pm/K | 66.3 | 73.4 | 50.4 |
| Strip 400 × 200 nm (Lightmatter [SMTAI]) | 2.486 | 4.336 | 1.086 | 0.234 | 0.807 | 0.921 | 61.7 | 64.3 | 71.2 | 45.9 |
| Rib 500 × 220 nm, 90 nm slab | 2.780 | 3.989 | 1.039 | 0.115 | 0.835 | 0.958 | 63.8 | 66.6 | 73.7 | 51.5 |
| Strip 500 × 220 nm, bent R = 5 µm | 2.719 | 4.176 | 1.082 | 0.136 | 0.852 | 0.952 | 63.5 | 66.2 | 73.3 | 50.2 |
| Rigorous ceiling λ·dn_Si/dT/n_g,Si | — | — | — | — | — | 1 | 66.3 | 69.1 | 76.6 | — |

[verified: an independent full-vector FDFD solver (tidy3d 2.12.0 local mode solver, uniform 15 to 20 nm grid, no
sub-pixel averaging; scratch script, not femwell) gives: strip 500 × 220: n_eff 2.717, n_g 4.163, ∂n_eff/∂n_Si 1.080,
∂n_eff/∂n_ox 0.131, power fraction 0.856, dλ/dT 63.6 / 66.3 / 73.5 pm/K. Strip 400 × 200: 2.508, 4.360, 1.092, 0.237,
0.819, 61.7 / 64.3 / 71.2. Rib (infinite 90 nm slab): 2.797, 3.999, 1.045, 0.107, 0.845, 64.0 / 66.7 / 73.9. All dλ/dT
agree with the femwell values to 0.2 pm/K. The bent case was not re-checked.]

A bent rib with a 4 µm slab at R = 5 µm leaks into the slab in our model, so its result is not reported. The
textbook's assumed n_eff = 2.5 and n_g = 4.2 agree with the FEM for the 400 × 200 nm strip to about 3 % (n_g 4.34),
so they are now **derived** rather than purely assumed.

### 2.6 Material inputs

| Quantity | Value | Unit | Provenance | Source |
|---|---|---|---|---|
| dn_Si/dT at 1.3 µm, 295 K | 1.94 × 10⁻⁴ | /K | printed | Frey, Leviton, Madison, Proc. SPIE 6273, 62732J (2006), Table 4 (arXiv:physics/0606168) [verified] |
| dn_Si/dT at 1.5 µm, 295 K | 1.87 × 10⁻⁴ | /K | printed | same table; the textbook's 1.86e-4 is a C-band room-temperature value [verified] |
| n_Si at 1.3 µm, 295 K; dn/dλ | 3.50616; −0.14803 /µm | — | printed | same, Tables 2 and 3. These give n_g,Si ≈ 3.70 [verified: 3.699] |
| First-order temperature term of Si dn/dT | about +14 % of the zeroth order per 100 K above 300 K; second order ≤ 1 % | — | printed | Souza, Grieco, Frateschi, Fainman, Nat. Commun. 9, 665 (2018), supplementary (arXiv:1710.00061) |
| dn_Si/dT at 1.31 µm, 105 °C | ≈ 2.17 × 10⁻⁴ (range 2.1 to 2.3) [corrected: ≈ 2.14 × 10⁻⁴ from Souza's full polynomial, (1.8e-4 + 2.4e-7·ΔT − 2.1e-10·ΔT²), ratio 1.104 from 295 to 378 K; the 2.3 upper end has no source] | /K | derived (extrapolated from the two rows above) | — |
| Si expansion coefficient | 2.6 × 10⁻⁶ at 300 K, about +33 % per 100 K [corrected: Souza prints α₁ = 2.5 × 10⁻⁶ /K at 300 K. The +33 % is only the first-order term. The second-order term is negative (α₃ = −2.3 × 10⁻¹¹ /K³, about −9 % per 100 K), so the net rise is about +27 % per 100 K, and +22 % from 25 to 105 °C] | /K | printed | Souza et al. 2018, supplementary, Fig. 2 and Table 1 |
| Expansion contribution to dλ/dT | about 2.0 to 2.2 at 25 °C and about 2.8 at 105 °C, plus ≲ 1 from cross-section changes [corrected: with Souza's α(T), 1.9 to 2.3 at 25 °C and 2.3 to 2.8 at 105 °C across the three cross-sections; 2.8 is the rib, the top of the range] | pm/K | derived | (λ/n_g)·n_eff·α |

### 2.7 What is left of the 86 pm/K gap

| Temperature | Thermo-optic (strip/rib FEM) | + expansion | Rigorous ceiling + expansion | 86 pm/K over the best estimate |
|---|---|---|---|---|
| 22 °C | 64 to 67 pm/K | 66 to 69 | ≈ 71 | ×1.25 to 1.30 |
| 105 °C | 71 to 74 pm/K | 74 to 77 | ≈ 79 to 80 | ×1.12 to 1.16 |

All derived. [verified: the 22 °C row recomputes exactly. With 2.14 × 10⁻⁴ /K and Souza's α(T), the 105 °C row becomes
thermo-optic 71 to 73, plus expansion 73 to 76, ceiling plus expansion ≈ 79, and ×1.13 to 1.18. The conclusions do
not change.] Conclusions:

- The textbook is right that the thermo-optic effect alone does not reach 86 pm/K at room temperature.
- It is wrong about the size of the gap (×1.72) and about the "impossible from confinement weighting" argument.
- At the hot end the residual is about 10 to 15 %.
- The remaining explanations, in our order of plausibility:
  1. a thermometer that is not at the ring (the textbook's thermal-divider hypothesis, Ch09 section 9.3);
  2. "~15" as a rounded, generic introduction figure;
  3. a referencing temperature near the hot end;
  4. small stress-optic, doping and absorption-change terms. Our estimates for these are each ≲ 2 pm/K: about
     −0.1 pm/K from the built-in voltage, and about +1.5 pm/K from loss rising with T under self-heating
     (derived, rough).

### 2.8 Consequences for our model

- Carry dλ/dT as **dλ/dT(T) = s₀·[1 + β(T − T₀)]**, with s₀ = 60 to 90 pm/K at T₀ = 25 °C and β ≈ 1.4 × 10⁻³ /K.
- Over an 80 K excursion the nonlinearity is about s₀·β·80²/2 ≈ 0.3 nm (derived). That is irrelevant to the local
  loop gain but matters for feed-forward tables and initialization.
- Keep the **heater gain K_G = dλ/dT·R_th in pm/mW** as the primary measured quantity, and derive R_th from it.
- Allow a ring node and a sensor node with a divider ratio.

---

## 3. Audit table of modelling choices

"Location" uses Ch = chapter, § = section, eq. = equation label. Ch17 "steps" are the 12-step plant.

### 3.1 Thermo-optics and temperature

| # | Textbook location | Textbook model | Verdict | Better model | Reasoning and evidence |
|---|---|---|---|---|---|
| T1 | Ch09 eq. 9.1–9.3; `resonanceShiftPerK` | dλ_r/dT = (λ/n_g)(∂n_eff/∂T + n_eff·α_sub); expansion dropped | **correct** (dropping expansion: simplified-ok) | Keep. Add expansion (2 to 3 pm/K, rising about 33 %/100 K [corrected: net about +27 %/100 K; see §2.6]) as an optional term | Re-derived (§2.1). Expansion is 3 to 4 % of the total; it matters only when trying to explain 86 pm/K |
| T2 | Ch09 eq. 9.4; `REF.confinement = 0.85`; ∂n_eff/∂T = Γ·dn_Si/dT + (1−Γ)·dn_ox/dT | Two-material average with weights summing to 1 | **wrong** | Γ_i = ∂n_eff/∂n_i from a mode solver (1.04 to 1.09 for Si; 0.12 to 0.23 for SiO₂), or the w_i form of §2.3 | FEM §2.5. The 0.85 equals the power fraction (0.852), and the true Si weight is 1.077 |
| T3 | `REF.dLambdaDT_derivedPmK = 50`; Ch09 eq. 9.5; test "lands near 50 pm/K" | 50 pm/K derived | **wrong** | 64 to 67 pm/K at 22 °C (thermo-optic plus expansion), 74 to 77 pm/K at 105 °C | §2.5 and §2.7. The 50 comes from using the power Γ |
| T4 | `REF` comment on `dLambdaDT_pmK`; test "cannot reach the device value, even at perfect confinement" (55 < cap < 62); `SOURCES_AND_GENERATIONS.md` §3.1; Ch17 question Q1 ("only about 58 pm/K … 1.48 times bulk") | Γ = 1 gives a hard cap of 58 pm/K | **wrong** | Ceiling λ·(dn_Si/dT)/n_g,Si = 66 pm/K at 1.86e-4, 69 at 1.94e-4, about 77 at 105 °C | Γ is not bounded by 1. An ordinary 500 × 220 nm strip already gives 63.6 pm/K at 1.86e-4. Ch09 §9.2–9.3 already contain the correct ceiling, so these four places are stale |
| T5 | Ch09 eq. 9.4c and Fig. 9.1a, "rigorous ceiling ≈ 70 pm/K" and "66 with dispersion" | λ·dn_Si/dT/n_Si | **correct** in form; **simplified** in number | Use n_g,Si (exact, §2.3) and a dn/dT for 1310 nm that depends on T | The textbook's proof of the identity is scalar and non-dispersive. The exact chain-rule proof (§2.3) gives the dispersive version. With 1.86e-4 it understates the ceiling at 1310 nm and at high T |
| T6 | `REF.dnSi_dT = 1.86e-4`, constant | Constant, room temperature, C-band value | **wrong** for 1310 nm over 0–105 °C | dn/dT(λ, T): 1.94e-4 at 1.3 µm and 295 K (Frey 2006), about +14 %/100 K | Printed sources in §2.6 |
| T7 | `REF.dnSiO2_dT = 1e-5` | Constant | **simplified-ok** | 0.95e-5 to 3e-5 depending on the oxide (Souza 2018 uses 3e-5) | The oxide weight is 0.12 to 0.23, so the oxide contributes only 0.4 to 2 pm/K |
| T8 | `detuningPmFromK`, `kelvinFromDetuningPm`; Ch17 eq. 17.3 | Linear λ_r = λ_r0 + (dλ/dT)(T − T₀) | **simplified-ok** for the local loop, **simplified** for feed-forward and initialization | ∫ dλ/dT(T) dT, quadratic in T | About 0.3 nm of curvature over 80 K (derived) |
| T9 | `REF.dLambdaDT_pmK = 86`, design value; Ch09 §9.3 | "~15 GHz/°C printed for Lightmatter's own MRM" | **unverifiable** | Parameter 60 to 90 pm/K, swept; default about 70 pm/K (derived physics mid-range) | The G26 text is a generic statement in the introduction ("the intrinsic thermal sensitivity of MRMs (~15 GHz/°C)"). It gives no method, no reference temperature and no thermometer. The book is right to design conservatively at the high end and to sweep |
| T10 | `REF.dLambdaDT_benchPmK = 62.5` | 50 nm/s ÷ 800 °C/s, inferred | **correct** as an inference, and **consistent** with the corrected physics at room temperature | Keep as the low anchor | Agrees with the FEM result of 64 to 67 pm/K within the rounding of "800" |
| T11 | Ch09 checkpoint after §9.1: the colleague's "5.3 × 10⁻⁵ × 1310 nm ≈ 70 pm/K … which two things are wrong?" | Calls the 70 pm/K estimate "wrong twice over" | **wrong** (pedagogy) | λ·(dn_Si/dT)/n_Si is exactly the all-energy-in-silicon limit of the correct formula | Contradicts Ch09's own eq. 9.4c a few pages later |
| T12 | Ch09 §9.10 deeper box: the relOMA curvature implies "a sensitivity near 49 pm/K … the third time … evidence pointing at 50–63 pm/K" | Lorentzian small-signal k = 4.343·9·(dλ/dT/FWHM)² compared with k = 0.67 | **wrong** inference | The curvature constrains FWHM/(dλ/dT) ≈ 6 to 8 K; it does not constrain dλ/dT alone [corrected: the ratio-only point stands, but k = 0.67 is a secant over 2.85 K (≈ 0.65·FWHM), not a local curvature. Matching the published drops directly gives FWHM/(dλ/dT) ≈ 3.5 to 4.2 K; see §4.3 and V5] | With the 170 to 230 pm swing that the textbook's own Ch06 inversion requires: (Q 3500, 56 pm/K), (Q 2600, 74 pm/K) and (Q 2000, 97 pm/K) all give k ≈ 0.67. See §4.3 and the script [corrected: these pairs match the local curvature only. Matched as the secant: about (Q 3500, 89 pm/K), (Q 2620, 119), (Q 2015, 155)] |
| T13 | Ch09 §9.3 thermal-divider candidate; Ch17 eq. 17.11 interface allows a thermistor word | Ring temperature versus sensor temperature, divider ratio | **correct** as a hypothesis | Plant with distinct ring and sensor thermal nodes and a configurable divider and lag | This is the most plausible remaining explanation of the gap at room temperature (§2.7) |

### 3.2 Ring optics

| # | Textbook location | Textbook model | Verdict | Better model | Reasoning and evidence |
|---|---|---|---|---|---|
| O1 | Ch05 eq. 5.5, 5.7, 5.10, 5.21, 5.24; `ringAllPassT`, `fwhmNm`, `ringTmin`, `tForQ`, `aFromLoss` | All-pass Airy transfer, T_min = ((a−t)/(1−at))², FWHM = (1−ta)λ²/(πn_gL√(ta)), 1/Q = 1/Q_int + 1/Q_c | **correct** | Keep | Standard (Bogaerts 2012 form). Re-derived |
| O2 | Ch05 eq. 5.28–5.31; `ringAddDrop` | Add-drop through and drop ports, drop peak κ₁²κ₂²a/(1−t₁t₂a)² | **correct** | Keep | Re-derived. The through-port numerator t₁ − t₂ae^{−jφ} is right |
| O3 | Ch17 eq. 17.5, Lorentzian with error ε ≈ x²π²/(12𝓕²) | Lorentzian in the inner loop, exact Airy for FSR-wide sweeps | **simplified-ok** | Use exact Airy (or temporal CMT) everywhere. It is cheap in Python | The error bound is correct (re-derived). Periodicity must be kept for initialization |
| O4 | Ch17 steps 5–6; Ch12 eq. 12.15 | G26 modulator handled as an all-pass notch (T_min) plus an independent drop peak (T_peak) | **simplified** | One add-drop ring with (a, t₁, t₂), or CMT (κ_e1, κ_e2, κ_i). T_min, T_peak, the linewidth and the absorbed fraction all follow from it and conserve energy | G26 printed: "monitor photocurrent at the MRM drop port". The monitor tap (κ₂) loads Q and changes T_min. The textbook treats the two ports as independent knobs |
| O5 | `notchDepthDbAtHeaterV`; Ch17 warning box "T_min is a function of the actuator" | T_min interpolated against **heater voltage** from the G25 sweep (28–30 dB at low drive, 17 dB at 1.5 V) | **wrong independent variable** (the data are fine) | T_min from κ(λ_r) (coupler dispersion) and α(T, V_bias). Fit both to the G25 sweep, in which λ_r and T moved together | In locked operation the heater code swings about 30,000 codes while λ_r stays at λ_L. The textbook model would then predict sensor-gain and extinction changes that a λ-dependent coupler would not. The sweep cannot separate the two causes, so carry both as hypotheses |
| O6 | `REF.Q = 3500`; `REF.qFromLinewidth = [3900, 4200]` | G25 Q from the talk | **unverifiable** for G26 | Parameter Q_L from 2000 to 8000, as the textbook says. Note that the G26 EO bandwidth (66.8 GHz, printed in slides) suggests a different design | The relOMA curvature allows Q ≈ 2000 to 3500 (§4.3) [corrected: that range comes from the local-curvature reading. Matched as a secant, the map gives Q ≈ 3600 to 4400 at 86 pm/K, or Q ≈ 5000 to 6100 at 62.5 pm/K. It still fixes only FWHM/(dλ/dT)] |
| O7 | `REF.neff = 2.5`, `REF.ng = 4.2`, assumed | — | **simplified-ok** (now supported) | FEM: 400 × 200 nm strip n_eff 2.486, n_g 4.336; 500 × 220 nm strip 2.712, 4.157; rib 500 × 220 nm / 90 nm slab 2.780, 3.989 (derived) | §2.5 |
| O8 | `REF.fsrTHz = 1.8` (G25 design target); `fsrMeasuredBoundNm = 14.1`; wafer maps imply 8.8 nm (Ch00); `radiusUm = 6.3` derived | FSR 10.3 nm used in arithmetic | **unverifiable** | FSR as a parameter from 8.7 to 16 nm. Note one consistent cluster (next column) | The "15 µm diameter" (Lightmatter slides, via REF) gives FSR 8.7 nm at n_g 4.2. The wafer maps imply 8.8 nm, and the swept device's efficiency at 1.195 V is 0.348 nm/mW, which matches 8.8 nm / 25.45 mW (derived). The ">14.1 nm" bound conflicts with all three |
| O9 | Ch07 eq. 7.10–7.11, aliasing at ρ = FSR/Δf_ch = 4.5 | The repeat lands half a slot off-grid, so the FSR check "fails" but nothing collides | **correct** | Keep. In a model, compute all orders | `SOURCES_AND_GENERATIONS.md` §3.9 says "fails" without the half-slot qualification. Ch07 has it right [verified, with a note: "nothing collides" holds for the data path. But the half-slot alias is only 200 GHz from two TX channels, so the drop-port monitor sees about 2.7 % of each (see S6). It also lands exactly on the G26 RX grid, which is offset by 200 GHz (printed). So the alias matters for monitor crosstalk and for BiDi reflections] |
| O10 | Ch17 steps 1–6 (one laser line per ring, P_in fixed); Ch15 §4 (bus effects, discussed qualitatively) | Each ring sees only its own line | **simplified** | Cascade on the bus: every ring's through response at every wavelength, so that P_in,i = P_laser,i·Π_upstream T_j(λ_i). Neighbour leakage enters the drop-port monitors | Matters for initialization (a sweeping ring crosses other lines) and for monitor crosstalk (§3.5 S6). `REF.offResPassLossDb = 0.2` is the static version |
| O11 | Not modelled | CW/CCW backscatter splitting, coupler dispersion inside one linewidth, ring-to-bus reflections | **simplified-ok** for the TX loop at Q ≈ 3500 | Optional | Backreflection matters for the BiDi receiver (G26 printed: "spectral crosstalk from on-chip reflections and backscattering"), not for the TX lock |

### 3.3 Modulation and operating point

| # | Textbook location | Textbook model | Verdict | Better model | Reasoning and evidence |
|---|---|---|---|---|---|
| M1 | `modulatorLevels`; Ch06; Ch17 eq. 17.6a | The data shift a fixed Lorentzian rigidly by ±s/2 | **simplified-ok** for the thermal loop | Bias-dependent Δn_eff and Δα (Soref–Bennett) give λ_r(V) and a(V), so T_min and Q vary between levels. Optional dynamic CMT for eye shape | The loop sees only bit-averaged power (M4). Modulation loss is small next to the 1 − ta ≈ 0.115 of a Q = 3500 ring (derived) |
| M2 | `MOD_SHIFT_PM` [0, 56.4, 66.8, 84.4] pm at [0, 0.8, 1.2, 1.6] V; `modShiftPm`, `modEfficiencyPmPerV` | Piecewise-linear read-off; local efficiency 70, then 26, then 44 pm/V | **simplified** | Fit a monotone depletion law Δλ = A[(V+V_bi)^m − V_bi^m] with m ≈ 1/2 to 1/3, and carry the read-off scatter | A depletion junction's efficiency cannot rise again with bias (the textbook says so). The 26 → 44 rise is read-off noise |
| M3 | Ch17 eq. 17.3: + η_EO·V_i | Linear DC-bias shift | **simplified** | Use M2's nonlinear shift for the DC bias over the G26 axis of 1.0 to 3.75 V | — |
| M4 | Ch17 eq. 17.6b: T̄ = p₁T₁ + (1−p₁)T₀; Ch12 eq. 12.20 | Bit-averaged quasi-static optics | **correct** | Keep | τ_th ≫ bit period (20 ps) ≫ photon lifetime (2.4 ps at Q 3500). Verified |
| M5 | Ch09 eq. 9.7d, 9.7f; Ch17 x_opt = 1/√3 | Max-slope bias, and ΔOMA/OMA = −9(Δδ/FWHM)² | **correct** (small signal) | Use the numeric optimum for the actual swing | Re-derived: g″/(2g) = −9/4 at x = 1/√3. At the swing the G26 relOMA implies (≈ 190 pm at Q 3500) the optimum moves to about 130 to 146 pm (derived) |
| M6 | `REF.swingVpp = 1.3` (simulated), `swingMeasuredVpp = 0.9`; 65 pm swing everywhere | G25 swing | **unverifiable** for G26. The textbook itself shows it cannot produce the G26 relOMA (Ch06, "Does this chapter's model reproduce the published number?") | Swing as a parameter. At Q 3500, relOMA from −3.0 to −2.1 dB needs 169 to 227 pm (derived) | Our numbers agree with the textbook's inversion in direction and size |

### 3.4 Self-heating

| # | Textbook location | Textbook model | Verdict | Better model | Reasoning and evidence |
|---|---|---|---|---|---|
| H1 | Ch17 eq. 17.7; Ch10 | P_abs = η_abs·P_in·(1 − T̄ − T̄_drop), η_abs = 0.3 assumed (sweep 0.1 to 0.6) | **unverifiable**; the value is probably low | Get η_abs from the loss budget: η_abs = α_FCA/(α_FCA + α_scatter + …). Sweep 0.1 to 0.9 | G25 notch depths of 28 to 30 dB mean near-critical coupling, so Q_int ≈ 2·Q_L ≈ 7000. That implies α ≈ 29 /cm (≈ 125 dB/cm), far above passive strip loss (1 to 3 dB/cm), so free-carrier absorption in the doped junction dominates and becomes heat. η_abs of 0.6 to 0.9 is plausible (derived; assumes FCA-dominated loss) [verified: α = 2πn_g/(Q_int·λ) = 28.8 /cm = 125 dB/cm. Qualified: FCA dominance is not shown. Soref–Bennett at 1.3 µm (Δα = 6.0e-18·N + 4.0e-18·P) with an assumed 1 to 3 × 10¹⁸ cm⁻³ junction gives only about 4 to 13 /cm, 15 to 45 % of 29 /cm. The rest could be bend or sidewall scattering in a small ring, which does not heat the ring. Keep the 0.1 to 0.9 sweep; treat 0.6 to 0.9 as one hypothesis] |
| H2 | Ch17 eq. 17.9a: the same **K** for P_H and P_abs | The absorbed power heats the ring through the heater-to-ring R_th | **simplified** | Separate self-resistance R_rr ≥ R_rh (reciprocity), unless the heater sits in the ring | Heat absorbed in the ring is deposited at the ring node. The heater-to-ring transfer resistance is at most the ring's self-resistance. So the model underestimates self-heating per absorbed mW for an external (metal) heater, and is exact only for an in-ring heater |
| H3 | Ch10 eq. 10.8, A > 0.77·FWHM; blue-flank lock recommended | Thermal bistability threshold; δ < 0 is thermally self-stable | **correct** | Keep | Re-derived: the maximum slope of a Lorentzian peak is (3√3/8)(2/w). Laser blue of the hot resonance gives negative thermal feedback |
| H4 | Ch10 §5 (free carriers, self-pulsation) | Fast blue shift, mentioned and bounded | **simplified-ok** | Optional TPA/FCA terms | At milliwatt O-band powers in a Q ≈ 3500 ring these are second order |

### 3.5 Thermal network

| # | Textbook location | Textbook model | Verdict | Better model | Reasoning and evidence |
|---|---|---|---|---|---|
| N1 | Ch13 eq. 13.1–13.2 (one pole); Ch11 eq. 11.17–11.21 (heat equation to RC) | One-pole R_th/(1 + sτ_th) for design | **simplified-ok** for PI design | Keep as the design model only | — |
| N2 | Ch11 eq. 11.22–11.23; Ch17 eq. 17.9b; `stepThermal`; `REF.tauThUs = 10`, `tauSlowUs = 300`, `slowShare = 0.25` | Two-pole Foster network, values from a 2013 review | Structure **simplified-ok**; values **unverifiable** (assumed) | A 3- to 5-term Foster or Cauer network fitted to a FEM transient of the real stack (femwell has `thermal_transient`), then to a heater-step measurement. Sweep τ_f from 1 to 30 µs and τ_s from 0.1 to 3 ms | No Lightmatter source states a time constant (the textbook says so). In-ring doped-Si heaters and metal heaters differ by about ×10 in τ. The textbook's volume check (C_f = 2.84 nJ/K, about a 12 µm cube) is a good sanity test |
| N3 | Ch17 eq. 17.9b, exact ZOH α = 1 − e^{−Δt/τ} | — | **correct** | Keep, or use the matrix exponential for a full RC network | Re-derived |
| N4 | `crosstalkMatrix`; `REF.crosstalkNearest = 0.1`, `crosstalkNext = 0.03`; Ch11 eq. 11.24 | Banded symmetric Toeplitz **K** | Values **unverifiable** (assumed); structure **simplified-ok** | Nodal RC network: edge rings, a shared slow node, and the photonic-to-electronic die coupling | The two-node formula (Ch11, K₁₁ = R(R_c + R)/(R_c + 2R)) is **correct** (re-derived) |
| N5 | `stepThermal` applies the same 0.75/0.25 split and the same τ to every K_ij | Crosstalk has the same dynamics as the self term | **wrong** physically (the textbook flags it in Ch17's volume box) | Per-element transfer functions K_ij(s): delay plus extra poles, weighted towards the slow pole for i ≠ j | Diffusion to a neighbour 10 to 30 µm away is slower and lags. A decoupler built on equal dynamics is optimistic |
| N6 | Ch13 eq. 13.20 and the PiLoop widget (ambient through the heater poles) **vs** Ch17 eq. 17.2 and the ArrayLock widget (ambient added with no lag) | Two incompatible ambient injections | **simplified**, and **internally inconsistent** | Separate disturbance paths: (i) package/TEC uniform T (seconds); (ii) on-die aggressor with its own G_agg(s), with diffusion delay; (iii) XPU power events (10 to 120 °C/s within about 30 ms, printed secondary [MRVL]); (iv) laser sweep, optical only, no thermal path | G26 printed: "on-chip resistive heater integrated adjacent to a MRM … 808 °C/s within the MRM". Its heat reaches the ring through a different path from the ring's own heater |
| N7 | `REF.RthKPerMw = 4.7` = 0.405 nm/mW ÷ 86 pm/K (derived) | R_th from a derived efficiency and the printed sensitivity | **simplified**, and its uncertainty is understated | Parameterize **K_G = η_H [pm/mW]** (the measured gain) and dλ/dT. Then R_th = K_G/(dλ/dT) | η_H = FSR/25.45 mW depends on the unresolved FSR (8.8 to 14.1 nm gives 0.35 to 0.55 nm/mW), so R_th spans about 4.0 to 8.9 K/mW (derived). The book's claim that anchoring on the wafer maps "removes the dependence on the disputed FSR" holds in FSR units, not in pm/mW |
| N8 | Linear K (R_th independent of temperature) | — | **simplified** | Gain rising with temperature: k_Si(T) falls about 30 % from 300 to 400 K, so R_th rises; dn/dT also rises. The heater gain in pm/mW rises by an estimated 10 to 25 % from 25 to 105 °C (derived, rough; needs FEM) [qualified: the −30 % is right for bulk Si. But an SOI ring's R_th is usually set mostly by the buried and cladding oxide. Amorphous SiO₂ conductivity is flat to slightly rising with T (bulk data rise; thin-film MD shows little change from −55 to 150 °C, "Thermal conductivity of amorphous SiO2 thin film: a molecular dynamics study", Sci. Rep. 2018, doi:10.1038/s41598-018-28925-6, abstract only, via search). So R_th may change by only a few per cent, in either direction. dn/dT alone gives about +10 %, which is the firm lower part of the range] | Adds a loop-gain variation that the textbook attributes only to the DAC law and to TCR |

### 3.6 Heater electrical and DAC

| # | Textbook location | Textbook model | Verdict | Better model | Reasoning and evidence |
|---|---|---|---|---|---|
| E1 | Ch11 eq. 11.14–11.16; `heaterPowerFromDacCode`; `REF.heaterSelfHeatPerMw = 0.00904`; `HEATER_OHM_COLD` ≈ 45.4 Ω | R = R₀(1 + α·R_th,H·P_H); P_H(1 + yP_H) = V²/R₀; y fitted to the 22 % Δλ/V² droop | **simplified-ok** as a fit; physically **incomplete** | R = R₀[1 + α(T_amb − T₀) + α·R_th,H·P_H] with α, R_th,H and T_amb explicit. Electro-thermal feedback is negative for voltage drive and positive for current drive | (1) There is no ambient term: an 80 K die swing at α = 10⁻³ /K changes R and P by about 8 % at a fixed V code. (2) The measured droop is the net of TCR droop and the gain rise from dn/dT(T) and k(T), so y is an effective parameter. (3) α and R_th,H cannot be identified separately; R_th,H ≥ R_th bounds α ≤ y/R_th ≈ 1.5 to 1.9 × 10⁻³ /K (derived). The 1e-3 /K value is assumed |
| E2 | Ch11 V_FS = 1.6 V voltage DAC; Ch17 eq. 17.10a P_FS ≈ 37 mW, V_FS ≈ 1.44 V (derived); PWM/ΣΔ alternative | DAC law assumed | **unverifiable** | Configurable DAC law (voltage, current, power-linear PWM/ΣΔ), V_FS, the rate limit, and zero-order-hold glitches | Through the G25 V→λ curve, the G26 code range (19,500 to 49,300) over 83 °C implies dλ/dT = 70.9, 74.2, 79.5 and 88.4 pm/K for V_FS = 1.40, 1.44, 1.50 and 1.60 V (derived, script). So the code range does not independently confirm 86 pm/K |
| E3 | `REF.heaterOhm = 55.9`, `heaterVFsr = 1.195`, `heaterImA = 21.4`, `heaterPFsrMw = 25.45` | Wafer-map median, inferred from printed per-die values | **correct** as inferred | Keep, with a per-ring spread | — |
| E4 | `HEATER_V_LAMBDA` (G25 swept device) used as the actuator curve | 13.0 nm at 1.5 V; 8.88 nm at 1.195 V | **simplified-ok**, but it is a different device from the wafer median | Per-ring η_H drawn from the wafer distribution | At 1.195 V and 55.9 Ω the swept device gives 0.348 nm/mW, against 0.405 for the wafer median at an assumed 10.3 nm FSR (derived). The textbook notes the two are not the same device |
| E5 | Ch17 eq. 17.10b, clamp to [0, P_max] and rate limit ρ | Heater only heats; slew limit | **correct** | Keep | — |
| E6 | `REF.heaterDacSlewCodesPerMs = 460` (inferred); about 718 to 826 codes/ms at the peak (derived) | — | **unverifiable**, reasonable | Express slew in mW/s and pm/s as well | — |

### 3.7 Sensing and monitor

| # | Textbook location | Textbook model | Verdict | Better model | Reasoning and evidence |
|---|---|---|---|---|---|
| S1 | Ch06 §"The monitor's sign"; Ch17 step 6; `REF.monitorCurrentUa = 15` (read-off) | Drop-port monitor, peaks on resonance, sign flipped against the through port | **correct** | Keep; the port is a configuration setting | G26 printed: "The monitor photocurrent at the MRM drop port (green) remained stable" |
| S2 | Ch17 eq. 17.11a–b; Ch12 eq. 12.30–12.32 | Shot + TIA noise, boxcar NEB = 1/(2T_w), 16-bit ADC, I_FS = 20 µA (assumed), FIFO delay | **correct** (structure) | Keep | Re-derived NEB |
| S3 | `REF.afeNoisePaRtHz = 6.4` used for the monitor | The data-path AFE noise (simulated) applied to the monitor TIA | **unverifiable**, probably pessimistic | A monitor TIA with R_f ≈ 100 kΩ has √(4kT/R_f) ≈ 0.4 pA/√Hz | It does not matter: the loop is not noise-limited, which the textbook also concludes |
| S4 | Not modelled | Photodiode dark current, rising with temperature | **missing** | I_d(T) = I_d0·r^((T − T₀)/10 K), r = 1.6 to 1.9, as an additive offset | Colace et al., JLT 2008 (printed): ×1.6 to 1.9 per 10 °C. IBM RC24490 (printed): 12 nA at room temperature to 138 nA at 86 °C. Extrapolated to 105 °C: about 0.4 µA, which is 2.7 % of 15 µA. At the drop-port fractional slope √3/FWHM = 0.46 %/pm (Q 3500), that is a 5.8 pm lock offset (derived; the PD size is unknown) [corrected: Colace, Balbi, Sorianello, Assanto, JLT 26, 2211 (2008) print "nearly a factor 1.6 every 10 °C". The 1.6 to 1.9 range is Balbi et al., Physica E 41, 1086 (2009). IBM's own data (10 × 10 µm Ge-on-SOI, −0.5 V; E_a 0.30 to 0.35 eV, from search summaries) extrapolate to 0.22 to 0.24 µA at 105 °C. Its 12 → 138 nA slope gives 0.29 µA, ×1.6/10 °C gives 0.34 µA and ×1.9/10 °C gives 0.47 µA. That is 1.5 to 3.1 % of 15 µA, so a 3.2 to 6.7 pm offset. The slope √3/FWHM = 0.463 %/pm is verified] |
| S5 | Fixed responsivity 0.9 A/W (printed, talk) | — | **simplified-ok** | Optional weak T-dependence | At 1310 nm, Ge is well above its absorption edge |
| S6 | Not modelled in Ch17 | Neighbour channels leaking into a ring's drop-port monitor | **missing** | Bus cascade (O10). Leakage per neighbour = 1/(1 + (2Δf/FWHM)²) = 0.0067 (−21.7 dB) at 400 GHz and Q 3500 (FWHM 65.5 GHz); about 1.3 % with two neighbours (derived) | Changes of about 1 % in neighbour power or detuning give about 2 to 3 pm of lock error on a side-of-fringe lock (derived) [corrected: a 1 % change in one neighbour's power moves the lock by only about 0.02 pm. It is a change of 1 % of the ring's own monitor signal that gives 2.2 pm. The static leakage of two nearest neighbours (1.3 % of peak, 1.8 % of the locked signal) is a 3.8 pm offset if not calibrated out. Missing and larger: at FSR = 4.5 × 400 GHz, the ring's m ± 1 orders sit 200 GHz from channels i ± 4 and i ± 5. Each of those leaks 2.7 % (−15.7 dB, Airy), so two of them give about 16 pm. Those orders also fall exactly on the 200 GHz-offset counter-propagating grid. The model must compute every order (O9)] |
| S7 | Ch12 polarization section; Ch17 parameter row "normalize the monitor, or the loop will chase polarization" | The 7.7 dB swing (read-off) is a receive-side single-polarization monitor | **correct** in Ch12; the Ch17 row is ambiguous | The TX MRM monitor is not exposed to the link scrambler | G26 printed: "polarization dependent monitoring photodetectors at the input of the receive path". The Ch12 claim that the TX feed is polarization-maintaining fibre is **unverifiable** from the G26 paper, which only says the 8 wavelengths are "multiplexed to a single output fiber" |
| S8 | Ch12 eq. 12.38 and the reference-tap ratio (Ch12 §3) | Normalize the monitor against a reference tap | **correct** | Model the reference tap with its own responsivity, coupling and dark current | A 2.5 % laser power drift moves a side-of-fringe lock by about 5.4 pm at Q 3500 (derived) |
| S9 | Ch17 eq. 17.13 (T_lag = T_w/2 + T_d + T_s/2) **vs** Ch13 eq. 13.23 (T_d defined as the total of all four) | Lag accounting | **correct**; **notation inconsistent** | One symbol for the total lag and separate symbols for each part | The PiLoop widget uses T_d + T_s/2 with no window term |

### 3.8 Lock point and control

| # | Textbook location | Textbook model | Verdict | Better model | Reasoning and evidence |
|---|---|---|---|---|---|
| C1 | Ch17 steps 6 and 12, Ch14 §4: side-of-fringe (setpoint) lock on mean drop-port power at the max-OMA bias | Error = y_ref − y | **simplified-ok**, as the most likely G26 scheme | Keep as one of several selectable error signals: side-of-fringe, dither or extremum seeking, bit statistics | Which lock G26 uses is **unverifiable**. "Monitor photocurrent remained stable" fits a setpoint lock but does not prove one. The side-of-fringe point is biased by P_in (S8), dark current (S4), neighbour leakage (S6), p₁, and the smearing of the mean monitor curve by the swing (C2) |
| C2 | Ch17 sensor gain from the static Lorentzian slope at x = 1/√3 (0.46 %/pm at the drop port) | Small-swing slope | **simplified** | Compute the slope of the bit-averaged monitor curve at the actual swing and bias | At swings of 170 to 230 pm (§3.3 M6) the averaged curve is visibly broadened |
| C3 | `piStep` (clamping anti-windup); Ch13 PI with K_i/K_p = 1/τ_th | — | **correct** (controller, not plant) | — | — |
| C4 | Ch13 eq. 13.21, e_ss = ṙ/(K_iK_G); Ch11 eq. 11.27 | Ramp error of a type-1 loop | **correct** | — | Re-derived |
| C5 | Ch17 eq. 17.13; `SOURCES_AND_GENERATIONS.md` §4 "loop lag < 29 µs" | δ_floor = (dλ/dT)·(dT/dt)·T_lag; 29 µs at 86 pm/K and 2000 °C/s | Formula **correct**; numbers depend on T9 | Budget for 5 pm: 38 µs at 66 pm/K and 2000 °C/s; 34 µs at 73 pm/K; 72 to 99 µs at the peer-reviewed 808 °C/s (derived) [verified: 37.9, 34.2, and 72.0 (86 pm/K) to 99.0 (62.5 pm/K) µs] | — |
| C6 | `DitherLockWidget` | Thermal dither attenuated by \|1/(1 + jωτ)\|; phase ignored | **simplified-ok** | Include the phase (3.6° at 1 kHz with τ = 10 µs) | — |
| C7 | `ArrayLockWidget` | The controller reads the detuning δ plus Gaussian noise | **simplified** for teaching. It breaks Ch17's own interface rule ("not δ") and hides the Lorentzian capture-range limits | Read the monitor counts | — |
| C8 | `PiLoopWidget` | Through-port monitor (not the G26 drop port); ambient through the heater poles | **simplified** | — | See N6 |

### 3.9 Disturbances

| # | Textbook location | Textbook model | Verdict | Better model | Reasoning and evidence |
|---|---|---|---|---|---|
| D1 | Ch17 eq. 17.8 (step plus capped ramp); Ch18 sawtooth | — | **simplified-ok** | Add an asymmetric sawtooth, TEC sweeps over hours, XPU events and laser sweeps as separate scenario types (N6) | The textbook already warns against merging the three aggressor methods |
| D2 | `REF.transientCPerS = 2000` (slides), `transientPeerReviewedCPerS = 808` (paper) | °C/s measured "at the ring" | **unverifiable** (method of measurement) | Specify the disturbance as a detuning slew in pm/s, or as an aggressor power, in addition to °C/s | If the in-MRM °C/s was inferred from the resonance shift through ~15 GHz/°C, then the °C/s and dλ/dT are not independent, and only their product (pm/s) is physical. Note also that 808 °C/s × 100 ms ≈ 81 K ≈ the printed 25 → >105 °C excursion. With both ramps inside one 100 ms period, the instantaneous edges must be faster than 808 °C/s, which agrees with the slides' >2000 °C/s (derived) [verified, sharpened: the paper prints "ramped up and down between 25 °C and >105 °C with a period of 100ms". If one period holds both ramps, the mean slew over the period is at least 2 × 80 K / 0.1 s = 1600 °C/s. So 808 °C/s cannot describe both edges, and at least one edge is at least about 1600 °C/s on average. If "period" means one ramp, 808 °C/s ≈ 80 K / 0.1 s is simply the ramp average. The paper does not say which] |
| D3 | Ch18 Def "design-for-test stimulus" cites [LM26-P] for ">2000 °C/s between about 30 and 113 °C" | — | **citation imprecise** | Cite [LM26-S] for 2000 and 30–113, and [LM26-P] for 808, 25 → >105 °C and 100 ms | G26 paper text (printed) |
| D4 | Ch17 eq. 17.1, laser line = grid + global + per-line + drift | — | **simplified-ok** | Add laser power drift and steps (the failover case from Ch15), and optionally RIN. ±20 GHz accuracy is printed in G26 | — |

### 3.10 Metrics and acceptance (relOMA)

| # | Textbook location | Textbook model | Verdict | Better model | Reasoning and evidence |
|---|---|---|---|---|---|
| R1 | Ch06 relOMA definition: 10·log₁₀(OMA/1 mW) with 1 mW into the modulator ([HC] footnote) | — | **unverifiable** (footnote not available to us) | Keep. The model must report OMA/P_in at the modulator input plane | The textbook's own model gives −6.6 dB against the published −2.1 to −3.0 dB, and the textbook flags this (Ch06). Either the reference plane or the G25 swing and linewidth do not describe G26 |
| R2 | `relOmaDb`, stabilized | Linear in bias: −2.1 dB at 1.0 V to −3.0 dB at 3.1 V and beyond; no T dependence | **simplified-ok** as an acceptance **target** | Never use it as plant output; compute relOMA from the spectral model | — |
| R3 | `relOmaDb`, unstabilized; anchors (0, peak), (2.85 °C, −8), (6, −16), (70, −54), log-interpolated; peak at 70 °C equal to the loop-closed level | Empirical | **simplified-ok** as a target; the far anchors are extrapolations | Let the loop-open shape emerge from the spectral model plus dλ/dT(T) | — |
| R4 | `REF.omaPenaltyDbPerC = 0.67` (0.5 to 0.8), peak-referenced | k from the "−8 dB" contour at 2.85 °C, 5.45 dB below the peak | **correct** as an inference, given the shared-colour-scale reading | Use it to constrain FWHM/(dλ/dT) ≈ 6.1 to 7.8 K (derived) [corrected: 6.1 to 7.8 K results from treating the 2.85 °C secant as a curvature at the optimum. Fit the published drops (5.45 dB at 2.85 °C, 13.45 dB at 6 °C) with the large-signal model instead: ≈ 3.5 to 4.2 K on the monotonic flank. The 5 °C map sampling makes either number soft] | Dividing pm by 86 pm/K to get dB is self-consistent only if the map's temperature axis is the same thermometer as the ~15 GHz/°C |
| R5 | Ch09 eq. 9.7f / 9.11a, k_model = 4.343·9·(dλ/dT/FWHM)² (max-slope bias) **vs** Ch18 K_LORENTZIAN_PEAK = 4.343·(2dλ/dT/FWHM)² (about the line peak) | Two different expansions | Both **correct** as limits; **which one applies depends on the swing** | Compute k from the large-signal OMA (script). At Q 3500 and 86 pm/K, k = 1.99 dB/K² at the 65 pm swing, falling to 1.50 to 1.68 at 170 to 230 pm | The small-swing limit is 2.07; the about-the-peak limit is 0.92 |
| R6 | Ch09 eq. 9.7c–9.7f; Table 9.3 | Penalty arithmetic | **correct** | — | Re-derived |

### 3.11 Library helpers not covered above

| # | Function | Verdict | Note |
|---|---|---|---|
| L1 | `wavelengthToFrequencyTHz`, `deltaFGHzFromDeltaLambda`, `dbmToMw`, `photonRate`, `shotNoiseArms`, `photonLifetimePs` (= Qλ/2πc, the energy lifetime), `buildUpFactor` (= κ²/(1−at)²), `neffDispersive`, `roundTripPhase` | **correct** | Spot-checked numerically against the test values |
| L2 | `plasmaDispersion1310` and `plasmaDispersion1550` (Soref–Bennett coefficients) | **correct** | Δn = −6.2e-22·ΔN − 6.0e-18·ΔP^0.8 and Δα = 6.0e-18·ΔN + 4.0e-18·ΔP at 1.3 µm match Soref & Bennett (1987) |
| L3 | `slabNeff`, `stripNeffEIM`, `stripGroupIndexEIM` | **simplified-ok** | The effective-index method gives n_g within the 3.8 to 4.8 band the test asserts. Use FEM for weights and confinement |
| L4 | `stripConfinementEstimate` (0.35 + 0.75x heuristic) | **wrong** for any quantitative use (labelled heuristic) | Do not use it for thermo-optic weights. Use FEM (§2.5) |
| L5 | `solveLinear` / `invertMatrix` | **simplified-ok** | A singular pivot is skipped silently (`continue`, then `row[i] \|\| 1`). Guard it in our model |
| L6 | `linkBudgetMargin` | **correct** | Trivial |

---

## 4. Supporting calculations

### 4.1 Where the textbook's 0.85 comes from

The Poynting fraction in silicon from FEM is 0.852 for a 500 × 220 nm strip at 1310 nm. Ch03 defines the
confinement factor as "the share of the mode's power inside the core", and Ch09 then reuses the same number as the
index weight. The same symbol Γ also appears as a third quantity in Ch10 eq. 10.2b (A_eff ≈ wh/Γ). There are three
different "Γ"s, and the textbook itself warns that "there are at least three quantities with that name".

### 4.2 Why "1.48 × bulk" is not impossible for the index weight

To reach 86 pm/K at n_g = 4.2 needs ∂n_eff/∂T = 2.75 × 10⁻⁴ /K (textbook eq. 9.4a). That is Γ_Si ≈ 1.48 at
1.86e-4, or ≈ 1.42 at 1.94e-4. A weight that high is impossible, but not because Γ ≤ 1. The bound is
Γ_Si ≤ n_g/n_g,Si ≈ 4.2/3.68 ≈ 1.14 when all the weight is in silicon. The FEM value of 1.08 is already close to
that bound.

### 4.3 relOMA curvature against swing (`06_reloma_curvature.py`, derived)

T_min = 0.02. k in dB/K² about the OMA optimum:

| FWHM (Q) | Swing | relOMA | k at 50 pm/K | k at 66 | k at 73 | k at 86 |
|---|---|---|---|---|---|---|
| 374 pm (3500) | 65 pm | −6.65 dB | 0.67 | 1.17 | 1.44 | 1.99 |
| 374 pm (3500) | 194 pm | −2.55 dB | 0.54 | 0.94 | 1.15 | 1.60 |
| 500 pm (2620) | 260 pm | −2.55 dB | 0.30 | 0.53 | 0.64 | 0.89 |
| 650 pm (2015) | 338 pm | −2.55 dB | 0.18 | 0.31 | 0.38 | 0.53 |

The pairs (FWHM, dλ/dT) that match k = 0.67 at the −2.55 dB operating point are (374 pm, 56 pm/K),
(500 pm, 74 pm/K) and (650 pm, 97 pm/K). That is a constant FWHM/(dλ/dT) ≈ 6.7 K, and 6.1 to 7.8 K for k from
0.5 to 0.8.

[verified, then corrected. Every number in the table above recomputes with an independent script (curvature at the
optimum, T_min = 0.02). The swings for −3.0, −2.55 and −2.1 dB are 168.7, 194.5 and 227.3 pm, and the optimum bias
moves from 108 pm (small signal) to 129.5 to 145.8 pm. But this table answers the wrong question. The textbook's
k = 0.67 is (8 − 2.55) dB / (2.85 °C)², a **secant** from the peak to the −8 dB contour (`REF.omaPenaltyDbPerC`).
2.85 K is about 0.65·FWHM at 86 pm/K, where relOMA is far from quadratic. The published anchors are not quadratic
either: 5.45 dB at 2.85 °C and 13.45 dB at 6 °C, a ratio of 2.5 where a parabola gives 4.4. A like-for-like test
asks which (FWHM, dλ/dT) gives a 5.45 dB drop at 2.85 K in the same large-signal model. On the monotonic flank, where
the resonance moves away from the laser, the answer is FWHM/(dλ/dT) = 4.21 K: (374 pm, 89 pm/K), (500 pm, 119 pm/K),
(650 pm, 155 pm/K). The 6 °C anchor gives 3.45 K (Q 3500 → 109 pm/K). On the other flank, the resonance crosses the
laser within about 1.6 K and the OMA rises again on the second flank. A map sampled every 5 °C cannot resolve that.
Conclusions (derived): (i) the ratio-only point stands; (ii) at Q 3500 the map points to about 89 to 109 pm/K, close
to the printed 86, and not to 56; (iii) at 86 pm/K it points to Q ≈ 3600 to 4400, which agrees with
`REF.qFromLinewidth` 3900 to 4200; (iv) with 5 °C sampling, treat all of this as a weak constraint.]

### 4.4 Heater code range against dλ/dT (`06_dac_code_check.py`, derived)

The code range runs from 19,500 to 49,300 of 2¹⁶ over 83 °C. The rows below assume a voltage DAC and use the G25
V→λ curve.

| V_FS | V range | Δλ | Implied dλ/dT |
|---|---|---|---|
| 1.60 V | 0.476 to 1.204 V | 7.34 nm | 88.4 pm/K |
| 1.50 V | 0.446 to 1.128 V | 6.60 nm | 79.5 pm/K |
| 1.44 V | 0.428 to 1.083 V | 6.16 nm | 74.2 pm/K |
| 1.40 V | 0.417 to 1.053 V | 5.89 nm | 70.9 pm/K |

[verified: the arithmetic recomputes, including with a PCHIP interpolant (±0.2 pm/K). Two further sources of spread
widen the conclusion. (i) The V→λ curve is the G25 swept device, at 0.348 nm/mW. The wafer median is 0.405 nm/mW at
10.3 nm FSR, so a median ring would scale every row by about ×1.16, to 82 to 103 pm/K. (ii) The 83 K comes from the
~30 → ~113 °C slide test, while the paper's 25 → >105 °C (80 K) gives 74 to 92 pm/K. So the code range does not pin
dλ/dT, as the note says.]

---

## 5. Internal inconsistencies found in the textbook

1. **Thermo-optic cap.** Ch09 §9.2–9.3 use the 66/70 pm/K ceiling. `REF` (comment on `dLambdaDT_pmK`), the
   `physics.test.ts` test "cannot reach … even at perfect confinement", `SOURCES_AND_GENERATIONS.md` §3.1 and Ch17
   question Q1 still use the obsolete 58 pm/K cap.
2. **What the 0.85 means.** Ch09 says it is the rigorous index weight. Numerically it is the power fraction (§4.1).
3. **Ch09's checkpoint** calls 70 pm/K "wrong twice over". Ch09 eq. 9.4c calls the same number the ceiling.
4. **Ambient injection** differs: Ch13 and PiLoop filter it through the heater poles; Ch17 eq. 17.2 and ArrayLock add
   it directly.
5. **T_d** means total lag in Ch13 eq. 13.23 but transport delay only in Ch17 eq. 17.11b and 17.13.
6. **Heater full scale** is 1.6 V in Ch11, `heaterPowerFromDacCode` and the HeaterActuator widget, but 1.44 V is
   derived in Ch17. The textbook notes this.
7. **Tuning range in FSR units**: "about 1.27 FSR" (REF comment, Ch17) uses the 10.3 nm target. The same device's
   read-off bound is FSR > 14.1 nm, which would make 13 nm less than one FSR. The 8.8 nm wafer-map reading would make
   it about 1.5 FSR. Ch17's "hopping folds the requirement to one FSR and the actuator covers 1.26 FSR" depends on
   which FSR is true.
8. **Monitor in the widgets**: PiLoop and DitherLock use a through-port monitor; the G26 plant in Ch17 uses the drop
   port.
9. **Ch18 citation** of [LM26-P] for >2000 °C/s and 30–113 °C (D3).

---

## 6. What our model will do: agree or disagree

**Agree, and keep:** O1–O3, O9, M4, M5, H3, N3, E3, E5, S1, S2, S8, C3–C5, R6, L1, L2. Also keep the
controller/plant interface discipline (Ch17 step 12).

**Disagree, and change:**

| Item | Our model |
|---|---|
| T2, T3, T4, T6 | dλ/dT from the index weight. Default s₀ ≈ 70 pm/K at 25 °C, β ≈ 1.4 × 10⁻³ /K. Swept from 60 to 90 pm/K. The 86 pm/K value stays as a named "Lightmatter-printed" corner |
| O4 | One add-drop ring with (a, t₁, t₂), or CMT, conserving energy |
| O5 | T_min from κ(λ) and α(T), not from heater voltage. The textbook's variant stays as an option for comparison |
| N5, N6 | Per-element thermal transfer functions and separate disturbance paths |
| N7 | K_G (pm/mW) is primary; R_th is derived |
| E1 | Heater R(T_amb, P), with the DAC law configurable |
| S4, S6 | Dark current that rises with T; neighbour leakage via the bus cascade |
| H1, H2 | η_abs swept from 0.1 to 0.9; separate self-heating resistance |
| R1–R5 | relOMA computed from the spectral model at a configurable swing. The published contours become acceptance targets, not plant outputs |

**Carry as swept unknowns:** T9, O6, O8, M6, N2, N4, E2, S3, C1 (lock scheme), D2 (aggressor provenance) and R1.

---

## 7. Open questions, which belong in the Lightmatter Q&A

1. **What is "~15 GHz/°C" referred to?** Which thermometer and which temperature range, and is it measured or a
   generic figure? Is the dλ/dT of the G26 MRM measured against stage temperature?
2. **How was "808 °C/s within the MRM" (and >2000 °C/s) measured?** If it came from the resonance shift or the heater
   code through ~15 GHz/°C, it is not independent of dλ/dT.
3. **What is the G26 MRM's cross-section and FSR?** Strip or rib, and with which slab? FSR ≥ 2.8 THz, or 4.5 × 400 GHz?
   What are its loaded Q and drop coupling?
4. **What is the heater DAC law and full scale?** Voltage, current or PWM/ΣΔ? And what is the heater's TCR and type:
   doped Si in the ring, or a metal on top?
5. **Which error signal does the TX lock use?** Side-of-fringe on the drop port, dither, or bit statistics? Is the
   monitor dark-current-corrected or normalized against a reference tap?
6. **How is relOMA referenced?** What is the reference plane of the 1 mW, and what driver swing applied for the
   contour maps?
7. **What are the measured thermal step response and crosstalk matrix?**
8. **Is the TX laser feed polarization-maintaining?** The G26 paper does not say.

---

## 8. Sources

Primary, read directly:

- R. Anand et al. (Lightmatter), "16-Wavelength 800-Gbps Bidirectional Link over Single-Mode Fiber Using Microring
  Transceivers", OFC 2026, paper M4B.3. Local PDF. Relevant printed text:
  - Introduction: "intrinsic thermal sensitivity of MRMs (~15 GHz/°C)".
  - §2: "two sets of 8 wavelengths on a 400 GHz grid … offset by 200 GHz".
  - §3.1: "polarization dependent monitoring photodetectors at the input of the receive path". [verified; the text
    is in the Fig. 2(e) caption, which §3.1 refers to]
  - §3.2: "808°C/s within the MRM … between 25 °C and >105 °C with a period of 100 ms", and "monitor photocurrent at
    the MRM drop port".
  - Fig. 3(a): TEC sweep from 35 to 105 °C over 31 h.
  - §4: "±20 GHz".
- B. J. Frey, D. B. Leviton, T. J. Madison, "Temperature-dependent refractive index of silicon and germanium", Proc.
  SPIE 6273, 62732J (2006); arXiv:physics/0606168. Tables 2–4 (n, dn/dλ, dn/dT up to 295 K).
- M. C. M. M. Souza, A. Grieco, N. C. Frateschi, Y. Fainman, "Fourier transform spectrometer on silicon with
  thermo-optic non-linearity and dispersion correction", Nature Communications 9, 665 (2018); arXiv:1710.00061,
  supplementary §1. The silicon TOC first-order term is about 14 % per 100 K above 300 K, and the CTE about +33 % per
  100 K.
- The textbook repository (HEAD 02c9c54): `src/physics/index.ts`, `src/physics/physics.test.ts`, chapters Ch05–Ch18,
  the widgets listed above, and `docs/SOURCES_AND_GENERATIONS.md`.

Secondary, from search summaries only. These were not read in full, and the claims rest only on the numbers quoted:

- L. Colace et al., "Temperature dependence of Ge on Si p-i-n photodetectors", J. Lightwave Technol. (2008): dark
  current ×1.6 to 1.9 per 10 °C. [corrected: L. Colace, M. Balbi, V. Sorianello, G. Assanto, J. Lightwave Technol.
  26, 2211–2214 (2008), abstract: "nearly a factor 1.6 every 10 °C". The ×1.6 to 1.9 is from M. Balbi, V.
  Sorianello, L. Colace, G. Assanto, "Analysis of temperature dependence of Ge-on-Si p-i-n photodetectors", Physica
  E 41, 1086–1089 (2009), abstract. Both were read from the authors' group citation pages, abstracts only.]
- IBM Research Report RC24490 (Ge-on-SOI lateral PIN): 12 nA at room temperature to 138 nA at 86 °C (−0.5 V), with
  an activation energy of 0.3 to 0.35 eV.
- G. Cocorullo, F. G. Della Corte, I. Rendina, Appl. Phys. Lett. 74, 3338 (1999): Si dn/dT from 300 to 550 K at
  1523 nm. The abstract and fit were not retrieved, so no number from it is used here.

Computations, all derived and reproducible with the scripts in `docs/research/scripts/`:

- femwell 0.1.12 FEM, order-2 elements, run under `tools/memguard.py --limit-gb 3`.
- The relOMA curvature and DAC code-range checks, in plain numpy and scipy.

---

## Verification log

An independent checker re-derived or re-sourced the claims below on 2026-10-09 and tried to refute each one. The
checker did not reuse the note's scripts. The numerical checks used new scratch scripts: a tidy3d 2.12.0 local FDFD
mode solver run under `tools/memguard.py --limit-gb 3` (peak 1.3 GB), and plain numpy/scipy for the relOMA, leakage
and dark-current arithmetic. Sources were opened directly where possible: the G26 PDF, Frey 2006 (arXiv PDF) and
Souza 2018 (arXiv PDF, supplementary §1). Where only abstracts or search summaries were reachable, the entry says so.
Each verdict is **confirmed**, **corrected** or **unverifiable**.

| # | Claim (where) | Verdict | Evidence |
|---|---|---|---|
| V1 | FEM: 500 × 220 strip at 1310 nm has n_eff 2.712, n_g 4.157, ∂n_eff/∂n_Si 1.077, power fraction 0.852; 400 × 200 has 1.086; rib has 1.039 (§2.5, headline 1) | confirmed | Independent FDFD solver (tidy3d, not femwell): 2.717 / 4.163 / 1.080 / 0.856; 400 × 200: 1.092; rib: 1.045. All dλ/dT within 0.2 pm/K. Finite-difference ∂n_eff/∂n, so the check does not depend on the overlap formula |
| V2 | Exact identity Σ n_g,i·∂n_eff/∂n_i = n_g; rigorous ceiling λ·(dn_Si/dT)/n_g,Si = 66.3 pm/K (§2.3, headline 2) | confirmed | Re-derived (β depends on the k₀·n_i only, so the chain rule gives the identity). tidy3d reproduces it to 1 × 10⁻⁴. w_i ≥ 0 because ∂β/∂ε_i ≥ 0. 1310 × 1.86e-4 / 3.677 = 66.3 pm/K (65.9 with Frey's n_g,Si = 3.70) |
| V3 | Γ_i = (n_g/n_i)·u_i (§2.3) | confirmed, with a caveat | Holds with the waveguide-only n_g = Σ n_i·Γ_i (1.082 against 1.080). With the dispersive n_g it is 5 % high (1.134). Note added in §2.3 |
| V4 | dλ/dT table: 63.6 / 61.7 / 63.8 pm/K at 1.86e-4, 66.3 at 1.94e-4, 73.4 at 2.15e-4; power-Γ formula 50.4 / 45.9 / 51.5 (§2.5) | confirmed | Hand recomputation from the tabulated weights agrees to ±0.1 pm/K. The identity checks on the table (e.g. 3.677·1.077 + 1.4616·0.133 = 4.155 against 4.157) also hold |
| V5 | relOMA k ≈ 0.67 dB/°C² constrains FWHM/(dλ/dT) to 6 to 8 K; (Q 3500, 56 pm/K) etc. reproduce it (headline 5, T12, R4, O6, §4.3) | **corrected** | k is a secant, (8 − 2.55)/2.85², taken 0.65·FWHM from the optimum. It was compared with the local curvature. Matching the published drops in the same large-signal model gives FWHM/(dλ/dT) ≈ 4.2 K (2.85 °C anchor) and 3.5 K (6 °C anchor). That is 89 to 109 pm/K at Q 3500, or Q 3600 to 4400 at 86 pm/K. The ratio-only conclusion and "not 50 pm/K" both stand, and the latter more strongly. The map's 5 °C sampling keeps it a weak constraint |
| V6 | G26 relOMA −3.0 to −2.1 dB needs a 169 to 227 pm swing at Q 3500; k table in §4.3; optimum bias 130 to 146 pm; k = 1.99 → 1.50 to 1.68 (M5, M6, R5) | confirmed | 168.7 / 227.3 pm; −6.65 dB at 65 pm; k at 86 pm/K = 1.99 (65 pm), 1.68 (169), 1.50 (227); bias 129.5 to 145.8 pm; small-swing limit 2.06, peak limit 0.92 |
| V7 | Frey 2006 Table 4: dn/dT = 1.94e-4 /K (1.3 µm, 295 K), 1.87e-4 (1.5 µm); Tables 2 and 3: n = 3.50616, dn/dλ = −0.14803 /µm | confirmed | Read directly from arXiv:physics/0606168, Tables 2 to 4. n_g,Si = 3.699 |
| V8 | Si dn/dT rises about 14 % per 100 K (Souza 2018 supp.); 2.17e-4 /K at 105 °C (§2.6, headline 3) | 14 %: confirmed. 2.17e-4: **corrected** to 2.14e-4 | Souza supp. §1 prints "close to 14%". The fit is 1.8e-4 + 2.4e-7·ΔT − 2.1e-10·ΔT² (13.3 %/100 K, first order), so the ratio from 295 to 378 K is 1.104. Effect on dλ/dT: −0.3 pm/K |
| V9 | Si CTE 2.6e-6 at 300 K, +33 %/100 K; expansion adds about 2.8 pm/K at 105 °C (§2.6, T1) | **corrected** | Souza prints α₁ = 2.5e-6 /K. +33 % is the first-order term only; the second-order term is about −9 %/100 K, so the net is about +27 %/100 K (+22 % from 25 to 105 °C). Expansion: 1.9 to 2.3 pm/K at 25 °C and 2.3 to 2.8 at 105 °C |
| V10 | G26 printed text: "~15 GHz/°C" in the introduction, with no method; 808 °C/s "within the MRM"; 25 °C to >105 °C, 100 ms period; "monitor photocurrent at the MRM drop port"; 35 to 105 °C over 31 h; ±20 GHz; heater "integrated adjacent to a MRM"; 400 GHz grid offset by 200 GHz; "multiplexed to a single output fiber"; reflections and backscattering | confirmed | All verbatim in the local PDF. The polarization-PD sentence is in the Fig. 2(e) caption. The Fig. 3(b) caption says "between 25°C and 105°C" (no ">") |
| V11 | 15 GHz/°C = 86 pm/K at 1310 nm; 58 pm/K Γ = 1 cap; 1.48× bulk (T4, T9, §4.2) | confirmed | 15e9·(1.31e-6)²/c = 85.9 pm. 1310·1.86e-4/4.2 = 58.0 pm/K. 86·4.2/1310 = 2.757e-4 = 1.48 × 1.86e-4. Bound Γ_Si ≤ n_g/n_g,Si = 1.14. The stale 58 pm/K text was confirmed in `index.ts` (REF comment), `physics.test.ts` (55 < cap < 62), `SOURCES_AND_GENERATIONS.md` §3.1 and Ch17 |
| V12 | Ge PD dark current ×1.6 to 1.9 per 10 °C (Colace JLT 2008); 0.4 µA and 5.8 pm at 105 °C (headline 7, S4, §8) | **corrected** (citation and range) | Colace et al., JLT 26, 2211 (2008) abstract: "nearly a factor 1.6". The 1.6 to 1.9 is Balbi et al., Physica E 41, 1086 (2009). IBM RC24490 (search summary only; the server refused connections): 12 → 138 nA, −0.5 V, 10 × 10 µm, E_a 0.30 to 0.35 eV. Extrapolated to 105 °C: 0.22 to 0.47 µA, a 3.2 to 6.7 pm offset. "Up to about 6 pm" is the top of the range |
| V13 | Neighbour leakage 0.67 % per neighbour at 400 GHz, Q 3500; 1 % change → 2 to 3 pm (S6, headline 7) | leakage: confirmed. Sensitivity: **corrected**. Omission found | 1/(1 + (800/65.4)²) = 0.66 % (Airy 0.78 %). A 1 % change in one neighbour's power gives 0.02 pm; 1 % of the own signal gives 2.2 pm. Added: at FSR = 4.5 slots, adjacent-order leakage is 2.7 % per channel (~16 pm for two). Those orders also coincide with the RX grid |
| V14 | G26 DAC code range implies 70.9 to 88.4 pm/K for V_FS 1.40 to 1.60 V (E2, §4.4) | confirmed (arithmetic); conclusion widened | Recomputed (linear and PCHIP). Using the wafer-median heater efficiency instead of the swept G25 device scales the result by ×1.16. Using 80 K instead of 83 K gives 74 to 92 pm/K |
| V15 | Q_int ≈ 7000 → α ≈ 29 /cm (125 dB/cm), so FCA dominates and η_abs ≈ 0.6 to 0.9 (H1) | α: confirmed. FCA dominance: unverifiable | 2π·4.2/(7000·1.31e-4 cm) = 28.8 /cm. Soref–Bennett 1.3 µm coefficients (verified as reproduced in US 9,310,185) with an assumed 1 to 3e18 cm⁻³ doping give only 4 to 13 /cm |
| V16 | k_Si falls ~30 % from 300 to 400 K, so the heater gain rises 10 to 25 % (N8) | partly confirmed; mechanism unverifiable | The bulk-Si drop is right. An SOI ring's R_th is usually dominated by oxide, whose k is flat to slightly rising. dn/dT alone gives about +10 % |
| V17 | 808 °C/s × 100 ms ≈ 81 K; edges must be faster than 808 °C/s (D2) | confirmed, sharpened | If one 100 ms period holds both ramps, the mean slew is ≥ 1600 °C/s, so at least one edge is ≥ ~1600 °C/s. The paper does not say whether "period" is per ramp or per cycle |
| V18 | Lag budgets 38 / 34 µs at 2000 °C/s and 72 to 99 µs at 808 °C/s; 29 µs at 86 pm/K (C5) | confirmed | 5 pm/(dλ/dT·dT/dt) |
| V19 | Bistability A > 0.77·FWHM, max Lorentzian slope (3√3/8)(2/w); NEB 1/(2T_w); photon lifetime 2.4 ps; dither phase 3.6° (H3, S2, M4, C6) | confirmed | Re-derived: 8/(3√3)·(w/2) = 0.770w; Q/(2πf) = 2.43 ps; atan(2π·1 kHz·10 µs) = 3.6° |
| V20 | R_th spans 4.0 to 8.9 K/mW; REF 4.7 = 0.405/0.086; 15 µm diameter → FSR 8.7 nm; swept device 0.348 nm/mW (N7, O8, E4) | confirmed | 0.35/0.086 = 4.0, 0.55/0.0625 = 8.9; λ²/(4.2·π·15 µm) = 8.67 nm; 8.878 nm / 25.55 mW = 0.348 nm/mW |
| V21 | femwell has `thermal_transient` (N2) | confirmed | `femwell/thermal_transient.py` (`solve_thermal_transient`) is present in the installed femwell 0.1.12 |
| V22 | Soref–Bennett 1.3 µm coefficients in `plasmaDispersion1310` (L2) | confirmed (secondary) | Same coefficients as the reproduction of Soref & Bennett (1987) in US patent 9,310,185. The 1987 paper itself was not opened |
| V23 | Ch09 says "the 0.85 that this book uses is this rigorous weight … about 71 % of the energy" (§2.4) | confirmed | Ch09.tsx lines 1132 to 1141. tidy3d gives u_Si = 0.954 and Γ_Si = 1.08 for 500 × 220, so the textbook statement is contradicted as the note says |

Net effect on the note's conclusions. The thermo-optic correction (headlines 1 to 3, T2 to T6) survives independent
re-solving and re-sourcing; only small, sub-pm/K changes were needed at 105 °C. The relOMA inference (headline 5)
keeps its main point, but its numbers change. Read like-for-like, the published map is consistent with about 86 pm/K
at Q ≈ 3600 to 4400. It does not favour 56 pm/K. The sensor items (S4, S6) needed a citation fix and a corrected
sensitivity statement. Adjacent-FSR-order leakage, which the note missed, is the largest monitor-crosstalk term if
FSR = 4.5 × 400 GHz.
