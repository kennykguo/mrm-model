# 03 · MRM compact-model physics and equations

Scope: the equation-based ("compact") physics for a silicon depletion-mode **add-drop** microring modulator
(MRM) with an integrated heater, a **drop-port monitor photodiode** and **50 Gb/s NRZ** drive, written for a
time-domain plant model that control algorithms will run against. Lightmatter's G26 transmitter (OFC 2026
M4B.3) is this device class: 8 add-drop MRMs on one bus, a monitor photodiode at each MRM drop port, about
15 GHz/°C, 50 Gb/s NRZ. [corrected: G26 prints "8 MRMs coupled to a single bus waveguide", 50 Gbps NRZ, and
"the monitor photocurrent at the MRM drop port" for the one MRM of Fig. 3(b). "Add-drop" and "at each MRM" are
inferred from that phrase, not printed. The "~15 GHz/°C" is printed in the introduction as the generic
"intrinsic thermal sensitivity of MRMs", not as a measurement of the tested device.]

Provenance words follow `docs/SOURCES_AND_GENERATIONS.md` of the textbook: **printed** (verbatim in the source),
**read-off** (digitized from a figure), **inferred** (arithmetic on printed values), **derived** (computed by us),
**assumed** (no source). Generation tags: **G25** (OFC 2025 Tu3J.6 test vehicle), **G26** (OFC 2026 M4B.3 link).
Numbers that we computed come from the scripts in `docs/research/03_checks/` (section 13).

Related files in this folder: `01_lightmatter_public.md` (Lightmatter corpus, process, patents),
`02_materials.md` (silicon constants at 1310 nm), `04_thermal_and_control.md` (thermal network, disturbances,
control literature), `06_textbook_audit.md` (textbook audit). This file does not repeat their tables. It
cross-references them where they overlap.

---

## 0. Key findings

1. **Use temporal coupled-mode theory (TCMT) with two counter-propagating modes as the optical core**, and use
   the exact Airy (transfer-matrix) form, or a round-trip delay-line model, wherever more than one
   resonance order matters. At the reference ring (Q_L = 3500, finesse 27.6), single-mode TCMT matches the
   exact Airy response to 0.005 dB in the through port and 0.1 % in linewidth. It **underestimates
   drop-port leakage** by 0.7 dB at 400 GHz offset and by 4 dB at FSR/2 (derived). A sum of TCMT modes fixes
   the drop port only if adjacent orders carry alternating signs, (−1)^m (derived, section 2.7).
2. **The exact small-signal electro-optic response of an MRM is not "RC pole × photon-lifetime pole".**
   For pure resonance modulation it is, exactly (derived, and checked to 1e-15 against direct
   linearization and to 4 digits against a TCMT ODE):
   H(s) = −(∂T/∂Δ)·(1 + s/(κ_i+κ_d))·ω_n²/(s² + κs + ω_n²), with ω_n² = (κ/2)² + Δ².
   This has a complex pole pair at −κ/2 ± iΔ and a zero at −(κ_i+κ_d). It is the form published by
   Karimelahi & Sheikholeslami (2016) and consistent with Müller et al. (2014) and with Sacher & Poon (2008).
   At the maximum-slope bias (2Δ/κ = 1/√3) and critical coupling, the optical-only bandwidth is
   **0.84 × FWHM = 55 GHz** for Q_L = 3500, not FWHM (65 GHz). At zero detuning the pole is at FWHM/2. At
   larger detuning or with over-coupling the response peaks: 1.0 dB at 2Δ/κ = 0.81, 5 dB when over-coupled
   (derived, section 5.6). [verified: re-derived by hand (zero at −(κ−κ_e) = −(κ_i+κ_d)) and checked against an
   independent 2×2 linearization to 2e-15; 0.844 FWHM = 55.2 GHz, 1.01 dB and 5.07 dB reproduced. The
   Karimelahi & Sheikholeslami abstract describes a third-order response (one real pole, one zero, a complex
   pole pair) with a detuning-sign-dependent zero, consistent with this form.]
3. **Self-heating is the only optical nonlinearity that matters at milliwatt O-band powers in a Q ≈ 4000
   doped ring, and it matters a lot.** At 2.5 mW in (+4 dBm, G25) and on resonance, linear absorption heats the ring by
   300 to 900 pm of resonance shift, 0.8 to 2.4 linewidths (derived). TPA is at most 0.2 % of the loss,
   TPA-carrier dispersion is −0.03 to −4 pm, and Kerr is 0.3 pm (derived, section 7). The thermal bistability
   threshold is P_abs,max ≥ (4√3/9)·FWHM_λ/K_SH = 0.71 mW absorbed at K_SH = 405 pm/mW. That is
   P_in,th ≈ 0.8 to 2.4 mW for η_abs = 0.9 to 0.3 (derived). G25 ran at about 2.5 mW, so the model must
   carry thermal bistability and hysteresis. [verified: the fold condition (1+u²)²/(2u) minimized at u = 1/√3
   gives K_SH·P_abs,max ≥ 0.7698·FWHM_λ; 0.711 mW at 374 pm and 405 pm/mW. The result rests on the assumed
   K_SH = 405 pm/mW, so it is only as good as that assumption.]
4. **The fraction of intrinsic loss that becomes heat (η_abs) is probably 0.6 to 0.95, not 0.3.** The one
   published separation for a doped MRM gives 19 GHz absorption out of 30 GHz intrinsic loss, 0.63
   (de Cea et al. 2019, Table 1, inferred). [verified: Table 1 prints γ₀ = 30 GHz, γ_lin = 19 GHz and
   γ_rad = 11 GHz, all from "DC transmission for different input powers" (their ref. 15, Borselli et al.); the
   device is a 1545.2 nm, 10 µm-radius, 45 nm zero-change ring, so this is a C-band proxy, not a 1310 nm value.] A Q_i of about 7000 at 1310 nm means about 125 dB/cm of
   intrinsic loss, against 1 to 3 dB/cm for a passive strip, so dopant free-carrier absorption dominates
   (derived). This agrees with `06_textbook_audit.md` H1.
5. **Coupling dispersion explains the direction of G25's extinction change with heater drive, and a fit gives
   its size.** An exponential κ_e/κ_i ∝ exp(s·(λ − λ_c)) fits the G25 read-off notch depth against resonance
   wavelength (30 → 17 dB over 13 nm) with s = 1.48 %/nm, rms 1.7 dB (derived). A straight point coupler with
   a 150 to 250 nm gap gives about 0.6 to 1.0 %/nm on its own (derived, crude), so part of the slope may come
   from the coupler shape or from loss dispersion.
6. **Depletion efficiency falls with reverse bias as the junction capacitance does.** Δλ(V) ∝ ΔQ_j(V), the
   removed depletion charge. Fit to the four G25 read-off points with V_bi fixed at 0.9 V:
   Δλ = 137.6·(√(0.9+V) − √0.9) pm, rms 4.1 pm. That gives 72 pm/V at 0 V, 50 at 1 V, 40 at 2 V and 32 at
   3.75 V (derived).
7. **Backscatter matters for the bidirectional link through reflection, not through resonance splitting.**
   A lumped backscatter of field reflectance r_bs per round trip gives a coupling rate g = r_bs/T_rt and a
   splitting of r_bs·FSR/π. At r_bs = 0.0124 (Bahadori et al. 2018, AIM ring) the reference ring splits by
   only 7 GHz against a 65 GHz linewidth. Its on-resonance reflection back toward the input is −19 dB,
   though, and it is −31 dB at r_bs = 0.003 (derived). G26 names on-chip reflections and backscatter as a
   bidirectional-link penalty (printed).
8. **The drop-port monitor measures stored energy, and therefore the self-heating power.** P_drop = κ_d·U,
   so P_abs = η_abs·(κ_i/κ_d)·P_drop (derived). A drop monitor is a zero-background Lorentzian **peak** with no
   sign information. A through tap carries the same steady-state information (P_thru = P_in − P_drop −
   κ_i·U) but sits on a large pedestal and needs P_in.
9. **Numerics: eliminate the optical field adiabatically for control simulation.** The field relaxes in
   2/κ = 4.9 ps. At a 2000 °C/s aggressor the quasi-static error is 5e-9 (derived). Keep a fast TCMT
   (or delay-line) layer only for eye diagrams, OMA and relOMA, ISI and peaking, and to tabulate
   data-averaged absorbed power and monitor current for the slow layer (section 9).

---

## 1. Anchor values for the reference ring

These are the numbers the equations below are evaluated at. They are parameters, not constants.

| Quantity | Value | Provenance and source |
|---|---|---|
| λ₀ | 1310 nm (ω₀ = 1.438e15 rad/s, f₀ = 228.9 THz) | G26 printed ("centered at 1310 nm") |
| Q_L | 3500; 3900–4200 from linewidth | G25 printed on the talk; read-off (textbook REF) |
| FWHM at Q_L = 3500 | 65.4 GHz = 374 pm | derived |
| κ (total energy decay rate) | 4.11e11 s⁻¹ | derived, κ = ω₀/Q_L |
| Energy lifetime 1/κ; field lifetime 2/κ | 2.43 ps; 4.87 ps | derived |
| n_g | 4.2 (textbook, assumed); 4.33 (derived, 400 × 200 nm strip, local mode solve); 3.9–4.16 (`01_…` §4.4) | assumed / derived |
| L (circumference) | 39.6 µm (R = 6.3 µm, derived from 1.8 THz) | derived (textbook) |
| T_rt = n_g·L/c; FSR | 0.555 ps; 1.80 THz = 10.3 nm | derived; the FSR is disputed (8.7–16 nm, `06_…` O8) |
| Finesse F = FSR/FWHM | 27.6 | derived |
| Build-up at critical coupling (all-pass) | F/π = 8.8 | derived |
| Stored energy on resonance at critical coupling | U = 2P_in/κ = 4.9 fJ per mW | derived |
| Resonance thermal sensitivity K_T | ~15 GHz/°C = 86 pm/K | G26 printed (the "~" is in the source) [verified: 15 GHz at 1310 nm = 85.9 pm; printed in the introduction as the generic "intrinsic thermal sensitivity of MRMs", not as a measured value of the tested device] |
| Heater gain | 0.44 nm/mW printed (G25); 0.405 nm/mW implied by 25.45 mW per FSR at FSR 10.3 nm | printed / inferred [verified against REF: 0.44 × 25.45 = 11.2 nm, so the two printed numbers agree only at FSR 11.2 nm; 0.405 holds only at the assumed 10.3 nm] |
| Monitor current at the drop port | ≈ 15 µA (15.5 ± 0.5) | G26 read-off, Fig. 3(b) [verified: re-read at 400 dpi, green trace at 15.4–15.5 µA on both the main and the inset axes] |
| Data | 50 Gb/s NRZ PRBS9 (G26; 56 Gb/s characterization) | printed |

---

## 2. Optical core: TCMT of an add-drop ring with clockwise / counter-clockwise coupling

### 2.1 Conventions

- Time dependence e^{+iωt}. Every field below is a slowly varying envelope in a frame rotating at the laser
  frequency ω_L.
- a₁ is the mode excited by the input port (call it CW). a₂ is the counter-propagating mode (CCW).
  **|a|² is stored energy in J.** s is a guided-wave amplitude with **|s|² in W**.
- All κ are **energy** decay rates in s⁻¹. The corresponding amplitude rates (the "1/τ" of Haus, Little et al.
  1997 and Karimelahi 2016) are κ/2. In particular 1/τ_e = κ_e/2 and the Haus coupling is μ_e² = 2/τ_e = κ_e.
- Δ = ω_L − ω_r in rad/s. Δ > 0 means the laser is at a **higher frequency (shorter wavelength) than the
  resonance**, which is the "blue side". In wavelength, δλ = λ_L − λ_r ≈ −λ²Δ/(2πc). The normalized detuning
  is x = 2Δ/κ, so x = 1 is the half-power point and x = ±1/√3 is the maximum slope of a Lorentzian.

### 2.2 Field equations (four ports, two modes)

$$
\frac{da_1}{dt} = -\Big(i\Delta + \frac{\kappa}{2}\Big)a_1 - i g\,a_2 - i\sqrt{\kappa_e}\,s_{\rm in} - i\sqrt{\kappa_d}\,s_{\rm add}
$$
$$
\frac{da_2}{dt} = -\Big(i\Delta + \frac{\kappa}{2}\Big)a_2 - i g^{*}a_1 - i\sqrt{\kappa_e}\,s'_{\rm in} - i\sqrt{\kappa_d}\,s'_{\rm drop}
$$
$$
\kappa = \kappa_i + \kappa_e + \kappa_d
$$

| Symbol | Meaning | Unit |
|---|---|---|
| a₁, a₂ | CW and CCW mode amplitudes, \|a\|² = stored energy | J^½ |
| s_in, s_add | forward inputs at the input port (bus 1) and the add port (bus 2) | W^½ |
| s′_in, s′_drop | backward inputs entering at the through port and at the drop port (reflections from downstream, for example the monitor PD) | W^½ |
| κ_i | intrinsic energy loss rate: absorption + scattering + radiation, = v_g·α | s⁻¹ |
| κ_e, κ_d | energy decay rate into bus 1 (input/through) and bus 2 (drop/add) | s⁻¹ |
| g | CW↔CCW coupling rate from backscatter (complex) | rad s⁻¹ |
| Δ | ω_L − ω_r(t); ω_r carries all tuning (thermal, voltage, carriers, Kerr) | rad s⁻¹ |

Sources: the single-mode form is Haus, *Waves and Fields in Optoelectronics* (1984) and Little et al., JLT
15, 998 (1997). The CW/CCW coupling term follows Little, Laine & Chu, Opt. Lett. 22, 4 (1997), Kippenberg,
Spillane & Vahala, Opt. Lett. 27, 1669 (2002) and Johnson et al., Opt. Express 14, 817 (2006) (printed forms;
the sign and phase convention here is ours).

### 2.3 Port outputs

$$
s_{\rm thru} = s_{\rm in} - i\sqrt{\kappa_e}\,a_1,\qquad
s_{\rm drop} = s_{\rm add} - i\sqrt{\kappa_d}\,a_1,
$$
$$
s_{\rm refl} = s'_{\rm in} - i\sqrt{\kappa_e}\,a_2 \;(\text{back out of the input port}),\qquad
s_{\rm add,out} = s'_{\rm drop} - i\sqrt{\kappa_d}\,a_2 \;(\text{out of the add port}).
$$

The phase of each coupling term is a reference-plane convention. Fix it once. Only the 2×2 coupler symmetry
(through and cross terms 90° apart) is physical.

### 2.4 Energy normalization and conservation

With g Hermitian (the a₂ equation carries g*), the equations conserve energy exactly:

$$
\frac{d}{dt}\big(|a_1|^2+|a_2|^2\big) = \sum|s_{\rm inputs}|^2 - \sum|s_{\rm outputs}|^2 - \kappa_i\big(|a_1|^2+|a_2|^2\big).
$$

The heat-relevant powers follow from this. The power lost inside the ring is κ_i·U, with U = |a₁|² + |a₂|².
Of that, the absorbed share is η_abs·κ_i·U (section 6). The circulating power is P_circ = U/T_rt = U·v_g/L.
A dissipative (non-Hermitian) part of g, from scatterers that radiate, can be added as an extra loss. Li and
Bogaerts (Photonics Res. 6, 620, 2018) show that coupler back-coupling makes the split peaks asymmetric. A
control-oriented plant model does not need this.

### 2.5 Steady state: closed forms

With only s_in present and D ≡ κ/2 + iΔ:

$$
a_1 = \frac{-i\sqrt{\kappa_e}\,D}{D^2+|g|^2}\,s_{\rm in},\qquad
a_2 = \frac{-\sqrt{\kappa_e}\,g^{*}}{D^2+|g|^2}\,s_{\rm in}
$$
$$
\frac{s_{\rm thru}}{s_{\rm in}} = 1-\frac{\kappa_e D}{D^2+|g|^2},\quad
\frac{s_{\rm drop}}{s_{\rm in}} = \frac{-\sqrt{\kappa_e\kappa_d}\,D}{D^2+|g|^2},\quad
\frac{s_{\rm refl}}{s_{\rm in}} = \frac{i\kappa_e g^{*}}{D^2+|g|^2},\quad
\frac{s_{\rm add,out}}{s_{\rm in}} = \frac{i\sqrt{\kappa_e\kappa_d}\,g^{*}}{D^2+|g|^2}.
$$

For g → 0 these reduce to the familiar forms:

$$
T(\Delta) = \frac{(\kappa_i+\kappa_d-\kappa_e)^2/4+\Delta^2}{\kappa^2/4+\Delta^2}
 = 1-\frac{\kappa_e(\kappa_i+\kappa_d)}{\Delta^2+\kappa^2/4},\qquad
D_{\rm drop}(\Delta) = \frac{\kappa_e\kappa_d}{\Delta^2+\kappa^2/4},\qquad
U = \frac{\kappa_e P_{\rm in}}{\Delta^2+\kappa^2/4}.
$$

Through-port minimum: T_min = ((κ_i + κ_d − κ_e)/κ)². Drop peak: 4κ_eκ_d/κ². Dissipated fraction on
resonance: 4κ_eκ_i/κ².

### 2.6 Mapping between TCMT and the transfer-matrix (Airy) parameters

Transfer-matrix notation (Bogaerts et al., Laser Photonics Rev. 6, 47, 2012): a is the round-trip amplitude
transmission, r₁ and r₂ are the self-coupling (through) amplitudes of the input and drop couplers,
k_x = √(1 − r_x²), and θ is the round-trip phase.

$$
T_{\rm thru}=\frac{r_2^2a^2-2r_1r_2a\cos\theta+r_1^2}{1-2r_1r_2a\cos\theta+(r_1r_2a)^2},\qquad
T_{\rm drop}=\frac{(1-r_1^2)(1-r_2^2)\,a}{1-2r_1r_2a\cos\theta+(r_1r_2a)^2}.
$$

| TCMT quantity | Transfer-matrix equivalent | Notes |
|---|---|---|
| T_rt | n_g·L/c | round-trip group delay |
| FSR (Hz) | 1/T_rt = c/(n_g L) | |
| κ_i | −2 ln(a)/T_rt = α·v_g, with a = e^{−αL/2} | α = power attenuation coefficient [m⁻¹] |
| κ_e | −2 ln(r₁)/T_rt ≈ k₁²/T_rt | |
| κ_d | −2 ln(r₂)/T_rt ≈ k₂²/T_rt | |
| κ | −2 ln(a·r₁·r₂)/T_rt | |
| FWHM (Hz) | κ/2π | Airy exact: (1/πT_rt)·arccos(1 − (1−ar₁r₂)²/(2ar₁r₂)) |
| Q_L, Q_i, Q_e, Q_d | ω₀/κ, ω₀/κ_i, ω₀/κ_e, ω₀/κ_d, with 1/Q_L = 1/Q_i + 1/Q_e + 1/Q_d | |
| Finesse | F = FSR/FWHM = 2π/(κT_rt) | |
| Extinction (through) | −10·log₁₀[((κ_i+κ_d−κ_e)/κ)²] | under- and over-coupling give the same dip; the phase tells them apart |
| Build-up P_circ/P_in on resonance | 4κ_e/(κ²T_rt) = (2F/π)(κ_e/κ) | F/π at all-pass critical coupling |
| Lumped backscatter r_bs per round trip | \|g\| = r_bs/T_rt | splitting 2\|g\|/2π = r_bs·FSR/π |

The logarithmic mapping is the one to use. It matches the Airy linewidth to second order in (1 − ar₁r₂).
The residual is 0.1 % at Q_L = 3500: 65.385 GHz from TCMT against 65.456 GHz exact (derived).

### 2.7 When single-mode TCMT breaks down, and how to handle several FSRs

TCMT assumes four things: the loss and coupling per round trip are small (high finesse); one resonance
dominates near ω_L; κ_e and κ_d do not vary across the linewidth; and parameters change slowly against T_rt.
At the reference ring these hold near resonance and fail in the far tails (derived,
`03_checks/tcmt_checks.py`, add-drop with κ_d = 1 % of κ):

| Offset from resonance | Through: Airy / TCMT | Drop: Airy / TCMT |
|---|---|---|
| 32.7 GHz (half width) | −3.014 / −3.009 dB | −20.00 / −20.00 dB |
| 200 GHz | −0.119 / −0.115 dB | −32.66 / −32.84 dB |
| 400 GHz (G26 per-direction grid) | −0.034 / −0.029 dB | −38.06 / −38.77 dB |
| 902 GHz (FSR/2) | −0.014 / −0.006 dB | −41.89 / −45.81 dB |

[verified: an independent script reproduces every row to ±0.02 dB (FSR taken as 1.800 THz). Analytic check of
the drop-port sign rule: Σ_m (−1)^m/(x − m) = π/sin(πx), which at x = ½ is π against 2 for the nearest two
terms, i.e. +3.92 dB over the single-mode Lorentzian; that is exactly the −45.81 → −41.89 dB gap. Without
the sign the sum is π·cot(πx) = 0 at x = ½, which is why the unsigned sum collapses (−61 dB with seven modes).]

Recommendations:

1. **Quasi-static (control) model: evaluate the exact Airy form** at every laser line, with κ_e(λ), κ_d(λ),
   κ_i(λ) and θ(λ, T, V) as smooth functions. This handles all orders and any finesse at the same cost.
2. **Time-domain model, option A: one TCMT mode per (ring, resonance order) near each line.** Mode m sits
   at ω_r + 2πm·FSR, plus dispersion. The coupling into the drop port of adjacent orders alternates in sign,
   (−1)^m, because the drop field travels half a round trip. Without that sign the summed drop leakage is
   wrong by up to 20 dB at FSR/2. With it, a seven-mode sum matches Airy to 0.1 dB in the drop port
   (derived, `03_checks/multimode_vs_airy.py`). A truncated sum still misplaces the through-port background
   by about 0.01 to 0.02 dB.
3. **Time-domain model, option B: the round-trip delay-line recursion.** It is exact at any finesse and over
   any number of FSRs. It is the form INTERCONNECT's ring element uses, with a round-trip delay
   τ = L·n_g/c (Ansys docs, printed via search excerpt). In the rotating frame:

$$
b(t) = -ik_1 s_{\rm in}(t) + r_1r_2\,a\,e^{-i\theta(t)}\,b(t-T_{\rm rt}),\quad
s_{\rm thru}(t) = r_1 s_{\rm in}(t) - ik_1 r_2\,a\,e^{-i\theta(t)}\,b(t-T_{\rm rt}),\quad
s_{\rm drop}(t) = -ik_2\sqrt{a}\,e^{-i\theta(t)/2}\,b(t-T_{\rm rt}/2)
$$

   Here b is the circulating field just after the input coupler, |b|² is in W, and θ(t) = ω_L·n_eff(t)·L/c.
   A step equal to T_rt = 0.55 ps gives 36 samples per 50 Gb/s bit. A lumped reflector with r_bs and
   t_bs = √(1 − r_bs²) couples a second recursion for the CCW wave.

The G26 comb makes the multi-order case necessary. Eight channels at 400 GHz span 2.8 THz, which is wider
than the 1.8 THz design FSR. Each ring therefore sees lines of its own bank near its adjacent resonance
orders, at about ±200 GHz from them if the FSR is 1.8 THz (textbook correction 9; `06_…` O10, S6).

### 2.8 Backscatter in numbers

| r_bs (per round trip) | g/2π | Splitting | On-resonance reflection, critical all-pass |
|---|---|---|---|
| 0.003 | 0.86 GHz | 1.7 GHz | −31.6 dB |
| 0.0124 (Bahadori et al. 2018, 5 µm AIM ring, extracted) | 3.56 GHz | 7.1 GHz | −19.4 dB |
| 0.03 | 8.6 GHz | 17.2 GHz | −12.2 dB |

All rows are derived from §2.5, with R = κ_e²|g|²/((κ/2)² + |g|²)² at Δ = 0. [verified: all three rows
reproduced. Bahadori et al. 2018 (read in full) lump the scattering as a field reflectance r per round trip
with |t|² + |r|² ≈ 1 and give peaks at ω₀ ± (FSR/ξ)·r, i.e. a splitting of r·FSR/(πξ) Hz, the same mapping as
here. Their ring: AIM MPW, 400 × 220 nm strip, 5 µm radius, 300 nm gaps, near 1550 nm, Q ≈ 16 000,
r = 0.0124 extracted, so the value is a C-band, high-Q datum.] Splitting is resolved only when
2|g| > κ. [corrected: a visible split needs less than that. For this TCMT, a dip at Δ = 0 appears in the drop
port when 2|g| > 0.49κ (exactly g⁴ + 4g² − 1 > 0 with g in units of κ/2) and in the critically coupled through
port when 2|g| > 0.67κ (derived). Bahadori's ring has 2|g|/κ ≈ 0.71 (8.5 GHz split, 12 GHz linewidth) and does
show a split, which the 2|g| > κ rule would not predict. The conclusion below is unchanged: r_bs ≤ 0.03 gives
2|g|/κ ≤ 0.26 at Q_L = 3500.] At Q ≈ 4000 (κ/2π ≈ 55–65 GHz) the doublet is hidden, which is why de Cea et
al. (2019) drop the CW/CCW coupling for modulator-grade Q. The **reflection is not small**, though, and it goes straight back
toward the input of the bidirectional link. Li et al. (Laser Photonics Rev. 10, 420, 2016) report that
silicon-ring backscatter varies strongly from one resonance to the next and that the couplers contribute.
**Model r_bs per ring and per order as a random parameter. No Lightmatter value is published.** Any
back-reflection from the drop-port photodiode enters as s′_drop and also feeds a₂.

---

## 3. Wavelength dependence of coupling and loss, and extinction against resonance position

### 3.1 Why it matters to control

The heater moves the resonance over a full FSR or more: 13 nm at 1.5 V in G25, four reads agree. κ_e, κ_d
and κ_i all depend on λ, so the notch depth, the linewidth and the drop-peak height all move with the heater
code. The plant gain (dP/dλ at the lock point) and the monitor's peak current therefore change with
operating point. G25 shows a notch of 28–30 dB at low drive and 17 dB at 1.5 V (textbook REF `erVsHeaterV`,
read-off, four sources).

### 3.2 Recommended parameterization

$$
\kappa_e(\lambda)=\kappa_{e0}\,e^{s_e(\lambda-\lambda_0)},\quad
\kappa_d(\lambda)=\kappa_{d0}\,e^{s_d(\lambda-\lambda_0)},\quad
\kappa_i(\lambda,V,T)=\kappa_{i0}\,e^{s_i(\lambda-\lambda_0)}+v_g\,\Gamma_j\,\Delta\alpha_j(V)+\dots
$$

λ here is the **resonance** wavelength. Within one linewidth (0.37 nm) the coupling changes by only about
0.3 %, so evaluate κ at the resonance, not at each laser line, unless a line is many nanometres away.

Physical expectations:

- **Evanescent coupler.** The mode tail decay constant γ = k₀√(n_eff² − n_clad²) falls with λ, so the
  coupling grows with λ. Our estimate for a 400 × 200 nm strip at 1310 nm: d ln κ_e/dλ ≈ +0.6, +0.8 and
  +1.0 %/nm at gaps of 150, 200 and 250 nm (derived, crude, including the growth of the point-coupler length
  ∝ √(R/γ)). A local mode solve of the coupled strips gave 0.5–1.3 %/nm, but the solver was not converged in
  gap, so treat this as a range, not a value (derived, `03_checks/coupler_dispersion_modesolve.py`).
  Arc or pulley couplers add phase-matching dispersion and can be much stronger.
- **Free-carrier loss** in the doped junction scales roughly as λ², so s_i ≈ +0.15 %/nm (derived).
- **Bend radiation** grows roughly exponentially with λ at small radius. It is unquantified for
  Lightmatter's ring.

### 3.3 Fit to G25

We fitted the G25 notch-depth sweep against the resonance wavelength from the same sweep
(`HEATER_V_LAMBDA`) with x(λ) = κ_e/(κ_i + κ_d) = exp(s(λ − λ_c)) and
ER = −20·log₁₀|1 − x|/(1 + x). Result: **s = 1.48 %/nm, critical coupling at λ_c ≈ 1297.7 nm,
rms 1.7 dB** (derived). [verified: an independent multi-start least-squares fit on the same REF arrays gives
s = 1.48 %/nm, λ_c = 1297.73 nm, rms 1.66 dB and the same eight modelled values. Note λ_c lies below every data
point, so the fit cannot reproduce the 28 → 30 dB rise at low drive.] Modelled notch: 30.2, 29.6, 26.1, 24.0, 22.1, 20.3, 18.7, 18.0 dB at the eight read-off
points. The read-off values are 28, 30, 30, 24, 22, 20, 18, 17 dB. The sign of s (over-coupled getting
stronger, or under-coupled getting weaker) cannot be determined from intensity data alone, which is the
same point Huang et al. (2026) make. The textbook attributes the trend to coupler dispersion (Ch06). That is
the right mechanism. A straight point coupler gives about half of the fitted slope, so carry
**s ∈ [0.5, 2] %/nm** and fit it per device.

### 3.4 Published compact models that include wavelength-dependent coupling

- Ansys INTERCONNECT "Optical Ring Modulator" (RING). Static through transmission is printed as
  (t₁ − t₂·a·e^{−iφ})/(1 − t₁t₂a·e^{−iφ}). Modulation enters through tables (n_eff and loss against
  voltage). Three time-domain calculation modes are selected in "Numerical/Digital Filter Properties"
  (printed via search excerpt; the page is behind a bot check we did not bypass). The CML Compiler
  "ring_modulator" model builds single-bus and double-bus rings with electrical and thermal modulation. It
  supports statistical corners and figure-of-merit lookup tables that tie Q and extinction together
  (printed via excerpt). Wavelength-dependent coupling enters through the coupler data file. We could not
  confirm the exact functional form.
- Our parameterization above is equivalent, and it is open.

---

## 4. Survey of public compact models

| Model | What it includes | What it leaves out | Use for us |
|---|---|---|---|
| Ansys INTERCONNECT RING (Optical Ring Modulator) | Airy transfer, round-trip delay τ = L·n_g/c, n_eff(V) and loss(V) tables, thermal tuning, three time-domain modes | closed source; not free | reference behaviour only |
| Ansys **NL_RING** (Optical Nonlinear Ring Modulator, 2025 R2.1) | extends RING with self-heating and free carriers; adds a drop port. Defaults (printed via search excerpt): dn/dT = 1.86e-4 /K, thermal time constant **0.28 µs**, heat capacity **16 pJ/K**, thermal confinement 1, dn/dN = −9.5e-27 m³, free-carrier time constant **45 ns**, TPA inputs; both effects off by default | equations not retrieved | default values give R_th = τ/C = 17.5 K/mW (inferred), an upper reference |
| Sorace-Agaskar, Leu, Watts & Stojanović, Opt. Express 23, 27180 (2015) | Verilog-A library of primitive photonic elements carrying amplitude and phase; ring modulators built from primitives; electro-optic co-simulation | self-heating not stated | architecture reference for a circuit-level co-simulation |
| Rhim, Ban, Yu, Lee & Choi, Opt. Express 23, 8762 (2015) | Verilog-A; electrical equivalent circuit plus coupled-mode optics; validated against a fabricated MRM | thermal | the equivalent-circuit topology |
| Saxena & Shawon, arXiv:2311.16174 (2023) | Verilog-A; coupled real-valued ODEs with analytic signals; micro-disk and ring; embedded heater; 7× faster transients | not peer-reviewed | numerics idea: real-valued envelope ODEs |
| Dubé-Demers, …, W. Shi, JLT 33, 4240 (2015) | analytical electrical plus optical dynamics, ring and disk | — | small-signal cross-check |
| Karimelahi & Sheikholeslami, Opt. Express 24, 7585 (2016); Karimelahi PhD thesis, U. Toronto (2016) | closed-form small-signal response: complex pole pair, real RC pole, one zero; loss modulation included; measured | large signal | **our §5.6** |
| Müller et al., Sci. Rep. 4, 6310 (2014) | closed-form small-signal response with complex δω_r (loss modulation, fitted q = 0.13); peaking; asymmetric sidebands; peaking-amplifier equivalent circuit; Q 3500–5000 | thermal | §5.6 and peaking |
| Pile & Taylor, Opt. Express 22, 14913 (2014) | small-signal transfer functions for resonance and coupling modulation in cavity parameters | — | cross-check |
| Yu et al., Opt. Express 22, 15178 (2014) | analytic dynamic model; OMA against bandwidth trade-off through coupling and wavelength | — | design trade-off |
| Johnson, Borselli & Painter, Opt. Express 14, 817 (2006) | TCMT with CW/CCW standing waves plus TPA, FCA, FCD, thermal ODEs and mode-volume definitions | external modulation | **§7 equations** |
| de Cea, Atabaki & Ram, Opt. Express 27, 24274 (2019) | Johnson's model plus external modulation (first-order junction, τ = 10 ps), Poynting-normalized overlaps, measured MRM parameters | CW/CCW coupling (dropped for Q < 1e5) | **§6 and §7 parameters** |
| Arbabi & Goddard, IEEE Photonics J. 4 (2012) | lumped thermal capacitance plus TCMT; self-heating response as a **second-order** linear system; pump-probe validation | — | thermal-network order |
| Coenen et al. (imec), PSC 2023; IEEE TCPMT 15, 1006 (2025) | coupled dynamic thermo-optic analysis and compact model of self-heating; arrays | abstracts not retrieved | candidate thermal-array model |
| Huang, …, Palermo, Fiorentino, Beausoleil (HPE and TAMU), arXiv:2609.36690 (Sep 2026) | 1310 nm depletion MRMs at Q_L 2500–3700; TCMT (Eq. 34), RC and package network, thermal states, startup branch selection, Verilog-A | **not peer-reviewed**; its self-heating constants are **assumed** (160 pm per mW absorbed; P_abs ≃ 0.84·P_bus/(1+x²)); it uses 80 pm/K in Sec. II.1 and 54.6 pm/K in Sec. IV.2 | the closest analogue to G26; its numbers are cross-checks, not anchors |
| Geravand, …, W. Shi (Laval), Nat. Photon. 19, 740 (2025) | coherent (phase plus amplitude) MRM dynamics; chirp; push-pull MZI-assisted MRM | — | chirp, if dispersion ever matters |
| Flexcompute PhotonForge example "Optical Peaking in a Microring Modulator" | delay-line ring against CMT; at zero detuning the response is a single pole at 1/(2πτ_a) = FWHM/2; peaking near f ≈ \|Δ\| | commercial tool | independent confirmation of §5.6 |

[verified, this table: NL_RING defaults 0.28e-6 s, 16e-12 J/K, dn/dT 1.86e-4 /K, thermal confinement 1 and both
effects off by default were re-confirmed through a search excerpt of the Ansys help page; the 45 ns free-carrier
lifetime and dn/dN = −9.5e-27 m³ were not re-confirmed (the excerpt stops before them). τ/C = 17.5 K/mW is
correct arithmetic. PhotonForge page re-read: single pole at 1/(2πτ_a), τ_a the field lifetime, FWHM =
1/(πτ_a). Karimelahi & Sheikholeslami bibliographic record (Opt. Express 24(7), 7585, 2016) and abstract
confirmed; Saxena & Shawon arXiv:2311.16174 abstract confirmed (Verilog-A, microdisk and ring, heater, about
7× faster transients); Huang et al. arXiv:2609.36690 exists (v1, 29 Sep 2026) with the listed authors.]

---

## 5. Depletion (PN and PIN) electro-optic physics

### 5.1 Junction electrostatics (abrupt PN; Sze & Ng 2006)

$$
W(V_R) = \sqrt{\frac{2\varepsilon_{\rm Si}(V_{bi}+V_R)}{q}\,\frac{N_A+N_D}{N_AN_D}},\qquad
x_n = W\frac{N_A}{N_A+N_D},\quad x_p = W\frac{N_D}{N_A+N_D}
$$
$$
C_j(V_R)=\frac{\varepsilon_{\rm Si}A_j}{W}=\frac{C_{j0}}{(1+V_R/V_{bi})^{m}},\quad m=\tfrac12\ (\text{abrupt}),\ m=\tfrac13\ (\text{linearly graded});\qquad
Q_j(V_R)=\int_0^{V_R}C_j\,dV
$$

V_R is the reverse bias (positive), V_bi = (kT/q)ln(N_AN_D/n_i²) ≈ 0.9–1.0 V at 1e17–1e18 cm⁻³, and A_j is the
junction area (junction height × modulated length L_j). In a **PIN** junction W = W_i + W_dep(V). C_j and the
efficiency then depend on bias more weakly, because the intrinsic region of width W_i contributes no
modulated carriers. V_bi falls by about 1.1–1.2 mV/K (`02_materials.md` §11), so the efficiency at fixed bias
drifts by a few percent over 100 K (derived there).

### 5.2 Index and loss change (plasma dispersion at 1.3 µm)

Nedeljkovic, Soref & Mashanovich, IEEE Photonics J. 3, 1171 (2011), 1.3 µm row (printed; transcribed from the
open-source Tidy3D implementation and identical to `02_materials.md` §7), with N in cm⁻³:

$$
\Delta n = -\big(2.98\!\times\!10^{-22}\,\Delta N_e^{1.016} + 1.25\!\times\!10^{-18}\,\Delta N_h^{0.835}\big),\qquad
\Delta\alpha\,[{\rm cm^{-1}}] = 3.48\!\times\!10^{-22}\,\Delta N_e^{1.229} + 1.02\!\times\!10^{-19}\,\Delta N_h^{1.089}
$$

Keep Soref & Bennett (1987) behind a switch: Δn = −(6.2e-22·ΔN_e + 6.0e-18·ΔN_h^0.8),
Δα = 6.0e-18·ΔN_e + 4.0e-18·ΔN_h. The two fits differ by about 10 % in Δn at modulator doping. [verified: the
NSM 1.3 µm row matches the installed tidy3d table (`default_nedeljkovic_soref_mashanovich_coeffs`, wvl = 1.3)
digit for digit. At 1e18 cm⁻³ NSM gives Δn_e = 5.78e-4 and Δn_h = 1.34e-3 against Soref–Bennett 6.2e-4 and
1.51e-3, so the gap is 7 % (electrons) and 11 % (holes).]

**From material to mode.** Use the Poynting-normalized weight (Robinson et al., Opt. Express 16, 16659,
2008; also used by de Cea 2019). It can exceed 1 in high-index-contrast guides:

$$
\Delta n_{\rm eff}(V)=\iint \Delta n(x,y;V)\,w(x,y)\,dA,\qquad
w=\frac{c\,\varepsilon_0\,n\,|E|^2}{\iint {\rm Re}(\mathbf E\times\mathbf H^*)\cdot\hat z\,dA}
$$

For a lateral junction at x_j, a 1-D approximation is
Δn_eff(V) ≈ ψ(x_j)·[Δn_e(N_D)·(x_n(V) − x_n(0)) + Δn_h(N_A)·(x_p(V) − x_p(0))]·h_j. Here ψ (m⁻²) is the
local value of w and h_j the junction height. The resonance and loss then move as

$$
\Delta\lambda_r(V)=\frac{\lambda}{n_g}\,\frac{L_j}{L}\,\Delta n_{\rm eff}(V),\qquad
\Delta\kappa_i(V)=v_g\,\frac{L_j}{L}\,\Delta\alpha_{\rm eff}(V).
$$

For the plain strip at 1310 nm the bulk weight is ∂n_eff/∂n_Si = 1.084 (derived, 400 × 200 nm, local mode
solve). Lightmatter's process stack (GF 45SPCLO, 160 nm photonics SOI, `01_…` §4) implies a rib with slabs,
so recompute w for the real cross-section.

### 5.3 Efficiency against bias: the charge-control form

Electrons enter Δn almost linearly (exponent 1.016). Holes enter as N^0.835 at fixed doping, but the removed
charge is set by W(V). Therefore Δλ_r ∝ ΔQ_j(V) to within the hole sub-linearity, and

$$
\frac{d\lambda_r}{dV}\;\propto\;C_j(V),\qquad
\Delta\lambda_r(V)=A\left[(V_{bi}+V)^{1-m}-V_{bi}^{1-m}\right].
$$

Fit to the G25 read-off ladder (0, 56.4, 66.8, 84.4 pm at 0, 0.8, 1.2, 1.6 V, textbook `MOD_SHIFT_PM`; derived,
`03_checks/tcmt_checks.py`):

| Model | Fit | rms | pm/V at 0 / 1 / 2 / 3.75 V |
|---|---|---|---|
| m = ½, V_bi = 0.9 V fixed | A = 137.6 pm V^−½ | 4.1 pm | 72.5 / 49.9 / 40.4 / 31.9 |
| m = ⅓, V_bi = 0.9 V fixed | A = 96.6 pm V^−⅔ | 4.8 pm | — |
| m = ½, V_bi free | V_bi hits the 0.3 V bound | 2.9 pm | 93.7 / 45.0 / 33.8 / 25.5 |

[verified: all three rows reproduced by an independent least-squares fit to `MOD_SHIFT_PM` (A = 137.6, rms
4.12 pm; A = 96.6, rms 4.82 pm; free fit A = 102.6 at the 0.3 V bound, rms 2.86 pm). REF documents the ladder as
cumulative shift against reverse bias, which is the reading used here. The residuals are +7.5, −2.1 and
−2.6 pm at 0.8, 1.2 and 1.6 V, so the 0.8 V point carries most of the misfit.]

The free fit drives V_bi to the bound, which is unphysical. Four noisy read-off points cannot pin V_bi, so use
the fixed-V_bi row. The textbook's flat 50 pm/V matches only near 1 V. The G26 bias axis runs 1.0 to 3.75 V
(printed). Across it, this model's efficiency falls from about 50 to 32 pm/V. This is the
"pm/V falls with reverse bias" shape the task names.

### 5.4 Loss modulation that comes with index modulation

Write the complex resonance shift as δω̃_r = δω_r + i·δ(κ_i/2) and the loss ratio as q = δ(κ_i/2)/δω_r.
From the plasma dispersion: q = λΔα/(4πΔn), for a small change in carrier density at fixed doping. With the
NSM derivatives at 1e18 cm⁻³ (`02_materials.md` §7): q ≈ 0.10 for electrons, 0.04 for holes and 0.06 for both
together (derived). Müller et al. fitted q = 0.13 in the C band (printed). They note that the fit is somewhat above, but near,
the value calculated from bulk-silicon coefficients. [corrected: Müller et al. print that the fitted 0.13 "is higher
(2×) but still reasonably close to" the value calculated from bulk-silicon coefficients, so the fit is about twice
the bulk estimate, not "somewhat above". The main text does not print the operating wavelength; the device is a
10.3 µm-radius IME/OpSIS ring, so "C band" is assumed, not printed. The derived q values here (0.100, 0.041,
0.062) are verified by recomputation.] q moves the small-signal zero (§5.6) and makes the response asymmetric
between blue and red detuning. Carry q ∈ [0.03, 0.15].

### 5.5 Electrical drive network

Use a large-signal, charge-based form, so that C_j(V) is handled exactly. It is the topology of Rhim et al.
2015 and Karimelahi 2016, reduced:

$$
R_s\,\frac{dQ_j}{dt}=V_{\rm drv}(t)-V_j(Q_j)-R_{\rm drv}\Big(\frac{dQ_j}{dt}+C_{\rm pad}\frac{dV_{\rm pad}}{dt}\Big),\qquad
V_j=Q_j^{-1}(Q),\qquad \Delta\lambda_V(t)=K_Q\,[Q_j(t)-Q_j(V_{\rm ref})].
$$

Here V_drv is the ideal driver waveform (NRZ with finite edge rate), R_drv the driver output resistance,
C_pad the pad and wiring capacitance, R_s the series resistance of slab and contacts, and
K_Q = A/(2C_j0·V_bi^½) for m = ½. Small-signal, this is H_E(s) = 1/(1 + sR_eqC_eq).

No G25 or G26 value of C_j or R_s is printed. Representative 1310 nm values from Huang et al. 2026: C_j 6.6 to
20.3 fF and R_s 16 to 247 Ω (printed, other devices). [corrected: Huang et al. Table 1 (arXiv HTML, read through
a fetch summary) lists C_j of 6.0 / 10.0, 6.6 / 13.2 and ~20 fF, and R_s of 68.1 / 35.4, 385.9 / 246.7 and ~16 Ω
across the devices they compare; the ~20 fF and ~16 Ω entries are marked as calculated in the source paper, not
measured. The printed span is therefore C_j 6.0–20.3 fF and R_s 16–386 Ω.] Müller 2014: C_j 17 fF at −1 V,
R_s 215 Ω, RC limit 43 GHz (printed, 1550 nm). [verified: all three printed in Müller et al. (Q 3500–5000);
1/(2π·215 Ω·17 fF) = 43.5 GHz. Corrected: "1550 nm" is not printed in the main text (see §5.4 note).] G25 drives the junction directly from a monolithic CMOS driver with no 50 Ω line
(printed in the paper, per the textbook). The "0.9 Vpp" is the pattern-generator swing into the on-chip driver,
which is rail-to-rail by design (`01_…` finding 4).

### 5.6 Small-signal electro-optic response, and what Sacher & Poon (2008) showed

**Exact optical response, pure resonance modulation.** Linearize §2.2 (g = 0) about the steady state, with
δω_r(t) as input and the through power as output (derived; verified in `03_checks/tcmt_checks.py`):

$$
H_{\rm opt}(s)\equiv\frac{\delta P_{\rm thru}(s)}{P_{\rm in}\,\delta\omega_r(s)}
=-\frac{\partial T}{\partial\Delta}\;\Big(1+\frac{s}{\kappa_i+\kappa_d}\Big)\;
\frac{\omega_n^2}{s^2+\kappa s+\omega_n^2},\qquad
\omega_n^2=\frac{\kappa^2}{4}+\Delta^2,\qquad
\frac{\partial T}{\partial\Delta}=\frac{2\Delta\,\kappa_e(\kappa_i+\kappa_d)}{\omega_n^4}
$$

Total electro-optic response: H_EO(s) = H_E(s)·(dω_r/dV_j)·H_opt(s), where H_E is the RC network of §5.5.

Properties:

- **Poles** at s = −κ/2 ± iΔ. The real part is the **field** decay rate κ/2, which is half the linewidth.
  The imaginary part is the laser–resonance detuning. Damping ratio ζ = (κ/2)/ω_n = 1/√(1 + x²).
- **Zero** at s = −(κ_i + κ_d). At critical all-pass coupling this is −κ/2. Over-coupling (κ_e > κ_i + κ_d)
  moves the zero toward the origin and boosts high frequencies. With loss modulation (q ≠ 0) the zero moves
  further, depending on the sign of Δ (Karimelahi & Sheikholeslami 2016; thesis Eq. 3.12). Their abstract
  describes the full response as third order: one real pole, one zero and a complex-conjugate pole pair
  (printed).
- **DC gain ∝ Δ.** At Δ = 0 pure index modulation gives no first-order intensity modulation. The response
  there is set by q and has a single real pole at κ/2.

Numbers at Q_L = 3500 and 1310 nm (FWHM 65.4 GHz), optical part only, electrical 3 dB point (|H|² = ½)
(derived):

| Bias point x = 2Δ/κ | Coupling | Optical 3 dB bandwidth | Peaking |
|---|---|---|---|
| 0.30 | critical | 38.7 GHz (0.59 FWHM) | 0 dB |
| **0.577 (= 1/√3, maximum static slope)** | critical | **55.2 GHz (0.84 FWHM)** | 0.10 dB |
| 0.813 (through-port at −4 dB) | critical | 75.7 GHz (1.16 FWHM) | 1.0 dB |
| 1.0 | critical | 95.2 GHz | 2.1 dB |
| 2.0 | critical | 246 GHz (DC gain much lower) | 8.8 dB |
| 0.813 | over-coupled, κ_e = 3κ_i | 152 GHz | 5.1 dB |
| 0.813 | under-coupled, κ_e = κ_i/3 | 54.4 GHz | 0.1 dB |
| 0.577 | critical, q = +0.13 / −0.1 | 51.3 / 58.7 GHz | 0.02 / 0.21 dB |

[verified: every row reproduced to the printed digit by an independent linearization (q entered as
δ(κ_i/2) = q·δω_r); the over-coupled row was checked as κ_e = 3(κ_i + κ_d) with κ_d = 1 % of κ.]

**What Sacher & Poon (2008) showed.** W. D. Sacher and J. K. S. Poon, "Dynamics of microring resonator
modulators," Opt. Express 16(20), 15741–15753 (2008). The Optica full text is behind a JavaScript bot check,
which we did not bypass. What follows rests on the published abstract (via Semantic Scholar) and on three
citing works that restate its results (Sacher et al., arXiv:1206.5337, 2012; Karimelahi's 2016 thesis;
Müller et al. 2014). We did not see the paper's own equations.

1. They derived a **time-dependent model of ring transmission** in which the round-trip phase (index), the
   round-trip loss, or the waveguide–ring coupling varies in time, with a small-signal approximation for
   each. Per a search excerpt of its section list, the paper has sections on time-dependent transmission,
   loss modulation, a small-signal approximation and numerical results.
2. For **intracavity modulation** (index or loss), the intensity-modulation bandwidth is limited by the
   photon lifetime. It shrinks as Q rises, because the stored field must leak out and refill. This is the
   −κ/2 real part of the poles above.
3. When the laser is detuned, the response shows a peak that they called a **"modulation resonance"**
   (Karimelahi thesis, p. 22, printed). It sits near the modulation frequency that equals the
   laser–resonance detuning: the ±iΔ imaginary part of the poles. Müller et al. explain the mechanism as
   interference between the detuned carrier and the perturbation field, which rings at the resonance
   frequency. The overshoot needs a dephasing time π/(2|Δ|) shorter than the field lifetime. [verified:
   Müller et al. print the dephasing time π/2(ω_r − ω₀) and require it to be shorter than τ_a, which they
   define as the 1/e decay time of the field amplitude.]
4. **Coupling modulation** has no such limit. Their abstract states that modulation bandwidths "approaching
   the free spectral range frequency are possible if the waveguide-ring coupling strength is varied"
   (printed). Sacher et al. later demonstrated coupling modulation up to 40 GHz, more than six times the
   cavity linewidth (arXiv:1206.5337 abstract, printed). [verified: both abstracts re-read; Sacher & Poon's
   abstract states that modulation bandwidths approaching the FSR frequency are possible with coupling
   modulation, and arXiv:1206.5337 states modulation free of cavity-linewidth limits "at up to 40 GHz", more
   than 6× the linewidth.]

J. K. S. Poon is a co-author of both the G25 and the G26 Lightmatter papers (printed author lists). Points 2
and 3 are what our closed form expresses. Point 4 does not apply to a depletion MRM, which is an
intracavity modulator.

**Consequences for the plant model and for reading Lightmatter's numbers.**

- The measured electro-optic S21 bandwidth is a **total** response: RC × optics at the detuning used in the
  measurement. If G25's 30.9 ± 2 GHz at −1 V (printed) was measured at x = 1/√3, the implied RC pole is
  34.8 GHz. At x = 0.813 it is 25.1 GHz (derived). The measurement detuning is not published. [verified:
  34.83 and 25.15 GHz recomputed, assuming a single-pole RC and critical coupling.]
- G26's 66.8 GHz with about 1.7 dB peaking at "4 dB detuning" (−2 V, LM26-S slide, via the textbook) is the
  detuning-peaking regime. Critical coupling at x = 0.813 gives 1.0 dB of optical peaking. The extra 0.7 dB
  needs over-coupling, a smaller q, a larger detuning than x = 0.813, or electrical peaking (derived; the
  meaning of "4 dB detuning" and Q_L of the G26 device are not published).
- Moving the bias moves the resonance (about 50 pm/V, or Δx ≈ 0.27 per volt at a 374 pm linewidth). At a
  fixed laser wavelength, a bias change therefore also changes the optical bandwidth. This is a confound when
  bias-dependent bandwidth is attributed to C_j alone.
- For eye diagrams the swing matters. 0.9–1.3 Vpp × 50–70 pm/V gives 45–90 pm, 0.12–0.24 FWHM. The response
  is then mildly large-signal, through the Lorentzian curvature and C_j(V), so use the time-domain model
  (§9).

---

## 6. Thermo-optics, self-heating and thermal bistability

### 6.1 Resonance against temperature

$$
\lambda_r(T)=\lambda_r(T_0)+\int_{T_0}^{T}K_T(T')\,dT',\qquad
K_T=\frac{\lambda}{n_g}\frac{\partial n_{\rm eff}}{\partial T}
=\frac{\lambda}{n_g}\Big[\frac{\partial n_{\rm eff}}{\partial n_{\rm Si}}\frac{dn_{\rm Si}}{dT}+\frac{\partial n_{\rm eff}}{\partial n_{\rm ox}}\frac{dn_{\rm ox}}{dT}\Big]+\text{expansion}
$$

- Lightmatter prints "~15 GHz/°C" (G26) = 85.9 pm/K.
- Derived for a 400 × 200 nm strip at 1310 nm (local mode solve, `03_checks/thermooptic_modesolve.py`):
  ∂n_eff/∂n_Si = 1.084 and ∂n_eff/∂n_ox = 0.234. That gives K_T = **61.7 / 66.3 / 72.8 pm/K** at
  dn_Si/dT = 1.86e-4 / 2.0e-4 / 2.2e-4 K⁻¹ (n_g = 4.33).
- The ceiling is λ·(dn_Si/dT)/n_g,Si: 69 pm/K at 300 K and 77 pm/K at 375 K (`02_materials.md` §3).
- The gap to 86 pm/K remains (`06_…` T4). Use K_T(T) = K_T0·[1 + β_T(T − T₀)] with β_T ≈ 1.2e-3 K⁻¹
  (+9 % from 300 to 375 K, `02_…`), and K_T0 swept over 62–86 pm/K.

### 6.2 Heat sources in the ring

$$
P_{\rm heat,ring}=\underbrace{\eta_{\rm abs}\,\kappa_{i,\rm lin}\,U}_{\text{linear absorption}}
+\underbrace{(\kappa_{\rm TPA}+\kappa_{\rm FCA})\,U}_{\text{nonlinear, §7}}
+\underbrace{\alpha_{\rm sw}\,C_jV_{pp}^2B\,\frac{R_s}{R_s+R_{\rm drv}}}_{\text{CV²f in the junction slab}}
+I_{\rm leak}V_R+I_{\rm ph}V_R
$$

| Symbol | Meaning | Unit |
|---|---|---|
| η_abs | absorbed share of the linear intrinsic loss (the rest is scattered and radiated) | — |
| κ_i,lin | linear part of κ_i | s⁻¹ |
| U | stored energy, \|a₁\|² + \|a₂\|², data-averaged for the slow model | J |
| α_sw | switching activity, ¼ for random NRZ | — |
| B | bit rate | s⁻¹ |
| I_ph | junction photocurrent (sub-bandgap or TPA), section 8 | A |

**η_abs evidence** (no Lightmatter value exists; textbook Q8):

- de Cea et al. 2019, Table 1 (45 nm zero-change MRM at 1545 nm): γ_lin = 19 GHz and γ_rad = 11 GHz
  (printed) give η_abs = 19/30 = **0.63** (inferred). It was measured through power-dependent DC
  transmission, the Borselli, Johnson & Painter (Opt. Lett. 32, 2954, 2007) method. [verified: Table 1 of the
  open-access PDF, including γ₀ = 30 GHz = γ_lin + γ_rad.]
- Zheng et al. (Oracle), Opt. Express 20, 11478 (2012) report that linear (dopant free-carrier) absorption
  dominates the loss of high-speed depletion rings and heats them. Bistability appears at about 100 µW in
  substrate-removed rings (printed). [verified against the abstract: about 100 µW input, free-carrier absorption
  from junction doping, and a thermal response of about 100–200 kHz.]
- Reference-ring loss budget: Q_i ≈ 7000 (near-critical, 28–30 dB notches) means α ≈ 29 cm⁻¹ = 125 dB/cm.
  A passive strip has 1–3 dB/cm. NSM gives 8.7 cm⁻¹ for 1e18/1e18 at full overlap and 31 cm⁻¹ at 3e18/3e18.
  So the doped junction plausibly supplies most of the loss, giving η_abs ≈ 0.6–0.95 (derived, if bend and
  slab leakage are small).
- Huang et al. 2026 assume a 0.84 prefactor; the textbook assumes 0.3 with a sweep of 0.05–1.

**Recommended:** η_abs = 0.8, swept over 0.5–0.95 (assumed central, derived range).

**Electrical (CV²f) heating** is not negligible. At C_j = 20 fF, 1 Vpp, 50 Gb/s and α = ¼: 0.25 mW in total,
of which the share in R_s heats the ring. At 405 pm/mW, 0.1 mW is about 40 pm, 11 % of a linewidth. It
switches with traffic (idle, squelch, pattern density). Yu, Shin, Kim, Zimmermann & Choi (Chin. Opt. Lett.
15, 071301, 2017) report data-rate and data-pattern dependence of the MRM resonance, which they attribute to
dynamic power dissipation (printed abstract; numbers not retrieved).

### 6.3 Thermal coupling and self-heating gain

$$
T_{\rm ring}(s)=T_{\rm env}(s)+Z_{rr}(s)\,P_{\rm heat,ring}(s)+Z_{rH}(s)\,P_H(s)+\sum_{j\ne r}Z_{rj}(s)\,P_j(s),\qquad
Z(s)=\sum_k\frac{R_k}{1+s\tau_k}\ (\text{Foster form, see }04\text{ §2.5})
$$

- Lightmatter anchor: Z_rH(0)·K_T = **405 pm/mW** (inferred from 25.45 mW per FSR at FSR 10.3 nm; 0.44 nm/mW
  printed for the G25 headline). The G25 heater is in-ring resistive (`01_…` §4.3). We therefore assume
  **K_SH ≡ Z_rr(0)·K_T ≈ 405 pm/mW** per absorbed milliwatt (assumed). Reciprocity of the heat equation
  makes Z_rr ≥ Z_rH plausible if the heater sits beside the ring rather than in it.
- Published comparisons: the HPE assumption of 160 pm/mW absorbed; the Ansys NL_RING defaults,
  17.5 K/mW × 80 pm/K = 1400 pm/mW (inferred).
- Ring-local self-heating time constants on other devices: 0.42–0.5 µs (de Cea, γ_th = 2.4 MHz, printed)
  [verified: Table 1 γ_th = 2.4 MHz; the text prints τ_th ≈ 0.5 µs];
  a 100–200 kHz response (Zheng 2012, printed); 0.28 µs (NL_RING default, printed). Arbabi & Goddard 2012
  find a second-order response. **Lightmatter's value is unpublished** (`04_…` covers the heater poles).
  Absorbed light enters at the waveguide node, with no heater-node lag (`04_…` §2.5).

### 6.4 Static self-consistency and the bistability condition

For each laser line, with the field adiabatically eliminated:

$$
\lambda_r=\lambda_{r,\rm cold}+K_{SH}\,P_{\rm abs}(\lambda_L-\lambda_r),\qquad
P_{\rm abs}(\delta)=\frac{P_{\rm abs,max}}{1+(2\delta/{\rm FWHM}_\lambda)^2},\qquad
P_{\rm abs,max}=\eta_{\rm abs}\frac{4\kappa_e\kappa_{i,\rm lin}}{\kappa^2}P_{\rm in}
$$

Three solutions exist (bistability) when

$$
K_{SH}\,P_{\rm abs,max}\;\ge\;\frac{4\sqrt3}{9}\,{\rm FWHM}_\lambda\approx0.770\,{\rm FWHM}_\lambda
\qquad(\text{the onset is at }u=2\delta/{\rm FWHM}=1/\sqrt3).
$$

This is the classic result. It agrees with textbook Ch10's 0.77 factor.

| Case | P_abs,max threshold | P_in threshold |
|---|---|---|
| Reference ring, K_SH = 405 pm/mW, FWHM 374 pm | 0.71 mW | 0.79 / 1.13 / 2.4 mW at η_abs = 0.9 / 0.63 / 0.3 |
| HPE, Q_L 3700 / 2500 (their model) | — | 1.8 / 2.6 mW (printed, assumed constants) |
| de Cea MRM (measured, 0.5 Gb/s) | — | best OMA at about 1.65 mW (measured) against 4.2 mW (model) |

[verified: the reference-ring row recomputes to 0.711 mW and 0.79 / 1.13 / 2.37 mW (0.81 / 1.15 / 2.42 mW if the
1 % drop coupling is kept in 4κ_eκ_i/κ²); the HPE 1.8 / 2.6 mW and the de Cea 1.65 / 4.2 mW figures are printed
in those papers.]

G25 at about +4 dBm (2.5 mW) is above the threshold unless η_abs < 0.28.

- **Stable side:** the laser blue of the hot resonance (x > 0). Heating moves the resonance away from the
  laser, which is restoring; this is thermal self-stability (Carmon, Yang & Vahala, Opt. Express 12, 4742,
  2004).
- **Acquisition:** bring the resonance to the target from the red side (heater power high, then reduce).
  Heating up from cold makes the resonance snap past the laser. Huang et al. 2026 give a concrete example
  (printed, their model): a direct 6.9 mW command lands 684 pm blue of the laser; a 16 mW excursion and then
  6.9 mW lands 142 pm red, the target. [verified: arXiv:2609.36690v1 Sec. IV.2, which also prints the assumed
  160 pm per mW absorbed, 54.6 pm per kelvin and P̄_abs ≃ 0.84·P_bus/(1 + x²) used in §4.]
- Self-heating also rescales the heater's small-signal gain by 1/(1 − ℒ) on the stable side (textbook Ch10).

---

## 7. Silicon nonlinear optics at milliwatt powers

### 7.1 Equations (Johnson et al. 2006 form, rates in s⁻¹)

$$
\kappa_{\rm TPA}=\Gamma_{\rm TPA}\frac{\beta c^2}{n_g^2V_{\rm TPA}}\,U,\qquad
\kappa_{\rm FCA}=\frac{c}{n_g}\,\sigma N,\qquad
\frac{dN}{dt}=-\frac{N}{\tau_{fc}}+\Gamma_{\rm FCA}\frac{\beta c^2}{2\hbar\omega\,n_g^2V_{\rm FCA}^2}\,U^2
$$
$$
\frac{\Delta\omega_r}{\omega_r}\Big|_{\rm FC}=-\frac{\Gamma_{\rm ring}}{n_g}\Big(\frac{dn}{dN_e}+\frac{dn}{dN_h}\Big)N,\qquad
\frac{\Delta\omega_r}{\omega_r}\Big|_{\rm Kerr}=-\frac{\Gamma}{n_g}\,n_2\frac{cU}{n_gV_{\rm Kerr}}
$$

[corrected: the κ_TPA, κ_FCA and dN/dt forms match Johnson et al. 2006 Eqs. (4), (9) and (25) (verified in the
full text). The shift forms do not: Johnson's Eq. (15), and de Cea's Eq. (1), are
Δω₀/ω₀ = −(1/n_Si)·[(dn_Si/dT)·ΔT̄ + (dn_Si/dN)·N̄], with ΔT̄ and N̄ averaged with an n²|E|² weight over the
whole mode, so the confinement sits in the average and the denominator is n_Si, not n_g. The n_g form above is
correct only if Γ means the index weight ∂n_eff/∂n_Si (1.084 for the 400 × 200 nm strip, §5.2), not an energy
confinement factor (0.65 in de Cea, 0.85 here). With Γ = 0.85 and n_g = 4.2, the FCD and Kerr shifts in §7.2
are about 20–25 % too small (derived); they stay negligible, so the hierarchy below does not change.]

| Symbol | Meaning | Unit |
|---|---|---|
| β | TPA coefficient | m W⁻¹ |
| V_TPA, V_FCA | nonlinear mode volumes (≈ A_eff·L to first order) | m³ |
| Γ_TPA, Γ_FCA, Γ_ring | confinement factors as defined by Johnson (Poynting-normalized per de Cea) | — |
| σ | total free-carrier absorption cross-section, σ_e + σ_h ≈ 1e-17 cm² near 1e18 cm⁻³ | m² |
| N | mode-averaged TPA electron–hole pair density | m⁻³ |
| τ_fc | carrier lifetime: 10–100 ps under reverse-bias sweep-out (assumed, `02_…` §8); 0.1 ns used by de Cea; 45 ns NL_RING default (undoped) | s |
| n₂ | Kerr index | m² W⁻¹ |

1310 nm constants (`02_materials.md` §8): β = 0.57 ± 0.17 cm/GW at 1315 nm (Lin et al., APL 91, 021111, 2007)
**against** 2.1 ± 0.4 cm/GW at 1220 nm (Bristow et al., APL 90, 191104, 2007). That is a factor of about 3
between groups, and we report both. n₂ ≈ (3–5)e-14 cm²/W. [verified: Bristow et al. print β = 2.1 ± 0.4 cm/GW
and n₂ = (4.7 ± 2.0)e-5 cm²/GW = 4.7e-14 cm²/W at 1220 nm, and a broad β maximum of 2 ± 0.5 cm/GW (full text);
Lin et al.'s 0.57 ± 0.17 cm/GW at 1315 nm (and 0.48 ± 0.14 at 1501 nm) was confirmed through a search excerpt
of the authors' PDF, not the journal page. Note the two wavelengths differ (1220 vs 1315 nm), so part of the
factor of 3 may be dispersion near the indirect-gap peak.]

### 7.2 Magnitudes for the reference ring

Critical all-pass, laser on resonance (worst case), A_eff = 0.094 µm², Γ = 0.85, intrinsic α_i = 28.8 cm⁻¹
(derived, `03_checks/tcmt_checks.py`):

| P_in | P_circ | I (GW/cm²) | α_TPA/α_i (β 0.57–2) | FCD shift, τ_fc 10 ps | FCD shift, τ_fc 1 ns | Kerr shift | Self-heating shift (η 0.3–0.9, 405 pm/mW) |
|---|---|---|---|---|---|---|---|
| 1 mW | 8.8 mW | 0.009 | 1.8e-4 – 6.5e-4 | −0.01 to −0.02 pm | −0.3 to −0.8 pm | +0.1 pm | +120 to +360 pm |
| 2.5 mW | 22 mW | 0.023 | 4.6e-4 – 1.6e-3 | −0.03 to −0.07 pm | −1.3 to −3.7 pm | +0.3 pm | +300 to +910 pm |
| 8 mW | 70 mW | 0.075 | 1.5e-3 – 5.2e-3 | −0.2 to −0.5 pm | −9 to −28 pm | +0.9 pm | +970 to +2900 pm |
| 10 mW | 88 mW | 0.093 | 1.8e-3 – 6.5e-3 | −0.3 to −0.8 pm | −14 to −41 pm | +1.1 pm | +1200 to +3600 pm |

The FCA from TPA carriers is at most 1 % of α_i (10 mW, 1 ns, β = 2). The hierarchy is
**thermal ≫ TPA-carrier dispersion > Kerr ≈ TPA loss**. At 10 mW with no sweep-out, FCD reaches about 1 % of
the thermal shift. Huang et al. 2026 find the same at 5 mW (5.7 pm at 1 ns, 0.1 pm at 10 ps, against 668 pm
thermal; printed). Self-pulsation (Johnson 2006; de Cea Fig. 2) needs FCD comparable to thermal. It is not
expected under reverse bias, but it is possible with a forward-biased or floating junction. Keep FCD and TPA
as **optional** terms. They cost almost nothing in the quasi-static model.

Cross-document note: `02_materials.md` §8 estimates I ≈ 0.09 GW/cm² at 2.5 mW from a build-up of about 30.
For Q_L = 3500 the build-up at critical coupling is F/π = 8.8 (§2.6), so the intensity is 0.023 GW/cm². The
conclusion (TPA negligible) is the same.

---

## 8. Drop-port monitor photodiode and other intracavity detectors

### 8.1 What the drop-port monitor measures

$$
I_{\rm mon}=\mathcal R_{\rm PD}\,\eta_{\rm c}\sum_{\ell}P_{{\rm drop},\ell}+I_{\rm dark}(T)+i_n(t),\qquad
P_{{\rm drop},\ell}=\kappa_d\,U_\ell\;(\text{plus }|s_{\rm add}|^2\text{ terms if the add port is lit}).
$$

The sum runs over every laser line ℓ on the bus, including other channels and other resonance orders, with
the Airy or multi-mode form of §2.7.

- It is a direct measure of **stored energy**, so P_abs = η_abs(κ_i,lin/κ_d)·P_drop. The monitor reads the
  self-heating drive (derived).
- Its shape against detuning is a zero-background Lorentzian **peak**, symmetric in Δ, with no sign
  information. The lock must stay on one flank. On the blue side (x > 0) self-heating is stabilizing. At the
  maximum-slope point the monitor reads 3/4 of its peak, which is the setpoint in a published drop-port lock
  (Xilinx/AMD patent, `01_…` §3.4).
- In steady state, by energy conservation, P_thru = P_in − P_drop − κ_i·U = P_in − (1 + κ_i/κ_d)·P_drop (no
  backscatter). A **through-port tap** therefore carries the same detuning information. It sits on a large
  pedestal (a notch) and needs P_in, and it includes upstream rings' modulation and other channels at full
  strength.
- The drop monitor is blind to CCW energy, which exits at the add port. A strong doublet (2|g| > κ) would show
  as a split drop peak. [corrected: the drop peak already splits at 2|g| > 0.49κ (derived, §2.8 note).]
- Neighbour leakage at 400 GHz is −21 dB relative to the drop peak per neighbour (Airy −38.06 dB against peak
  −16.99 dB at κ_d = 1 %κ, derived). `06_…` S6 estimates about 1.3 % for two neighbours.
- **Bandwidth.** A monitor TIA or integrating ADC (a Lightmatter patent describes a 1–10 µs integration
  window, `01_…`) averages over data. The slow model needs the **data-averaged** P_drop (§9).
- **Drop coupling, Lightmatter (derived, uncertain).** 15 µA at an assumed 0.9 A/W is 16.7 µW. With 1 to
  2.5 mW at the ring and a lock at x ≈ 0.6, this implies κ_d/κ ≈ 0.4–1.1 %. A GF PDK ring on the same process
  family is reported with about 5 % drop-port coupling (`01_…` finding 2). If that means 5 % of the round trip, then
  κ_d/κ ≈ 22 % and P_in would have to be about 50 µW. [verified: 16.7 µW / (P_in·4·0.5·(κ_d/κ)/1.36) gives
  κ_d/κ = 1.14 % at 1 mW and 0.45 % at 2.5 mW; 0.05/T_rt = 9.0e10 s⁻¹ = 22 % of κ.] **These two clues conflict.** Carry κ_d/κ over
  0.3–5 % and ask Lightmatter.

### 8.2 In-ring detectors (same observable, no extra port)

- **Defect-mediated silicon photodiodes** inside the ring: Logan et al. (McMaster), used for MRM
  stabilization by Padmaraju, Logan, Zhu, Ackert, Knights & Bergman, "Integrated thermal stabilization of a
  microring modulator" (OFC/NFOEC 2013, printed abstract).
- **Photoconductive heaters**: n-doped heater conductance changes with intracavity power (Jayatilleka et al.,
  Opt. Express 23, 25084, 2015; Optica 6, 84, 2019, printed abstracts).
- **The MRM junction's own photocurrent**, from defect and surface absorption plus TPA. It is about 1 mA/W in
  a 45RFSOI p-i-n (Kramnik et al., arXiv:2411.05921, via `01_…`). de Cea, Atabaki & Ram used it to power a
  modulator (Nat. Commun. 2021, printed abstract).
- All of these read U (the linear part) plus a U² TPA part. Their responsivity depends on bias and
  temperature. Junction photocurrent through R_s also shifts the bias: about 3 pm for 100 µA through 1 kΩ at
  33 pm/V (Huang 2026, printed).

---

## 9. Numerics for a control-oriented time-domain model

### 9.1 Time scales (reference ring)

| Process | Time constant | Source |
|---|---|---|
| Round trip | 0.55 ps | derived |
| Optical field | 2/κ = 4.9 ps (energy 2.4 ps) | derived |
| Bit at 50 Gb/s | 20 ps | printed rate |
| Junction RC | about 4–6 ps (25–35 GHz pole) | derived from the total-bandwidth inversion, §5.6 |
| Free carriers under bias | 10–100 ps | assumed |
| Ring self-heating | 0.3–1.6 µs (other devices) | printed elsewhere |
| Heater and slow thermal | µs to ms | `04_…` |
| Aggressors | 2000 °C/s; 7.66 Hz sawtooth; 20 min cycles | printed / read-off |

### 9.2 Adiabatic elimination (slow layer)

For time steps of 10 ns or more, replace the field ODE by its instantaneous steady state (§2.5, or Airy
per order). The error is of order (dΔ/dt)/(κ/2)². At 2000 °C/s × 86 pm/K = 172 nm/s that is 5e-9, and the
resonance moves 8e-7 pm per field lifetime (derived). [verified: dΔ/dt = 1.89e14 rad/s², (κ/2)² = 4.2e22 s⁻²,
ratio 4.5e-9; 172 nm/s × 4.87 ps = 8.4e-7 pm. The G26 printed ramp is 808 °C/s, so 2000 °C/s is conservative.] No control-relevant disturbance comes close.

The slow layer must use **data-averaged** optics. For NRZ with ones-density p₁ and level shifts ±Δ_m/2
about the bias detuning:

$$
\overline{P}_k(\Delta)=p_1P_k(\Delta-\Delta_m/2)+(1-p_1)P_k(\Delta+\Delta_m/2),\qquad k\in\{\text{thru},\text{drop},\text{abs}\}
$$

This is the textbook's Ch17 eq. 17.6b. `06_…` M4 judges it correct. At 50 Gb/s τ_field/T_bit ≈ 0.24, so ISI
and peaking change the true averages slightly. The better option is to **tabulate** P̄_thru, P̄_drop, P̄_abs,
OMA and ER against (Δ, V_bias, V_pp, κ's) once from the fast model with a PRBS segment, then interpolate. The
optics are linear in P_in at fixed Δ, so tables are per unit power. Self-heating enters only through the slow
T.

### 9.3 Fast layer: when it is required

Use the fast layer for:

- eye diagrams, OMA and relOMA, ER under drive, BER estimates;
- peaking and ISI changes as the controller moves detuning;
- chirp, if ever needed;
- PAM4 level spacing;
- dithers or pilot tones above about 1 GHz;
- validating the tables of §9.2.

Recommended integrator: **exponential (integrating-factor) step for the field**, exact for inputs that are
constant over the step:

$$
a(t+h)=e^{-(i\Delta+\kappa/2)h}\,a(t)+\frac{1-e^{-(i\Delta+\kappa/2)h}}{i\Delta+\kappa/2}\,\big(-i\sqrt{\kappa_e}\,s_{\rm in}\big)
$$

(2×2 matrix exponential with backscatter). Use h = 0.25–1 ps (20–80 samples per bit). Use an explicit or
implicit step for the junction charge ODE. A solve_ivp(DOP853) reference with rtol 1e-11 agrees with the
closed-form small-signal response to 4 digits (derived). Hold T, N and the heater fixed over a fast window of
about 1e3–1e5 bits.

### 9.4 Coupling the layers safely near bistability

With step h_slow ≪ τ_th, using P_abs from the previous step is stable while h·|ℒ|/τ_th < 1. Near the fold
(ℒ → 1) or during acquisition, **solve the scalar fixed point** λ_r = λ_cold + K_SH·P_abs(λ_L − λ_r) at each
step: bracket and bisect on δ, choosing the root continuous with the previous state. This reproduces
hysteresis correctly and never jumps branches spuriously.

---

## 10. Recommended equation set (implementation specification)

Indices: ring r, laser line ℓ, resonance order m. Fast variables: a₁, a₂ (optional), Q_j. Slow variables:
thermal node temperatures, N (optional), heater state.

1. **Laser lines.** s_in,ℓ(t) = √P_ℓ(t)·e^{iφ_ℓ(t)} at ω_ℓ. P_ℓ and ω_ℓ carry the laser drift and accuracy
   (±20 GHz spec, printed G26).
2. **Resonance of order m:**
   λ_r,m(t) = λ⁽⁰⁾_r,m + ∫K_T(T)dT + K_Q·[Q_j(t) − Q_j(V_ref)] + K_N·N + K_U·U, with
   λ⁽⁰⁾_r,m = λ⁽⁰⁾_r,0 − m·FSR_λ plus dispersion. K_T in pm K⁻¹; K_Q in pm C⁻¹; K_N = (λ/n_g)Γ(dn/dN_e +
   dn/dN_h) in pm·m³; K_U for Kerr in pm J⁻¹.
3. **Decay rates [s⁻¹]:** κ_e(λ_r), κ_d(λ_r) as in §3.2;
   κ_i = κ_i0·e^{s_i(λ_r−λ0)} + (L_j/L)·v_g·Δα_eff(V_j) + v_g·σ·N + κ_TPA(U). κ = κ_i + κ_e + κ_d.
4. **Fields (fast):** §2.2, one pair (a₁, a₂) per (ring, line, order m within ±1 of the line). Backscatter
   g = r_bs·e^{iφ_bs}/T_rt per ring and order.
5. **Outputs:** §2.3. The bus cascade s_in,ℓ^{(r+1)} = s_thru,ℓ^{(r)} carries every ring's through response
   to every line (`06_…` O10).
6. **Quasi-static (slow):** the closed forms of §2.5, or Airy per §2.6, evaluated with data-averaged
   tables (§9.2).
7. **Junction:** R_s·dQ_j/dt = V_drv − V_j(Q_j) (§5.5) with C_j(V) = C_j0(1 + V_R/V_bi)^{−m} + C_par.
8. **Heat:** P_heat,ring per §6.2, with η_abs, κ_i,lin, U, α_sw C_j V_pp² B.
9. **Thermal:** T_ring = T_env + Z_rr*P_heat,ring + Z_rH*P_H + Σ_j Z_rj*P_j (Foster RC sums, `04_…`).
10. **Carriers (optional):** dN/dt = −N/τ_fc + Γβc²U²/(2ħω n_g² V_FCA²).
11. **Monitor:** I_mon = 𝓡·η_c·Σ_ℓ P̄_drop,ℓ + I_dark(T) + noise, then the TIA or integrating ADC.
12. **Link output:** s_thru of the last ring; OMA and relOMA from the fast layer.

**Units checklist:** U [J]; |s|² [W]; κ [s⁻¹]; Δ, g [rad s⁻¹]; λ [m] internally (report in nm or pm);
T [K]; Q_j [C]; N [m⁻³]; I [A].

---

## 11. Parameter table

| Name | Value | Range | Unit | Provenance | Source / note |
|---|---|---|---|---|---|
| λ₀ | 1310 | 1301–1316 (G26 bank) [corrected: 1301.5–1318.6, derived] | nm | printed | LM26-P [corrected: G26 prints only "400 GHz grid centered at 1310 nm", two sets offset by 200 GHz; 16 lines at 200 GHz composite spacing about 1310 nm span 227.35–230.35 THz = 1301.5–1318.6 nm (derived). No 1301–1316 range is printed in LM26-P. `01_…` reads 1301.5–1318.9 nm off an HC slide.] |
| Q_L | 3500 | 2000–4200 | — | printed (talk) / read-off | LM-T via REF; `06_…` O6 |
| κ/2π (FWHM) at Q_L 3500 | 65.4 | 54.5–114 | GHz | derived | §1 |
| n_g | 4.2 | 3.9–4.33 | — | assumed / derived | REF; mode solve; `01_…` |
| FSR | 1.80 | 1.55–2.8 (8.7–16 nm) | THz | design printed; range derived | REF, `06_…` O8 |
| κ_e/κ | 0.5 | 0.35–0.65 | — | assumed | near-critical (28–30 dB notches, G25) |
| κ_d/κ | 0.01 | 0.003–0.05 | — | derived (uncertain) | §8.1 |
| s (d ln(κ_e/κ_i)/dλ) | 1.48 | 0.5–2 | %/nm | derived (fit to read-off) | §3.3 |
| r_bs (per round trip) | 0.005 | 0–0.03 | — | assumed; 0.0124 printed for an AIM ring | Bahadori 2018; Li 2016 |
| K_T | 86 | 62–86 | pm/K | printed (G26 "~") / derived | §6.1 |
| K_SH (self-heating gain) | 405 | 160–1400 | pm per mW absorbed | assumed (= heater gain) | §6.3 |
| η_abs | 0.8 | 0.5–0.95 | — | assumed central; 0.63 inferred (de Cea) | §6.2 |
| τ_self-heat | 1 | 0.28–10 | µs | assumed; others printed | §6.3 |
| Δλ(V) charge-model A | 137.6 | ±10 % | pm V^−½ | derived (G25 read-off fit, V_bi = 0.9 V) | §5.3 |
| Efficiency at 1 V / 3.75 V | 50 / 32 | — | pm/V | derived | §5.3 |
| q (loss/index modulation) | 0.06 | 0.03–0.15 | — | derived; 0.13 printed (Müller, 1550 nm) [corrected: wavelength not printed in Müller's main text; their fit is 2× their bulk-silicon estimate] | §5.4 |
| C_j | 15 | 6–30 | fF | assumed (other devices printed) | Huang 2026; Müller 2014 |
| R_s | 100 | 15–250 [corrected: 15–390] | Ω | assumed | same [corrected: Huang Table 1 includes 385.9 Ω, §5.5] |
| RC pole implied by G25 30.9 GHz | 35 (x = 0.58) / 25 (x = 0.81) | 25–35 | GHz | derived | §5.6 |
| Optical 3 dB bandwidth at x = 1/√3 | 55.2 | 51–59 (q) | GHz | derived | §5.6 |
| β_TPA | 0.9 | 0.57–2.1 | cm/GW | printed range | Lin 2007; Bristow 2007 |
| n₂ | 4e-14 | 3–5e-14 | cm²/W | printed range | Bristow 2007 |
| τ_fc (reverse bias) | 30 | 10–100 | ps | assumed | `02_…` §8 |
| A_eff | 0.094 | 0.07–0.15 | µm² | derived | textbook Ch10 |
| Bistability threshold P_abs,max | 0.71 | 0.3–1.5 | mW | derived (K_SH, FWHM) | §6.4 |
| I_mon operating | 15.5 | ±0.5 | µA | read-off | LM26-P Fig. 3(b) |

---

## 12. Disagreements with the textbook

1. **The "photon-lifetime pole" (Ch06 eq. 6.20 and the bandwidth figure, `src/chapters/Ch06.tsx` about lines
   3641–3940).**
   - *Textbook:* f_3dB,opt = 1/(2πτ_p) = f/Q = 65 GHz ("the optical bandwidth of a ring modulator equals its
     linewidth"). The measured 30.9 GHz is the junction RC pole, and the two are cascaded to give 26.3 GHz
     total.
   - *Our view:* the modulation response follows the **field**, not the energy. Its poles are −κ/2 ± iΔ and it
     has a zero at −(κ_i + κ_d). At zero detuning the pole is at FWHM/2 (32.7 GHz). At the maximum-slope bias
     the optical-only bandwidth is 0.84 FWHM = 55 GHz, and it grows with detuning, with peaking. A measured
     EO S21 is already RC × optics. Cascading it with a further optical pole double counts: if 30.9 GHz was
     measured at x = 0.58, the RC pole is about 35 GHz, not 30.9.
   - *Evidence:* the closed form above, verified against a TCMT ODE (`03_checks/tcmt_checks.py`); Karimelahi
     & Sheikholeslami 2016; Müller et al. 2014; the PhotonForge example (1/(2πτ_a) = FWHM/2 at zero
     detuning). Karimelahi's thesis also states that the 1/f² = 1/f_Q² + 1/f_RC² rule is inaccurate, because it
     ignores the detuning and the coupling condition.
2. **"Bandwidth rises with reverse bias; a photon lifetime cannot produce this rise" (Ch06, same section).**
   - *Our view:* at a fixed laser wavelength the bias also moves the detuning (about Δx = 0.27 per volt). The
     optical bandwidth depends strongly on detuning (38.7 → 95 GHz for x = 0.3 → 1). The C_j(V) explanation is
     plausible but not unique unless the laser was re-centred at each bias.
   - *Evidence:* §5.6 table. A pure abrupt-junction RC predicts +23.5 % from −1 to −2 V at V_bi = 0.9 V, against
     +14.6 % measured (30.9 → 35.4 GHz) (derived). [corrected: 30.9 GHz is the population figure at −1 V and
     35.4 GHz is one plotted device at −2 V, so the ratio mixes two measurements. For the same plotted device the
     textbook uses 31.8 GHz at −1 V (Ch06 `EO_BW_PLOTTED_1V_GHZ`, read-off), giving +11.3 %; with the paper's own
     caption value of 34.1 GHz at −2 V (REF note) it is +7.2 %. The +23.5 % abrupt-RC prediction (√(2.9/1.9))
     is verified. The conclusion strengthens: the measured rise is half or less of the pure-RC prediction.]
3. **η_abs = 0.3 central (Ch10, "η_abs is an assumption", about lines 900–960; Ch17 eq. 17.7).**
   - *Our view:* the central value should be about 0.8 (0.5–0.95). Self-heating at G25 power is then 2–3 times
     the book's worked numbers.
   - *Evidence:* de Cea 2019 Table 1 (19 of 30 GHz is absorption, 0.63); Zheng 2012; the loss budget
     Q_i ≈ 7000 → 125 dB/cm, far above passive loss (agrees with `06_…` H1).
4. **"Γ = 1 caps the shift at 58 pm/K" (REF comment on `dLambdaDT_pmK`, `src/physics/index.ts` lines
   107–117).**
   - *Our view:* ∂n_eff/∂n_Si is 1.084 for a 400 × 200 nm strip at 1310 nm, so it exceeds 1. K_T is 62–73 pm/K
     for dn_Si/dT of 1.86–2.2e-4. The real ceiling is λ(dn/dT)/n_g,Si ≈ 69–77 pm/K. The gap to 86 pm/K narrows
     but stays open.
   - *Evidence:* local mode solve (`03_checks/thermooptic_modesolve.py`); `02_materials.md` §3; `06_…` T4.
5. **TPA coefficient range "0.5 to 1 cm/GW between 1.3 and 1.6 µm" (Ch10, `BETA_TPA_CM_GW = 0.7`, about lines
   240–255).**
   - *Our view:* the published range at about 1.3 µm is 0.57 ± 0.17 (Lin 2007, 1315 nm) to 2.1 ± 0.4 cm/GW
     (Bristow 2007, 1220 nm). That is a factor-of-3 disagreement between groups, and the upper end is twice the
     book's range.
   - The textbook's conclusion (TPA is negligible for the modulator) still holds (§7.2).
   - *Evidence:* Bristow et al. APL 90, 191104 (printed); Lin et al. APL 91, 021111 (printed).
6. **Notch depth against heater drive "because the coupler's strength depends on wavelength" (Ch06).**
   - The direction is right. The size, though, needs s ≈ 1.5 %/nm in κ_e/κ_i (fit), and a straight point
     coupler gives about 0.6–1.0 %/nm (crude estimate).
   - Model the slope as a fitted parameter. Do not assume it all comes from κ_e.
   - *Evidence:* §3.3. This is a refinement, not a reversal.
7. **Single-ring, single-line optics in the plant chain (Ch17 steps 1–6).**
   - *Our view:* G26's 2.8 THz bank exceeds the FSR. Multi-order evaluation (Airy, or TCMT with (−1)^m drop
     signs) is needed for drop-monitor crosstalk and initialization sweeps.
   - *Evidence:* §2.7 (agrees with `06_…` O10, S6).

---

## 13. Verification scripts (derived numbers)

All run under `tools/memguard.py` within the stated memory.

- `03_checks/tcmt_checks.py` (numpy and scipy, < 0.2 GB): TCMT against Airy, backscatter, small-signal
  closed form against ODE, the pole–zero formula check, bandwidth and peaking against detuning and coupling,
  nonlinear magnitudes, the G25 bias and extinction fits, and the adiabatic error.
- `03_checks/multimode_vs_airy.py`: the multi-order TCMT sum with and without the (−1)^m drop sign.
- `03_checks/thermooptic_modesolve.py` (tidy3d local mode solver, 0.7 GB): n_eff = 2.491, ∂n_eff/∂n_Si =
  1.084, ∂n_eff/∂n_ox = 0.234 for 400 × 200 nm at 1310 nm. Grid 12.5 nm aligned to interfaces, no subpixel
  averaging.
- `03_checks/coupler_dispersion_modesolve.py` (1.5 GB): coupled-strip supermodes at gaps of 150, 200 and
  250 nm. **Not converged in gap**; used only for the 0.5–1.3 %/nm range. The single strip gives n_g = 4.33.

---

## 14. Open questions

1. Lightmatter's κ_d/κ (drop coupling) and the monitor photodiode responsivity. The 15 µA reading and the
   GF-PDK "about 5 %" conflict (§8.1).
2. The detuning at which G25's 30.9 GHz and G26's 66.8 GHz bandwidths were measured, and Q_L of the G26
   device. Without these, the RC pole cannot be separated from the optics (§5.6).
3. η_abs and the ring's self thermal impedance Z_rr(s), and whether the heater is in the ring (G25 slide says
   in-ring resistive). This sets self-heating gain and bistability margin.
4. Backscatter r_bs of the MRM and the reflection of the drop-port photodiode, both relevant to the
   bidirectional link.
5. Whether s (extinction against resonance position) comes from κ_e, from κ_i, or from both, and the sign
   (under- or over-coupled). This needs phase data or bias-dependent notch derivatives (Huang 2026, §V.2).
6. Lightmatter's junction type (PN or PIN), C_j, R_s and driver output resistance. Only the shift ladder is
   public.
7. β_TPA at 1310 nm: a factor of 3 between groups. It is irrelevant to the modulator conclusion but matters
   for the higher-Q receiver filter.
8. The temperature dependence of K_T across 0–105 °C, and why the printed ~86 pm/K exceeds the 69–77 pm/K
   ceiling (open in `02_…`, `06_…`).
9. The time constants of ring self-heating against heater heating on 45SPCLO (`04_…`).

---

## 15. References

Lightmatter (generation-tagged):
- [LM26-P] R. Anand et al., "16-Wavelength 800-Gbps Bidirectional Link over Single-Mode Fiber Using Microring
  Transceivers," OFC 2026, M4B.3 (local PDF; read in full).
- [LM-P], [LM-T], [LM26-S]: via the textbook `src/physics/index.ts` and `docs/SOURCES_AND_GENERATIONS.md`.

TCMT, rings and backscatter:
- H. A. Haus, *Waves and Fields in Optoelectronics*, Prentice-Hall (1984).
- B. E. Little, S. T. Chu, H. A. Haus, J. Foresi, J.-P. Laine, "Microring resonator channel dropping filters,"
  J. Lightwave Technol. 15(6), 998 (1997).
- B. E. Little, J.-P. Laine, S. T. Chu, "Surface-roughness-induced contradirectional coupling in ring and disk
  resonators," Opt. Lett. 22(1), 4 (1997).
- T. J. Kippenberg, S. M. Spillane, K. J. Vahala, "Modal coupling in traveling-wave resonators," Opt. Lett.
  27(19), 1669 (2002).
- S. Fan, W. Suh, J. D. Joannopoulos, "Temporal coupled-mode theory for the Fano resonance in optical
  resonators," JOSA A 20(3), 569 (2003).
- W. Bogaerts et al., "Silicon microring resonators," Laser Photonics Rev. 6(1), 47 (2012).
- A. Li, T. Van Vaerenbergh, P. De Heyn, P. Bienstman, W. Bogaerts, "Backscattering in silicon microring
  resonators: a quantitative analysis," Laser Photonics Rev. 10(3), 420 (2016) (bibliographic record only).
- M. Bahadori, S. Rumley, Q. Cheng, K. Bergman, "Impact of backscattering on microring-based silicon photonic
  links," IEEE Optical Interconnects Conf. (2018), doi:10.1109/OIC.2018.8422024 (read in full).

Dynamics and small-signal:
- W. D. Sacher, J. K. S. Poon, "Dynamics of microring resonator modulators," Opt. Express 16(20), 15741 (2008)
  (abstract only; full text behind a bot check).
- W. D. Sacher et al., "Breaking the cavity linewidth limit of resonant optical modulators," arXiv:1206.5337
  (2012).
- J. Müller et al., "Optical peaking enhancement in high-speed ring modulators," Sci. Rep. 4, 6310 (2014)
  (read in full).
- B. Pile, G. W. Taylor, "Small-signal analysis of microring resonator modulators," Opt. Express 22(12), 14913
  (2014) (abstract).
- H. Yu et al., "Trade-off between optical modulation amplitude and modulation bandwidth of silicon micro-ring
  modulators," Opt. Express 22(12), 15178 (2014) (abstract).
- S. Karimelahi, A. Sheikholeslami, "Ring modulator small-signal response analysis based on pole-zero
  representation," Opt. Express 24(7), 7585 (2016); S. Karimelahi, PhD thesis, University of Toronto (2016),
  Ch. 3 (read).
- R. Dubé-Demers et al., "Analytical modeling of silicon microring and microdisk modulators with electrical
  and optical dynamics," J. Lightwave Technol. 33(20), 4240 (2015).
- A. Geravand et al., "Ultrafast coherent dynamics of microring modulators," Nat. Photon. 19, 740 (2025).

Compact-model implementations:
- C. Sorace-Agaskar, J. Leu, M. R. Watts, V. Stojanović, "Electro-optical co-simulation for integrated CMOS
  photonic circuits with VerilogA," Opt. Express 23(21), 27180 (2015).
- J. Rhim, Y. Ban, B.-M. Yu, J.-M. Lee, W.-Y. Choi, "Verilog-A behavioral model for resonance-modulated silicon
  micro-ring modulator," Opt. Express 23(7), 8762 (2015).
- V. Saxena, M. J. Shawon, "Compact modeling and rapid simulation of silicon photonic micro-disk and ring
  modulators," arXiv:2311.16174 (2023).
- D. Coenen et al., "Coupled dynamic thermo-optical analysis and compact modelling of self-heating in ring
  modulator," PSC 2023; D. Coenen, M. Kim, H. Oprins, "Electro-opto-thermal dynamic compact model for ring
  modulator arrays," IEEE TCPMT 15(5), 1006 (2025) (records only).
- Z. Huang, Y. Yuan, Y. Peng, S. Palermo, M. Fiorentino, R. G. Beausoleil, "Co-design of silicon microring
  modulator beyond 200 Gb/s per lane …," arXiv:2609.36690 (29 Sep 2026; not peer-reviewed).
- Ansys Optics help center: "Optical Ring Modulator (RING) – INTERCONNECT Element"; "Optical Nonlinear Ring
  Modulator (NL_RING) – INTERCONNECT Element"; "ring_modulator – CML Compiler Model" (read through search
  excerpts; the pages are behind a bot check).
- Flexcompute PhotonForge docs, "Optical Peaking in a Microring Modulator" example.

Nonlinear and thermal:
- T. J. Johnson, M. Borselli, O. Painter, "Self-induced optical modulation of the transmission through a high-Q
  silicon microdisk resonator," Opt. Express 14(2), 817 (2006) (read in full).
- M. Borselli, T. J. Johnson, O. Painter, "Accurate measurement of scattering and absorption loss in
  microphotonic devices," Opt. Lett. 32(20), 2954 (2007).
- M. de Cea, A. H. Atabaki, R. J. Ram, "Power handling of silicon microring modulators," Opt. Express 27(17),
  24274 (2019) (read in full, including Table 1).
- M. de Cea, A. H. Atabaki, R. J. Ram, "Energy harvesting optical modulators with sub-attojoule per bit
  electrical energy consumption," Nat. Commun. (2021) (abstract).
- X. Zheng et al., "Enhanced optical bistability from self-heating due to free carrier absorption in substrate
  removed silicon ring modulators," Opt. Express 20(10), 11478 (2012) (abstract).
- T. Carmon, L. Yang, K. J. Vahala, "Dynamical thermal behavior and thermal self-stability of microcavities,"
  Opt. Express 12(20), 4742 (2004).
- A. Arbabi, L. L. Goddard, "Dynamics of self-heating in microring resonators," IEEE Photonics J. 4 (2012)
  (abstract, via `04_…`).
- B.-M. Yu, M. Shin, M.-H. Kim, L. Zimmermann, W.-Y. Choi, Chin. Opt. Lett. 15(7), 071301 (2017) (abstract via
  search; title not verified).
- A. D. Bristow, N. Rotenberg, H. M. van Driel, "Two-photon absorption and Kerr coefficients of silicon for
  850–2200 nm," Appl. Phys. Lett. 90, 191104 (2007) (read).
- Q. Lin et al., "Dispersion of silicon nonlinearities in the near infrared region," Appl. Phys. Lett. 91,
  021111 (2007) (abstract via search).
- M. Nedeljkovic, R. Soref, G. Z. Mashanovich, IEEE Photonics J. 3(6), 1171 (2011) (coefficients from the
  Tidy3D source); R. A. Soref, B. R. Bennett, IEEE J. Quantum Electron. 23(1), 123 (1987).
- J. T. Robinson, K. Preston, O. Painter, M. Lipson, "First-principle derivation of gain in high-index-contrast
  waveguides," Opt. Express 16(21), 16659 (2008).
- S. M. Sze, K. K. Ng, *Physics of Semiconductor Devices*, 3rd ed., Wiley (2006).

Intracavity detection and stabilization:
- K. Padmaraju, D. F. Logan, X. Zhu, J. J. Ackert, A. P. Knights, K. Bergman, "Integrated thermal
  stabilization of a microring modulator," OFC/NFOEC (2013) (abstract).
- H. Jayatilleka et al., "Wavelength tuning and stabilization of microring-based filters using silicon
  in-resonator photoconductive heaters," Opt. Express 23(19), 25084 (2015); "Photoconductive heaters enable
  control of large-scale silicon photonic ring resonator circuits," Optica 6, 84 (2019).
- C. Sun et al., "A 45 nm CMOS-SOI monolithic photonics platform with bit-statistics-based resonant microring
  thermal tuning," IEEE JSSC 51(4), 893 (2016) (record only; its drop-port monitoring scheme not confirmed
  here).

---

## Verification log

Independent check, 2026-10-09. Method: every formula re-derived by hand, and every number recomputed in
fresh scripts written for this check (numpy/scipy, < 0.1 GB, not the `03_checks/` scripts). Every citation was
re-opened at the primary source where reachable: the G26 PDF at 400 dpi, the de Cea, Bahadori, Bristow and
Johnson full texts, Müller via Europe PMC, abstracts through Semantic Scholar, arXiv and Crossref, and the
tidy3d source in `.venv`. Verdicts: **confirmed** (the claim survives the attempt to refute it),
**corrected** (it was wrong or mis-cited, and the text above is fixed), **partly confirmed** (the core holds,
but part of the claim could not be reached).

| # | Claim (section) | Verdict | Evidence / correction |
|---|---|---|---|
| 1 | Small-signal H(s): poles −κ/2 ± iΔ, zero −(κ_i+κ_d), DC gain −∂T/∂Δ (§0.2, §5.6) | confirmed | Hand derivation gives a numerator ∝ 2iΔ(s + κ − κ_e). An independent 2×2 linearization agrees to 2e-15 over 200 random cases. The Karimelahi & Sheikholeslami abstract (third order; zero depends on detuning sign) is consistent. |
| 2 | Bandwidth/peaking table: 0.84 FWHM = 55.2 GHz at x = 1/√3; 1.0 dB at x = 0.813; 5.1 dB over-coupled; q rows (§5.6) | confirmed | All 9 rows reproduced to the printed digit. x = 0.813 is the −4.00 dB through point. |
| 3 | Implied RC pole 34.8 / 25.1 GHz from G25 30.9 GHz (§5.6) | confirmed | 34.83 / 25.15 GHz, assuming critical coupling and a single-pole RC. |
| 4 | Bistability threshold K_SH·P_abs,max ≥ (4√3/9)·FWHM_λ, onset at u = 1/√3; 0.71 mW; P_in 0.79 / 1.13 / 2.4 mW (§0.3, §6.4) | confirmed | Fold condition (1+u²)²/(2u) has its minimum 8√3/9 at u = 1/√3. The numbers depend on the assumed K_SH = 405 pm/mW. |
| 5 | TCMT against Airy table; the (−1)^m drop-sign rule; 7-mode sum within 0.1 dB (§2.7) | confirmed | Reproduced to ±0.02 dB. Analytic identity Σ(−1)^m/(x−m) = π/sin(πx) gives exactly the +3.92 dB at FSR/2; the unsigned sum is π·cot(πx) = 0 there. |
| 6 | Backscatter: g = r_bs/T_rt, splitting r_bs·FSR/π, reflection −31.6 / −19.4 / −12.2 dB; Bahadori r = 0.0124 (§2.8) | confirmed, with one correction | Mapping matches Bahadori's ω₀ ± (FSR/ξ)·r (full text). Their ring: AIM, 5 µm, 400 × 220 nm, 1550 nm, Q ≈ 16 000. **Corrected:** "split only when 2\|g\| > κ" is wrong. A dip appears at 2\|g\| > 0.49κ (drop) or 0.67κ (critical through), consistent with Bahadori's observed split at 2\|g\|/κ ≈ 0.71. The model conclusion is unchanged. |
| 7 | η_abs = 0.63 from de Cea 2019 Table 1 (γ_lin 19 GHz, γ_rad 11 GHz); γ_th 2.4 MHz; best OMA 1.65 mW measured against 4.2 mW model; τ = 10 ps; τ_fc 0.1 ns (§0.4, §6.2, §6.3, §6.4) | confirmed | Table 1 read from the open-access PDF. The device is 1545.2 nm, not 1310 nm. |
| 8 | NSM 1.3 µm coefficients; Soref–Bennett 1.3 µm; ~10 % Δn gap; q = 0.10 / 0.04 / 0.06 (§5.2, §5.4) | confirmed | Matches tidy3d's tabulated NSM row digit for digit. Gap 7 % (e) / 11 % (h). q recomputed. |
| 9 | Müller et al. 2014: q = 0.13 "somewhat above" bulk, "C band"; C_j 17 fF, R_s 215 Ω, 43 GHz, Q 3500–5000 (§5.4, §5.5) | corrected | Müller prints that 0.13 is "higher (2×)" than the bulk estimate. The main text prints no operating wavelength (10.3 µm IME/OpSIS ring), so "C band" / "1550 nm" is assumed. C_j, R_s, 43 GHz and Q are confirmed. |
| 10 | Charge-control fit A = 137.6 pm V^−½, rms 4.1 pm; 72.5 / 49.9 / 40.4 / 31.9 pm/V; m = ⅓ and free-V_bi rows (§0.6, §5.3) | confirmed | Reproduced exactly from REF `MOD_SHIFT_PM`. REF documents the abscissa as reverse bias. |
| 11 | Extinction against resonance fit: s = 1.48 %/nm, λ_c ≈ 1297.7 nm, rms 1.7 dB (§0.5, §3.3) | confirmed | Multi-start fit gives 1.48 %/nm, 1297.73 nm, 1.66 dB. A crude point-coupler estimate (γ = k₀√(n_eff² − n_ox²) with n_eff dispersion) gives about 0.8 %/nm at a 200 nm gap, consistent with the 0.6–1.0 %/nm stated. |
| 12 | G26 printed facts: ~15 GHz/°C, 8 MRMs on one bus, 50 Gb/s NRZ PRBS9, 56 Gb/s eye, ±20 GHz, 808 °C/s, 25 to >105 °C, 100 ms, 400 GHz grid centred at 1310 nm offset 200 GHz, monitor at the MRM drop port, 15.5 ± 0.5 µA (§1, scope) | confirmed, with corrections | All printed items are present in the PDF, and the monitor read-off is confirmed at 400 dpi. **Corrected:** ~15 GHz/°C is a generic introduction statement about MRMs, not a device measurement. "Add-drop" and "at each MRM" are inferred. |
| 13 | λ₀ range "1301–1316 nm (G26 bank), printed, LM26-P" (§11) | corrected | Not printed in G26. The printed grid implies 1301.5–1318.6 nm (derived); `01_…` reads 1301.5–1318.9 nm off an HC slide. |
| 14 | Huang et al. 2026: C_j 6.6–20.3 fF, R_s 16–247 Ω (§5.5, §11) | corrected | Table 1 also lists C_j 6.0 fF and R_s 385.9 Ω (and 68.1 / 35.4 Ω), so the span is 6.0–20.3 fF and 16–386 Ω. The ~20 fF / ~16 Ω entries are calculated, not measured, in the source. Read through a fetch summary of the arXiv HTML, so treat these values as moderately confident. |
| 15 | Huang et al. 2026: 160 pm/mW absorbed, 0.84·P_bus/(1+x²), 80 pm/K (II.1) and 54.6 pm/K (IV.2), 1.8 / 2.6 mW thresholds, startup 6.9 / 16 mW → 684 pm blue / 142 pm red, FCD 5.7 / 0.1 pm against 668 pm, 3 pm at 100 µA·1 kΩ·33 pm/V; Q_L 3700 / 2500 at 1310 nm (§4, §6, §7, §8) | confirmed | All found in arXiv:2609.36690v1 (exists, 29 Sep 2026, authors as listed). |
| 16 | "+14.6 % measured (30.9 → 35.4 GHz)" bandwidth rise with bias (§12.2) | corrected | It mixes the population figure (30.9) with one plotted device (35.4). Same device: 31.8 → 35.4 GHz = +11.3 % (+7.2 % with the caption's 34.1 GHz). The +23.5 % RC prediction is confirmed. |
| 17 | Johnson 2006 equation forms for κ_TPA, κ_FCA, dN/dt and the FC/Kerr shift (§7.1) | partly confirmed | Loss and generation terms match Johnson Eqs. (4), (9), (25). **Corrected:** Johnson Eq. (15) and de Cea Eq. (1) divide by n_Si with mode-averaged ΔT and N. The report's Γ/n_g form is right only if Γ = ∂n_eff/∂n_Si. With Γ = 0.85 the FCD and Kerr shifts are about 20–25 % low, still negligible. |
| 18 | TPA: Lin 0.57 ± 0.17 cm/GW at 1315 nm against Bristow 2.1 ± 0.4 cm/GW at 1220 nm; n₂ (3–5)e-14 cm²/W (§7.1) | confirmed | Bristow full text gives 2.1 ± 0.4 cm/GW and n₂ = 4.7 ± 2.0e-14 cm²/W at 1220 nm. Lin's value is from a search excerpt of the authors' PDF (journal page not reached). The two values are at different wavelengths. |
| 19 | Sacher & Poon 2008 abstract (coupling modulation approaching FSR); Sacher et al. arXiv:1206.5337 (40 GHz, > 6× linewidth); Müller dephasing condition π/2(ω_r − ω₀) < τ_a (field lifetime) (§5.6) | confirmed | Abstracts re-read; Müller text quoted through Europe PMC. |
| 20 | Anchor arithmetic: f₀ 228.85 THz, FWHM 65.39 GHz = 374.3 pm, 1/κ 2.43 ps, FSR 1.8 THz = 10.30 nm, L 39.66 µm, F 27.5, build-up 8.76, U 4.87 fJ/mW, 15 GHz/°C = 85.9 pm/K, Q_i 7000 → 28.8 cm⁻¹ = 125 dB/cm, NSM 8.7 / 31.3 cm⁻¹, Airy FWHM 65.457 GHz, CV²f 0.25 mW (§1, §2.6, §6.2) | confirmed | All recomputed. F = 27.5 (not 27.6) at exactly 1.800 THz, which is rounding only. |
| 21 | Adiabatic-elimination error 5e-9 at 2000 °C/s (§0.9, §9.2) | confirmed | 4.5e-9. The G26 printed ramp is 808 °C/s, so 2000 °C/s is conservative. |
| 22 | Drop-coupling inference κ_d/κ ≈ 0.4–1.1 % and 22 % for a "5 % per round trip" reading (§8.1) | confirmed | The arithmetic holds. The premises (0.9 A/W responsivity, 1–2.5 mW at the ring, x ≈ 0.6) remain assumed. |
| 23 | Zheng et al. 2012: ~100 µW bistability, dopant FCA dominates, 100–200 kHz (§6.2, §6.3) | confirmed | Abstract. |
| 24 | Ansys NL_RING defaults (§4) | partly confirmed | 0.28 µs, 16 pJ/K, 1.86e-4 /K, confinement 1 and both effects off by default were re-confirmed by search excerpt. 45 ns and dn/dN were not reached. |
| 25 | Citation records: Karimelahi & Sheikholeslami OE 24(7) 7585; Saxena & Shawon arXiv:2311.16174; Padmaraju OFC/NFOEC 2013 (§4, §8.2, §15) | confirmed | Crossref, arXiv and Semantic Scholar. Padmaraju et al. also published a journal version: Opt. Express 21(12), 14342 (2013). |

Not checked here (outside the 10–15 highest-impact claims or owned by other files): K_T mode-solve values
(61.7 / 66.3 / 72.8 pm/K were recomputed from the stated ∂n_eff/∂n weights, but the mode solve itself was not
re-run), the coupler mode-solve range, Xilinx/AMD patent setpoint, Kramnik photocurrent, Yu et al. 2017, Coenen
et al., and Li et al. 2016 beyond its bibliographic record.
