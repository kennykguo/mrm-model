# 07 · Design critique of ARCHITECTURE.md (four reviews, merged)

Date: 2026-10-09. Subject: `docs/ARCHITECTURE.md` (draft contracts for the MRM plant).

This file merges four independent reviews, removes duplicates, resolves the conflicts between them, and ranks the
result. Tags used below:

- **[PHY]** silicon-photonics device physicist (optics, thermo-optics, junction, self-heating, monitor);
- **[CTRL]** control and FPGA engineer (signals, timing, controller API, runtime speed on the 16 GB M2);
- **[ARCH]** scientific-software architect (config tree, derive(), batching, numba, extensibility, reproducibility);
- **[VAL]** validation engineer (how anyone will know the model is accurate);
- **[R01] [R02] [R04] [R05] [R06]** the parallel research notes in this folder (`01_lightmatter_public.md`,
  `02_materials.md`, `04_thermal_and_control.md`, `05_tools_smoke/out/`, `06_textbook_audit.md`). They appeared while
  the reviews were running, and several of them settle conflicts between the reviewers.

Numbers marked **(checked)** were recomputed for this merge. Priorities: **P0** = the model is wrong or unusable
without it; **P1** = significant accuracy or usability gain; **P2** = nice to have.

## Summary

**Keep.** The exact, phase-based add-drop ring solved per laser line (FSR aliasing, order hops and wrong-line locks
then emerge on their own); codes in and codes out, with `Truth` kept apart from the controller; the (B, N) batch
dimension; provenance words on every number; committed extraction artefacts; the exact zero-order-hold Foster
thermal model; one config tree.

**Fix before any controller is written against it (the P0 list).**

1. Self-heating is the dominant nonlinearity at the published optical powers: 3 to 7 times above the bistability
   threshold at +4 dBm. The draft makes it optional, inherits a heat fraction about 3 times too low, and couples it
   explicitly. At dt = 1 µs the explicit coupling oscillates numerically.
2. `derive()` does not enforce closure. dλ/dT, dn_eff/dT and the heater efficiency are independent leaves that can
   disagree by up to 1.7 times, and sampled parameters never pass through `derive()`.
3. Time is a float, actuator latency has three owners, and the controller sees one ADC code per tick. It has no bias
   actuator and no data-path signals.
4. The runtime (Python on every tick, full traces) cannot run the 30.4 h soak or a Monte Carlo study in 16 GB.
   Decimated traces hide the short excursions that the acceptance test counts.
5. The plan fits every published number. Nothing would then be left to validate against, and the plan counts
   solver-against-solver agreement as validation.

**Build order** (so the team has a plant within weeks, not months, as [ARCH] warns):

1. P0-3, P0-4 (params, relations, realize, uncertainty); P0-5 (freeze the held-out targets); P0-6, P0-7 (timing,
   channels); the exact ring with the interim three-pole thermal model of P1-2 plus a sub-µs self-heating pole;
   P0-1, P0-2; a Python runner with the P0-8 accumulators; two baseline controllers (dither lock-in on the bias, and a
   drop-port set-point PI).
2. P0-9, P0-10 (compiled kernel, long runs); P0-11 (data path).
3. Extraction artefacts (thermal FEM ROM, coupling, junction LUT, modes for the 160 nm layer) then replace the
   assumed values one at a time. Each replacement goes through the calibration split of section 4.

---

## 0. Conflicts between the reviews, and the decisions

**D1. Heat-producing fraction η.** [PHY]: derive it from a loss budget, about 0.85 to 0.9. [VAL]: an interval of 0.3
to 0.9, anchored on de Cea et al. 2019 (0.63). Textbook: 0.3 assumed. **Decision: derive η from a loss budget
([PHY]). Nominal ≈ 0.85; epistemic range 0.5 to 0.95; 0.3 survives only as a named stress corner.** Reasons: Q_i ≈
2·Q_L ≈ 7000 to 8400 gives α_i ≈ 107 to 128 dB/cm (checked). GF prints 0.73 dB/cm for the 45SPCLO rib and 1.0 dB/cm
for the ridge [R01]. The strip bend radiates about 2e-4 dB/cm at R = 7.5 µm [R05 t01]. So η = 0.3 would need about 75
to 90 dB/cm of loss that does not heat, and no mechanism supplies that. [R06 H1] reaches the same conclusion. The de
Cea device has a different geometry, so it sets the low end, not the nominal. A rib-and-slab radiation solve
(P0-2) can reopen the low end.

**D2. Which thermo-optic quantity is independent, and its default.** [PHY]: derive dλ/dT from the solver's
dn_eff/dT(T) and put the printed 86 pm/K in a sibling preset. [VAL]: dλ/dT is the only input, with hypotheses H86,
H62.5 and H50. [ARCH]: relation groups, with the printed value as the default and the others as alternatives. [R04]:
keep 86 as the design value and sweep 60 to 90. [R06]: default about 70, with 86 as a corner. **Decision: dλ/dT at a
stated ring temperature is the single independent leaf ([VAL], [ARCH]). Its temperature shape comes from materials
data ([PHY], [R02], [R06]). The default hypothesis is H_solver; H86 is a required robustness corner; H50 is
dropped.** Reasons:
- The G26 "~15 GHz/°C" is an aside in the paper's introduction, with no method, thermometer or reference temperature
  [R06, R02].
- 86 pm/K lies above the materials ceiling for any Si-core, oxide-clad ring at 1310 nm: 71 pm/K at 300 K and 80 pm/K
  at 400 K, expansion included [R02 §3].
- The femwell FEM gives 66 to 69 pm/K at 22 °C and 74 to 77 pm/K at 105 °C, expansion included [R06 §2.7].
  Lightmatter's own bench equivalence (62.5 pm/K) and an imec measurement on O-band rings (62.54 pm/K [R04]) agree.
- The textbook's 50 pm/K used the power fraction 0.85 in place of the index weight ∂n_eff/∂n_Si = 1.086 [R06 T3].
  Its "58 pm/K cap at Γ = 1" is wrong; the real gap to 86 is 12 to 30 %, not 1.48 times.
- Rule 2 in §6 is amended accordingly (P1-14).

**D3. "Sibling preset" against "alternatives".** Draft rule 2: one sibling preset per disputed number. [ARCH]:
alternatives on the leaf. [VAL]: named corner presets. **Decision: alternatives on the leaf. Corner presets are
thin `extends:` files that only select alternatives.** Four disputes with 2 to 4 values each would otherwise need
64 sibling presets.

**D4. The action type.** [CTRL]: an `Action` dataclass. [ARCH]: a dict of named channels with port binding.
**Decision: named channels in Python. For compiled controllers, `realize()` generates a flat int64 layout from the
ChannelSpecs ([CTRL]'s ABI).** New actuators (bias, laser tuning, interleaver and polarization-controller phases)
then add channels and break nothing, and numba still sees fixed arrays.

**D5. Priority of the bias actuator.** [PHY] P1; [CTRL] and [ARCH] P0. **Decision: the API and the binding are P0
(P0-7). The bias physics (absolute α_FCA(V), η(V), the slow self-heating lead-lag) is P1 (P1-7).** Changing the
action type later breaks every controller; the physics can follow. [R04] adds that a heater dither cannot meet the
G26 aggressor and that a bias dither can, which makes the actuator central to novel controllers.

**D6. How the fast aggressor is specified.** [VAL]: a resonance trajectory at the ring. [CTRL]: aggressor power fed
into the thermal ROM. **Decision: power into the ROM ([CTRL]), with the waveform calibrated to a target stated in
resonance units at the victim ring ([VAL]). The referencing itself is a hypothesis: `resonance` (default) or
`temperature`.** Prescribing the ring temperature gives the sensors and the neighbouring rings the wrong
disturbance. A target in °C under a varied dλ/dT is circular if 808 °C/s was inferred from 15 GHz/°C. The target
shape is the slide-10 waveform [R01 D7]: T(t) from 25 to 113 °C, dT/dt a ±2200 °C/s sawtooth, period 130.5 ms. It is
not a single rate.

**D7. Drop-coupler strength.** [PHY]: calibrate q_drop from the 15 µA monitor current. [VAL]: 15 µA identifies only
a lumped product. **Decision: [VAL]. Only the lumped monitor gain is calibrated. [PHY]'s back-calculation (κ_d/κ ≈
0.4 to 1.2 %) and the GF PDK ring's "≈ 5 % drop-port coupling" [R01] together set a wide prior.** The 15 µA is a
product of line power, drop coupling, lock detuning and responsivity. The GF figure shows how far apart two
plausible readings can be.

**D8. ER against heater drive: fit all points, or hold some out.** [PHY], [ARCH]: fit or test all 8. [VAL]: hold out
1.4 and 1.5 V. **Decision: fit 0 to 1.2 V; hold out 1.4 and 1.5 V.** Otherwise no held-out test of the ER mechanism
remains.

**D9. relOMA as an acceptance number.** [PHY]: reproduce −2.1 dB at ≈ 1.05 V and −3.0 dB at ≈ 3.1 V within 0.3 dB.
[VAL]: exclude the absolute level, because its definition and the drive are unknown. **Decision: the definition is
printed. Hot Chips slide 16 gives "1 mW of laser into the MRM" [R01], so relOMA = OMA/P_in. The absolute level at
≈ 1.05 V becomes one calibration target for a single lumped modulation depth (CAL-14). The shape across bias and
temperature is held out (VAL-4).** A G25-like drive (45 to 65 pm at Q 3500 to 4000) reaches only −5.9 to −8.1 dB
(checked). The printed −2.1 to −3.0 dB needs about 170 to 230 pm [R06 M6]. The bench's "0.9 Vpp" was the
pattern-generator swing, not the junction swing [R01 D3], so the G26 drive is genuinely unknown. A second reading,
"OMA in dBm at the actual input power", would close the gap at +4 dBm (−1.9 to −4.1 dBm, checked). It stays only as a
question to confirm (Q4).

**D10. Meep FDTD.** [VAL]: drop it or make it optional. [PHY]: use cylindrical Meep to check FSR within 0.3 %.
[ARCH]: Meep needs conda on macOS arm64. **Decision: optional verification of the bent-mode radius reference only,
not in CI and not a validation.** The smoke runs (t03) give n_eff,bend = 2.586, 2.555 and 2.538 at 20, 30 and
40 px/µm, so they are not converged. Richardson extrapolation gives 2.486 to 2.493 depending on the pair (checked).
That spread is larger than the 0.0025 gap between tidy3d and femwell that it was meant to settle.

**D11. Self-heating numerics.** [ARCH]: require dt ≤ min τ/5. [PHY], [VAL]: an implicit solve. **Decision:
implicit, with branch continuation.** With τ ≈ 0.3 µs, dt ≤ τ/5 costs 16 times more steps. The implicit multiplier
a/(1 + (1 − a)g) is stable at dt = 1 µs: 0.02 to 0.05, against −1.1 to −5.9 for the explicit form (checked).

**D12. The long soak.** [CTRL]: exact multi-tick stepping plus validated time compression. [ARCH]: a quasi-static
mode plus dynamic windows. **Decision: all of them, each with a role** (P0-10). The soak is one experiment, not two:
33 to 108 °C, about 90 cycles of 20.3 min, ≈ 0.124 °C/s, 30.4 h [R01 D5].

**D13. What a controller may know.** [CTRL]: `ControllerSpec.nominal` built from a `controller_knowledge` preset.
[ARCH]: an explicit knowledge list. [VAL]: a spec built only from the unsampled config. **Decision: all three
together.** The `controller_knowledge` preset is filtered by `controller_interface.knowledge`, and the spec must be
identical across seeds and uncertainty worlds.

**D14. Polarization scramble as a TX stimulus.** [CTRL]: PGA autoranging for the 7.7 dB swing. [VAL]: those are
RX-path monitors. **Decision: [VAL].** The Fig. 2(e) caption reads "monitoring photodetectors at the input of the
receive path". The 7.7 dB also ignores the ADC offset [R01 D8]. TX-monitor autoranging is motivated by connector
re-mates and laser power steps instead. The textbook's acceptance-table row should be corrected.

**D15. Scaling the thermal time constants.** Draft: a per-ring `time_scale` sigma. [ARCH]: die level. [VAL]: split
into fast and slow. **Decision: die-level, one draw per uncertainty world, split into fast and slow.** A per-ring τ
scale is undefined for cross-coupling residues, and the fast and slow poles come from different layers.

**D16. The heater anchor.** Draft: `heater_efficiency_nm_per_mw` (0.44). [PHY], [ARCH]: P_FSR = 25.45 mW.
**Decision: anchor P_FSR (a 60-die median). Efficiency = FSR/P_FSR is derived, so in pm/mW it inherits the FSR
hypothesis [R06 N7]. The printed 0.44 nm/mW becomes a closure check: it implies FSR = 11.2 nm.**

**D17. FSR.** [VAL]: [1.72, 1.88] THz. [PHY]: the solver predicts 1.47 to 1.57 THz. **Decision: default to window B
(1.72 to 1.88 THz), with window A (1.32 to 1.48 THz) and two stress hypotheses (P1-6).** [PHY]'s estimate used a
200 nm film, but G25 is printed as GF 45SPCLO, whose photonics layer is 160 nm [R01 D1]. The "≥ 14.1 nm" bound is
unsupported, because the traces are about 1 nm wide windows [R01 D2]. Lightmatter's own patent claims a modulator
FSR below 11.44 nm [R01]. At n_g 3.9 to 4.04, a ring of 7.5 µm radius (the "15 µm diameter") has an FSR of 1.58 to
1.63 THz, which puts its next order about 0.5 FWHM from another lane's line (BND-1, checked).

**D18. RL environment priority.** [CTRL] P1; [ARCH] P2. **Decision: P1, built after the compiled kernel.** RL needs
about 1e7 steps, and only the batched plant gives that.

**D19. Generic optical netlist.** [ARCH] P1, against [ARCH]'s own scope warning. **Decision: P2.** Controllers see
channels, not optics. The netlist can arrive with the second element without breaking them. Keep the line table
(P1-12) and the named thermal sites (P1-2) now.

**D20. Size of the monitor dark current.** [PHY]: a noticeable fraction of 15 µA. [R02]: a few nA × 21 is below 1 %.
[R06 S4]: up to 0.4 µA, a lock offset of about 6 pm. **Decision: model it with an Arrhenius law (E_a 0.35 to
0.38 eV [R02]) and I_d0 as an uncertainty of 1 to 30 nA at 25 °C (GF prints < 30 nA [R01]).** At the top of that
range it uses the whole 5 pm budget of a set-point lock, so it stays P1. [PHY]'s "doubling every 8 to 10 K" is too
steep; the data give ×1.6 to 1.9 per 10 K.

**D21. Waveguide cross-section.** The textbook and the smoke tests use 400 × 200 nm (an SMTAI render). [R01]: 45SPCLO
has a 160 nm photonics layer. **Decision: thickness is an uncertainty, {160 (default), 200} nm, and every extraction
is run for both.** It changes n_eff, n_g, the confinement, dλ/dT and the FSR-to-radius map.

**D22. Default thermal dynamics.** Textbook: 10 µs and 300 µs (25 %), crosstalk 0.1 and 0.03 with the same dynamics
as the self path. [R04]: imec measured 5 µs (heater node) and 10 µs (waveguide) for a metal heater above a 15 µm ring.
A two-pole fit is 3.2 µs (43 %) + 13.0 µs (57 %). Undercut rings need 4 poles out to 124 µs. Crosstalk is 1.3 to
2.6 % out to 300 µm, arriving with 4 to 9 ms poles. **Decision: the interim model uses [R04]'s numbers (P1-2), and
the textbook values become a pessimistic corner.** The textbook's 10 µs agrees with the imec waveguide τ. Its 300 µs,
25 % slow pole and its fast, 10 % crosstalk have no support.

---

## 1. P0 changes

### P0-1. Self-heating on by default, solved implicitly, with a defined start-up history

Sources: [PHY] P0, [VAL] P0, [ARCH] risk, [R04] §2.5 and §3.3.

**Change.**
1. Set `fidelity.self_heating: on` in every preset except `teaching_*`.
2. In `mrm/plant/plant.py` and in the compiled kernel, solve each substep implicitly. Take ring k in `bus_order`
   (its input is fixed by rings 1 to k − 1) and solve
   `T_k = T_k^other + Σ_j [a_j·x_kj + (1 − a_j)·r_j^opt·P_abs,k(T_k)]`, with `a_j = exp(−dt/τ_j)`.
   `T_k^other` holds the heater, neighbour, aggressor and base contributions, already advanced. `P_abs,k(T)` sums all
   lines, both circulation directions and every longitudinal order at that ring. Use a bracketed Newton or secant
   step (at most 10 iterations, numba), started from the previous root (branch continuation). Never use the
   previous substep's P_abs on the same-ring path. Cross-ring optical heating (`cross_opt_to_ring`) may lag one
   substep; `realize()` asserts that its per-substep gain is below 0.05.
3. The thermal ROM must carry the waveguide-core self-heating poles. `thermal_fem.py` puts the absorption source in
   the doped waveguide (mesh 50 nm or finer, time axis from 10 ns), with no heater-node lag [R04]. `self_opt_to_ring`
   must contain poles below 1 µs. Add the row "waveguide-core self-heating, 0.05 to 1 µs" to the table in §3. Keep
   `sim.dt_plant_s = 1e-6`.
4. Make the reference planes explicit. New keys: `optical_path.chip_input_loss_db` and
   `optical_path.bus_loss_db_per_mm`. `laser.power_per_line_dbm` is the power at the laser-module output (10 dBm per
   line into the TX in the G26 hardware budget [R01 QDSOA]). G25's +4 dBm is measured at the ring. OCP's +9 dBm is a
   planning midpoint at the ring.
5. `mrm.params.derive` adds `self_heating_gain_pm_per_mw = efficiency × Z_opt→ring(0)/Z_heater→ring(0)`, using the
   FEM DC ratio, which is at least 1 by reciprocity for an external heater [R06 H2]. It also adds
   `bistability_ratio = G_sh·P_abs,peak/(0.77·FWHM)`. `Config.table()` lists both, and a warning is raised when the
   ratio is above 1.
6. `Plant.reset()` reaches its state through a defined history. Add `sim.reset_protocol: laser_ramp | branch_low |
   branch_high`, default `laser_ramp`: the lasers turn on over at least 10 τ_slow with every heater at code_min.
7. Metrics: the "best achievable OMA" is taken over open-loop-stable points only (G_sh·∂P_abs/∂λ_r < 1). The
   unrestricted value is reported next to it.

**Tests** (`tests/test_self_heating.py`).
- (a) At +4 dBm, slow heater sweeps up and down show hysteresis. Its edges sit where G·∂P_abs/∂λ_r = 1, within 2 % of
  the HWHM. At −10 dBm there is no hysteresis.
- (b) The implicit solve at dt = 1 µs agrees with a 10 ns reference to better than 1 pm RMS on the stable branch. At
  +9 dBm the steady detuning changes by less than 1 % of the FWHM between dt and dt/8.
- (c) An open-loop hold at the OMA point, with the resonance blue of the laser, diverges at the linearized rate
  (G·P′ − 1)/τ_fast. No oscillation appears anywhere on the stable branch.
- (d) A +10 % laser-power step with the heater code held moves the resonance by
  Δλ_r = G·(∂P_abs/∂P_in)·ΔP_in / (1 − G·∂P_abs/∂λ_r), within 5 %.

**Why it matters for a controller.**
- The Lorentzian bistability threshold is G·P_abs,peak > (8/3√3)·HWHM = 0.77 FWHM ≈ 265 pm at Q 3800 (checked).
- G ≈ 350 to 880 pm per absorbed mW (0.35 to 0.44 nm/mW times a Z ratio of 1 to 2; [VAL]'s 5 K/mW × 86 pm/K =
  430 pm/mW agrees). The threshold is therefore 0.3 to 0.76 mW absorbed.
- At +4 dBm near critical coupling the ring absorbs about 2 mW, so it sits 3 to 7 times above threshold. [R04] cites
  an undercut microdisk that turns unstable at −14 dBm.
- The controller then faces four things.
  - Rings are fabricated blue of their line (7.6 nm), so a heater-up acquisition sweep approaches on the
    positive-feedback side and jumps. Initialization must start from the correct side [R04, DEC19].
  - The OMA point with the resonance blue of the laser is open-loop unstable.
  - The other side self-locks, which divides the heater-to-detuning gain by 1 + G|P′| (up to 4 to 7 times). Loop
    gain then depends on laser power and on the side. The GF loop locks on this blue edge of the drop peak [R01].
  - Laser power becomes a wavelength disturbance: 2.5 % per 24 h is about 15 to 30 pm, and laser turn-on is about
    1 nm.
- Numerics: a lagged coupling at dt = 1 µs has a multiplier a − (1 − a)g of −1.1 (τ 0.3 µs, g 1.2) to −5.9 (τ
  0.5 µs, g 7, +9 dBm) (checked). The resulting period-2 oscillation is something a developer would try to "fix" in
  the controller.

### P0-2. Heat fraction from a loss budget, not an assumed scalar

Sources: [PHY] P0, [VAL] P0 (part), [R06] H1. Conflict D1.

**Change.**
- Remove `ring.optics.absorbed_fraction`. Add `ring.optics.loss_budget: {alpha_scatter_db_per_cm (literature),
  alpha_bend_radiation_db_per_cm (solver: rib plus slab at the design radius, for the 160 and 200 nm films),
  alpha_other_abs_db_per_cm (heater doping, silicide, metal, surface states)}`.
- The junction file supplies the absolute `alpha_fca_per_m(V)` (P1-7), not values relative to 0 V.
- `derive()` computes:
  - α_i(V, T) = α_FCA(V)·(1 + c_T·ΔT) + α_other_abs + α_scatter + α_rad;
  - Q_i(V) = 2π·n_g/(λ·α_i);
  - η(V) = (α_FCA + α_other_abs)/α_i, evaluated per NRZ level and per line.
- `q_intrinsic` becomes derived or calibrated, with a stated reference bias and temperature.
- The epistemic range of η comes from the budget components: 0.5 to 0.95, nominal ≈ 0.85. A named stress corner,
  `g26_low_eta`, sits at 0.3.

**Tests** (`tests/test_loss_budget.py`). Q_i(V) from the budget equals the exact ring's a(V). With the doping set to
zero, η = α_other/(α_other + α_scatter + α_rad).

**Why.** Q_L of 3500 to 4200 with a notch near 28 to 30 dB means near-critical coupling, so Q_i ≈ 7000 to 8400 and
α_i ≈ 107 to 128 dB/cm (checked). Passive loss on this platform is about 1 dB/cm [R01], so the loss is mostly
free-carrier absorption in the doped junction, and that becomes heat. η = 0.3 would cut the self-heating gain 3
times and put the ring at the threshold instead of 3 to 7 times above it: a qualitatively different plant. A
bias-dependent η is also what produces the slow thermal side effect of the bias actuator.

### P0-3. One thermo-optic anchor, and closure that derive() enforces

Sources: [ARCH] P0, [VAL] P0, [PHY] P1, [R02] §3, [R06] §2. Conflicts D2, D16.

**Change.**
- Add `mrm/params/relations.py` with `Relation(name, members, solver)`; every solver uses NumPy broadcasting. Each
  group must have exactly one member that is null or tagged `derived`. If every member is given, `derive()` computes
  the residual and raises `OverdeterminedError`, unless the preset states `tolerance_rel`.
  - **R1, thermo-optic.** dλ/dT(T) = (λ/n_g)·[∂n_eff/∂T(T) + n_eff·α_L(T)], with members
    `ring.thermal.dlambda_dt_pm_per_k`, a new `ring.thermal.dlambda_dt_ref_ring_temp_c`, `ring.optics.dn_eff_dt_per_k`
    and `ring.optics.n_g`. Presets set dλ/dT; setting `dn_eff_dt_per_k` is a load error. The temperature shape is
    dλ/dT(T) = s₀·[1 + β·(T − T₀)], with β ≈ 1.4e-3 /K (silicon dn/dT at 1310 nm rises about 14 % per 100 K [R06
    §2.6]; Li 1980 fit in [R02 §12]). Delete `d2n_eff_dt2_per_k2` as an independent leaf. Set
    `fidelity.thermo_optic_order: 2` as the default, evaluated at the absolute mode-weighted ring temperature.
    Expansion is α_L ≈ 2.6e-6 /K at 300 K and 3.1e-6 /K at 375 K, clamped to the substrate, worth about 2 pm/K.
  - **R2, heater.** efficiency = FSR/P_FSR, and Z_heater→ring(0) = efficiency/dλ/dT(T_lock). Anchor
    `ring.heater.p_fsr_mw = 25.45` (printed wafer median). `heater_efficiency_nm_per_mw` becomes derived.
  - **R3, geometry.** m = round(n_eff,bent·2π·r_ref/λ_res), with the FSR taken from the phase model. Members: the
    radius, n_eff, n_g, `resonance_offset_nm` and an optional measured `ring.optics.fsr_nm`.
  - **R4, Q.** 1/Q_L = 1/Q_i + 1/Q_t + 1/Q_d, with an optional `q_loaded` target.
  - **R5, latency.** One owner per I/O channel (P0-6).
- The dλ/dT hypotheses are alternatives on the leaf (P1-14):
  - `H_solver` (default): s₀ from the extraction for the actual cross-section. About 66 to 69 pm/K at 25 °C and 74 to
    77 pm/K at 105 °C, expansion included, for the 400 × 200 nm strip and a 500 × 220 nm rib [R06]. It must be
    re-solved for the 160 nm layer (D21).
  - `H86`: the printed 86 pm/K, as a constant at the ring. It lies above the materials ceiling [R02].
  - `H86_sensor`: H_solver at the ring, with Lightmatter's °C read at a node that sees 0.77 to 0.86 of the ring's
    temperature rise. [R06] rates this the most plausible way to reconcile 86 pm/K.
  - Kept as plain alternatives, not corners: bench 62.5 (Lightmatter), imec 62.54 (measured, O-band [R04]), OCP 80.
    The textbook's 50 is kept as a documented, refuted value.
- Diagnostics: `implied_index_weight` (∂n_eff/∂n_Si needed to produce the chosen dλ/dT, against the ceiling
  n_g/n_g,Si), `t_ring_lock_c` and `t_amb_order_hop_c`.

**Tests.** Every preset satisfies every relation to 1e-12 after `realize()`. `test_derive_closure`: efficiency ×
P_FSR = FSR exactly. `test_dldt_perturbation_vs_resolve`: λ·Σᵢ wᵢ·(dnᵢ/dT)/n_g,i agrees with a finite-difference
re-solve to within 2 %. `test_index_weight_identity`: Σᵢ n_g,i·∂n_eff/∂nᵢ = n_g to 1e-4 [R06 §2.3].

**Why.** Today §5.1 moves the resonance with `dn_eff_dt_per_k`, while §5.2 scales the ROM with
`dlambda_dt_pm_per_k`. If they disagree as the REF values do (86 against 50), the realized heater efficiency and the
ambient sensitivity are off by 1.7 times, and nothing flags it. dλ/dT sets:
- the ambient disturbance gain: 2000 °C/s is 172 nm/s at 86 pm/K and 130 to 155 nm/s under H_solver (25 to 105 °C);
- the locked ring temperature: 113 °C (H86) against about 132 °C (H_solver) for the 7.6 nm blue offset (checked);
- the 5 pm loop-lag budget at 2000 °C/s: 29 µs at 86 pm/K, 38 µs at 66 pm/K [R06 C5];
- the order-hop margin of BND-2.

The heater gain and the self-heating gain do not depend on dλ/dT, because both are anchored on measured efficiency.

### P0-4. Parameter realization: derive after sampling, and keep epistemic worlds apart from per-ring variation

Sources: [ARCH] P0, [VAL] P0. Conflict D15.

**Change.**
- Replace `Plant(cfg, batch, seed)`, `derive(cfg)` and `sample(cfg, B, seed)` with
  `mrm.params.realize(cfg, batch, seeds, grid=None) -> PlantParams`. It works in this order:
  1. It draws the epistemic `uncertainty.*` entries per world, shape (B,), and the per-device `variation.*` entries,
     shape (B, N), (B, L) or (B, 1), from separate seed streams.
  2. It runs the relation solvers on those arrays.
  3. It assembles the thermal MIMO system per world: DC scaling, τ scaling and ZOH coefficients. This moves out of
     `plant/thermal.py`.
  4. It quantizes every latency to ticks.
  5. It asserts the timing rules (P0-6) and the self-heating lag bound (P0-1).
- `PlantParams` is a frozen dataclass of float64 arrays with fixed broadcast shapes: die (B, 1), ring (B, N), line
  (B, L), ring × line (B, N, L), thermal residues (B, N_dst, N_src, K). The constructor becomes `Plant(params)`.
  `derive(cfg)` stays only as the B = 1 nominal view for documentation.
- Add a top-level `uncertainty:` section, separate from `variation:`. Its entries are `hypothesis` (discrete) or
  `interval`/`dist`. Each is drawn once per world and shared by all rings. The initial entries are in §4.8.
- `ring.thermal.time_scale` becomes `uncertainty.thermal.time_scale_fast` and `time_scale_slow` (die level). Delete
  `variation.time_scale_rel_sigma` and `variation.dlambda_dt_rel_sigma`: these are die-level and epistemic, not per
  ring.
- `experiment.compare` reports, for each controller, the nominal result, the worst named corner, and quantiles over
  256 to 512 Latin-hypercube worlds (`scipy.stats.qmc`), plus Kendall-τ rank stability. A new rule in §6: a
  controller claim is published only if it holds over the uncertainty set.
- An AST test asserts that nothing under `mrm/plant` imports `mrm.params.config`.

**Why.** The draft does not say how `derive` (on a scalar Config) and `sample` (returning (B, N)) compose. If derive
runs on the nominal config, the sampled heater R, Q, efficiency and dλ/dT never reach the decay rates, R_th or the ROM
scaling. Monte Carlo then varies only the fields the plant reads directly, with no warning. The largest model errors
(dλ/dT, τ, the FSR window, the driver law, the heater family) are shared by every ring on a chip. Sampling them per
ring averages the worst case away, and a single nominal plant lets a controller be tuned to a time constant that has
no Lightmatter source.

### P0-5. Freeze the split between calibration and held-out targets before any fitting

Sources: [VAL] P0, [ARCH] P1 (tests).

**Change.**
- Add `mrm/validation/targets/{calibration,validation,bounds,excluded}.yaml`. Each entry holds: id, quantity, value,
  sigma (including read-off error), conditions {T_c, heater_v or code, bias_v, p_in_dbm, port}, generation, source,
  provenance, replica, and a tolerance fixed before any comparison.
- Add `mrm/validation/replicas.py`, with one function per published measurement protocol: `notch_er_vs_heater_v`
  (supporting both all-pass and add-drop, because the G25 TX topology is ambiguous [R01 D14]),
  `loaded_q_from_3db_width`, `tuning_range_at_v`, `relOMA_map(mode='fixed_code'|'oracle')`,
  `monitor_current_at_lock`, `stabilizer_code_vs_T`, `apparent_shift_fixed_code`, `bistability_sweep`.
- `mrm/calibration` may read only `calibration.yaml`. `tests/test_validation_isolation.py` scans `mrm/calibration/**`
  for any reference to the validation or bounds files, and checks the SHA-256 of the held-out file against the hash
  recorded in the header of `docs/VALIDATION.md`.
- Every `calibrated` leaf lists `targets: [ids]`, its fit residual and its identifiability (§4.3).
- The contents are in section 4.

**Why.** Under rule 2 as written, `mrm/calibration` would fit every published number, and every later agreement would
be tautological. "Reproduces 86 pm/K" checks an input; 4.7 K/mW is 0.405/0.086 by construction. The split must be
frozen now: any split made later is contaminated by tuning.

### P0-6. Timing contract: integer ticks, one owner per delay, a fixed order of events

Sources: [CTRL] P0, [ARCH] P0.

**Change.** Add a new §5.0 "Timing".
- Time is an int64 count of `sim.dt_plant_s` ticks. `Observation.tick` replaces `t_s` (t_s stays as a derived
  property). `Plant.advance(inputs, n_ticks: int)` replaces `duration_s: float`. `mrm.params.load` rejects any
  hardware period or latency that is not a whole number of ticks, and names the key (ADC period, controller period,
  conversion latency, DAC latency, integration window).
- Each delay has one owner. `mrm/plant/io.py` (an ActuatorBank and a SensorBank, one entry per channel) owns the
  analog and converter path: `ring.monitor.adc_latency_s` (new), `ring.monitor.integration_window_s` (new),
  `ring.heater.latency_s`, `slew_codes_per_s` and `settle_tau_s`, plus the bias-DAC equivalents. The controller
  declares its own digital pipeline depth as `Controller.latency_ticks`. Delete
  `controller_interface.sensor_latency_samples` and `actuator_latency_samples`, and remove "actuator
  limits/slew/latency" from the runner in §5.5. The runner only schedules ticks.
- Order of events at a controller edge (draw it as a diagram in §5.0):
  1. the plant finishes its last substep;
  2. the SensorBank releases the samples whose sample tick plus ADC latency is at or before the edge
     (integrate-and-dump windows are running sums inside the substep loop, divided once, as in textbook Ch17);
  3. the controller step runs;
  4. the command latches after `latency_ticks`;
  5. the DAC latency FIFO, then the slew limit, then the settle filter.
  `applied_codes` is the DAC-register readback (the latched value), never the slewed analog value.

**Tests.** `test_loop_delay_impulse`: a probe controller sends a one-tick heater-code impulse, and the first
monitor-code response appears at exactly the sum of the declared latencies. `test_latency_invariant_to_rate`: sweep
`update_rate_hz` from 10 to 200 kHz; the delay in seconds does not change. `test_fifo_depth`.

**Why.** The lag budget for 5 pm at 2000 °C/s is 29 µs (86 pm/K) or 38 µs (66 pm/K). A simulator that is one 10 µs
tick optimistic gives away a third of it. Ch17 names a wrong FIFO depth as the most common reason a simulator is
quietly more stable than the hardware. The draft defines actuator latency in three places and counts it in samples
of a rate that experiments sweep. Float time drifts: `t += 1e-6` in float64 is 1.9 ms off after 1e10 steps, which is
2.8 h of simulated time ([CTRL], measured). On the 30.4 h soak that drops or duplicates ADC samples.

### P0-7. Named I/O channels: heater and bias actuators, ADC sample blocks, port binding

Sources: [CTRL] P0 (twice), [ARCH] P0, [PHY] P1, [R01] patents, [R04] C5 and C13. Conflicts D4, D5.

**Change.**
- Signals are channels with a `ChannelSpec {name, shape, dtype, bits, min, max, unit, rate_hz, latency_s}`. The
  signatures become `Controller.step(obs) -> dict[str, ndarray]`, `Observation.sensors: dict[str, ndarray]` (plus
  per-channel sample ticks or new-sample flags) and `Plant.advance(inputs: dict, n_ticks)`. For compiled controllers,
  `realize()` generates a fixed flat int64 layout (an offset per channel) that uses the ABI of P0-9.
- Actuators in version 1:
  - `tx.heater` (B, N) uint16;
  - `tx.bias` (B, N), with `ring.bias_dac: {bits, v_min_v: 1.0, v_max_v: 3.75, update_rate_hz, latency_s,
    slew_v_per_s, settle_tau_s, noise_rms_codes}`. The range is the printed G26 axis; the rest is tagged assumed;
  - `tx.assignment` (B, N), the logical ring-to-line map (P1-12);
  - `datapath.bitstat_threshold` (P0-11);
  - `monitor.pga_select` (P1-10);
  - an optional `laser.tune` (B, L).
- Sensors in version 1:
  - `tx.monitor.samples` (B, N, M) int, with `sample_ticks` (B, N, M) and a count array: every ADC sample since the
    last step;
  - `tx.monitor.latest` (B, N);
  - `temp.*` and `heater.readback_v/i` (P1-10);
  - the data-path counters (P0-11).
- New keys: `ring.monitor.adc_rate_hz`; `afe_type: rc | integrate_dump`; `integration_window_s` (a boxcar, textbook
  eqs. 12.42 and 17.11a); `adc_mux: {channels_per_adc, order}`, which gives round-robin sample phases. Default
  `integrate_dump` with a 2 µs window, after the Lightmatter patent's dual-slope integrating ADC with baseline
  subtraction (T_int 1 to 10 µs; tagged inferred, PAT-2 [R01]).
- Binding: `controller_interface.bind: {tx.heater: controller, tx.bias: scenario | controller, laser.tune:
  scenario}`. When the controller owns the bias, an absolute bias in the scenario is a load error; a scenario
  `bias_offset_v` (ripple, droop) stays additive. When the scenario owns the bias, the relOMA-plane sweep (scenario
  A7) runs with the loop closed.
- An optional `Controller.fast_kernel`: a numba function with the flat ABI, called once per ADC sample inside the
  plant loop. It may write a per-tick DAC offset (a dither). It models the FPGA DSP fabric (NCO, I/Q lock-in, CIC
  decimation), while `step` runs the slow control law.
- Allow `sim.dt_plant_s` down to 0.1 µs for electro-optic dither studies.

**Tests.** A 2f lock-in written once as a `fast_kernel` and once as a 1 MHz Python step gives bit-identical I/Q
accumulators. A config that binds the bias to both the controller and the scenario fails to load.

**Why.**
- With one code per tick, dither lock-in, extremum seeking, oversampling and windowed hill climbing cannot run at
  realistic rates. A 25 to 50 kHz dither at 8 samples per period forces a 200 to 400 kHz Python controller, which
  costs 7 to 14 s of wall time per simulated second at the measured 35 µs per tick.
- The bias is a nanosecond actuator (about 50 pm/V, about 0.14 nm or 0.4 FWHM over 1.0 to 3.75 V) that bypasses the
  thermal lag. A heater dither at the published kHz rates cannot meet the G26 aggressor; a bias dither can [R04
  constraint 1]. Dual-actuator mid-ranging is the strongest candidate for a novel controller against edges above
  2000 °C/s [R04 C13], and G26's relOMA moves most along the bias axis.
- Interleavers, polarization controllers and the Guide laser servo then add channels, instead of breaking every
  controller, the runner, the gym env and the bridge.

### P0-8. Metrics computed online in the plant loop; traces stored as min/max envelopes

Sources: [CTRL] P0, [ARCH] P0, [VAL] P1 and P2.

**Change.**
- The kernel updates streaming accumulators per (B, N) at every plant step:
  - a detuning histogram, with 0.1 pm bins within ±2 FWHM and 10 pm bins out to ±FSR/2 (to bound memory);
  - the time above each threshold, and the excursion count and run lengths, with hysteresis;
  - max |detuning|, the time-integrated relOMA, and the OMA penalty against the oracle at the same instant (P1-1);
  - the expected errors Σ BER(OMA(t))·bitrate·dt, which gives the BER over any window, including the 2 s instrument
    interval;
  - heater energy, saturation time, demanded peak slew and code range used;
  - the wrong-line flag against the declared assignment, and the time to lock.
- `run(record=...)` takes a record spec: channel × decimation × reducer (mean | min | max | last). Monte Carlo records
  metrics only. `metrics.summarize` only reduces the accumulators.
- The `metrics:` section defines the acceptance numbers: `relOMA_definition: oma_over_pin` (D9),
  `ber_sample_period_s: 2.0` (G26), `lock_threshold_pm`, `hysteresis_pm`.

**Tests.** The accumulators equal the metrics computed from a full trace on a 10 ms run. A 5 µs spike in a 1 s run
decimated to 1 ms gives `excursion_count == 1`.

**Why.** Memory: B = 256, N = 8, 10 signals, 1 s at 1 µs is 164 GB (checked). Plain decimation hides exactly the
excursions that the lock-excursion counter and the fast aggressor edges produce, so a controller with brief losses
of lock would score clean. The BER of the mean OMA also understates errors.

### P0-9. A compiled closed-loop kernel, and a stated compute budget

Sources: [CTRL] P0, [ARCH] P1 (twice).

**Change.**
- Add `mrm/sim/kernel.py`: `@njit(parallel=True, cache=True) run_block(plant_state, plant_params, ctrl_fn,
  ctrl_p_i64, ctrl_p_f64, ctrl_s_i64, ctrl_s_f64, scen_params, n_ticks, metric_acc, trace_buf)`, with prange over B
  only and many controller ticks per Python call.
- Compiled controller ABI: `step(p_i64, p_f64, s_i64, s_f64, obs_i64, act_i64)` plus `init(spec arrays) -> state
  arrays`, passed as a first-class function. It has the same shape as the C ABI for Verilator and cffi (P1-21).
  Python controllers stay supported on a slow path that is vectorized over B.
- Scenario generators are parametric and evaluated inside the kernel: piecewise-linear segments, square, sawtooth
  and burst envelopes with phase, parabolic segments (the slide-10 aggressor), and random walks from the counter RNG
  (P1-13). `DisturbanceFrame (T, B, …)` arrays are kept only for replaying recorded profiles.
- Numba rules in §2. Kernels take every parameter as an array argument and the fidelity switches as an int32 flag
  array. No kernel reads module-level config or closes over config values: numba freezes globals at compile time
  ([ARCH] checked this — a rebound global still returned the old value). Set `NUMBA_CACHE_DIR=.numba_cache/`. Allow no
  float reductions across b.
- §3 gets a cost table per acceptance scenario. `pytest -m perf` targets for B = 1, N = 8, L = 8 on the full plant:
  at least 0.1 simulated s per wall s at a fixed 1 µs step; at least 20 with 100 µs adaptive steps (P0-10); a 64-seed
  × 1 s Monte Carlo in under 5 min.

**Why.** Measured on this M2 with [CTRL]'s prototype: about 5.5 µs per 1 µs plant step at B = 1 (0.17 to 0.2
simulated s per wall s), and about 1.6 to 1.7 µs per element-step with prange at B ≥ 64. A numba controller costs
25 ns per tick, against about 35 µs for the Python path. A 30.4 h soak at 100 kHz is 1.1e10 controller calls, about
4.5 days of Python overhead alone. The full plant may cost 2 to 4 times the prototype, hence the lower targets.

### P0-10. Long runs: exact multi-tick stepping, a quasi-static mode, validated time compression

Sources: [CTRL] P0, [ARCH] P1, [R01] D5. Conflict D12.

**Change.**
- `sim.substep_policy: fixed | auto`. In `auto`, the thermal ROM advances over a whole controller period in one
  exact step: a zero-order hold for the held heater power, and an exact first-order hold for piecewise-linear base
  and aggressor inputs (closed form per Foster pole). Optics and monitor substeps run only where needed: at least
  `sim.min_samples_per_dither_period` (8) while a dither is active, at least one per ADC sample, and enough that
  |dλ/dt|·dt ≤ `sim.max_detuning_step_pm` (0.2). The implicit self-heating solve (P0-1) keeps large steps stable.
  Fall back to fixed substeps when |dP_abs/dt| exceeds a stated tolerance.
- `scenario.time_compression` (default 1) speeds up only slow exogenous signals (TEC ramp, laser drift, aging), never
  plant time constants or the controller rate. A load-time check keeps the compressed ramp at no more than 1 % of the
  fast-transient rate. A convergence test is mandatory before any compressed result is reported: one soak cycle at
  compression 1, 10 and 100 must agree on max detuning, wrong-line and FSR-hop events, heater range used and
  integrated errors.
- `experiment.kind: quasi_static` root-finds the locked steady state, with self-heating and static thermal gains,
  over ambient 0 to 105 °C and bias 1.0 to 3.75 V. It reports code-range coverage, FSR-hop and wrong-line risk, and
  the oracle relOMA map (BND-4). Dynamic soak windows (turnarounds, the fastest ramps) start from quasi-static states.
- The soak scenario is one experiment: 33 to 108 °C, ≈ 90 cycles of 20.3 min, ≈ 0.124 °C/s, 30.4 h [R01].
- `mrm run --dry-run` prints B, step counts, estimated RAM and estimated wall time.

**Why.** At 1 µs, 30.4 h is 1.1e11 steps, about 7 days at 5.5 µs per step. The soak ramps about 16,000 times slower
than the fast edges. It tests range, order hops, TCR droop, Q(T), drift and long-run integrator behaviour, not loop
bandwidth. With multi-tick stepping, a 10 kHz compiled controller runs the full soak in about 2 h of wall time
(1.1e9 ticks × about 6 µs); compression brings screening down to minutes.

### P0-11. Data-path observables for bit-statistics and BER-driven controllers

Sources: [CTRL] P0, [R04] C6, C8 and D21.

**Change.**
- Add `mrm/plant/datapath.py` and these channels:
  - `tx.ones_count` and `tx.bits` (B, N) per controller window (the transmitter knows its own data);
  - `datapath.bitstat_above` (B, N): the number of monitor samples above `datapath.bitstat_threshold` in a window of
    `datapath.bitstat_window_bits` (16 to 128 in Sun et al.). It is drawn as Binomial(n₁, p_above|1) +
    Binomial(n₀, p_above|0), from the instantaneous P₁ and P₀ at the monitor and the AFE noise;
  - `rx.error_count` (B, N), delivered every `datapath.error_count_period_s` (2.0 s, G26) after
    `datapath.error_count_latency_s`, and drawn as Poisson(Σ BER(OMA)·bits);
  - optionally `rx.eye_margin`.
- `fidelity.pattern_noise: off | binomial | prbs<k>` generates the ones density per substep, so the averaged monitor
  carries the ones-count random walk.
- Scenario events: ones-density steps 0 → 0.5 → 1 (Sun et al., every 200 ms), traffic start and stop, idle, squelch,
  and training patterns.
- Version 1 is statistical. Each window length is validated against `mrm/waveform` runs (P2-6).

**Why.** Bit-statistics locking (Sun et al.: 76 kHz decisions, lock within 7 ms over 600 GHz in VLSI 2015, or
524 GHz in JSSC 2016) is the closest published analogue of a production TX lock. Textbook Ch12 finds that the
ones-count fluctuation is the largest error term for a lock that does not know the count. In steady scrambled
traffic the ones density over a 10 µs thermal
window varies by only 0.07 %, so what matters are the events listed above [R04 §3.3]. The draft averages over mark
density and exposes no count, no threshold sampler, no error counter and no eye monitor. That whole controller class
cannot be written, and neither can data-balance normalization or the squelch failure mode.

---

## 2. P1 changes

### P1-1. Controller knowledge, Truth isolation, and an oracle baseline

Sources: [CTRL], [ARCH], [VAL], [R04] §1.8. Conflict D13.

**Change.**
- Remove `ControllerSpec.extras`. Add typed fields: bus_order, tick_s, adc_rate_hz, samples_per_tick,
  adc_full_scale_codes, adc_latency_ticks, heater-DAC facts, bias-DAC facts, temperature-sensor metadata, and the
  channel specs.
- Add `ControllerSpec.nominal: Config`. It is built from a `controller_knowledge` preset (nominal values, variation
  off, optional `mismatch.*` keys) and filtered by `controller_interface.knowledge: [line_grid, ring_to_line,
  heater_r_nominal, dlambda_dt_nominal, …]`.
- Add `mrm.control.models.linearize(nominal_cfg, operating_point, Ts) -> (A, B, C, D, delay_ticks)`, built from the
  same ROM, optics slope, AFE and ADC gain.
- Add `mrm.control.calibration.factory_cal(plant, procedure)`: a simulated factory heater sweep that returns a noisy
  per-ring table. It is passed at reset (warm start) or withheld (cold start, scenario B5).
- Add `mrm/validation/oracle.py`: a Truth-fed controller that holds each ring at the OMA-optimal, open-loop-stable
  detuning. It is used by the replicas and the OMA-penalty metric, and is not registered in
  `mrm.control.registry`.

**Tests.** `test_spec_independent_of_seed_and_uncertainty`. `test_control_does_not_import_truth` (a static import
scan of `mrm/control`). `test_no_truth_leak`: an Observation proxy raises on access to any undeclared channel, and no
array in the spec or the observation equals a sampled per-ring parameter.

**Why.** Kalman, MPC and observer controllers need a model; [R04] found none published for MRM thermal locking, so
this is the open niche for the capstone. The easy shortcut — the plant's own Config, or `extras` — hands them each
ring's sampled offset, Q and efficiency, and they look perfect.

### P1-2. Thermal ROM: a full source-by-observer matrix over named sites, and an interim model from measured data

Sources: [PHY], [ARCH], [VAL], [CTRL], [R04] §2, [R06] N5 and E1. Conflicts D15, D22.

**Change.**
- §5.2 becomes a generic table:
  - sources {heater, waveguide_absorption, junction, driver, aggressor};
  - observers {ring_mode_weighted, heater_resistor, monitor_pd, temp_sensor};
  - each pair gets self (d = 0) and cross residues on the shared τ set, tabulated by (dx, dy) relative to the heater
    orientation, not by radius alone;
  - the shared τ set must include the millisecond poles that carry the crosstalk. Mixed-sign residues are allowed for
    non-collocated pairs. A per-pair `transport_delay_s` is optional;
  - for d > 2R, other sources may reuse the heater cross residues, with the FEM error at d = pitch documented.
- `layout.sites: {name: {x_um, y_um, kind}}` replaces `array.pitch_um`. It supplies the aggressor, sensor, PD and
  driver placement, which no key provides today.
- The heater resistance follows R = R₀·[1 + α·(T_heater − T₀)], with T_heater taken from the ROM's heater observer,
  so it includes ambient, neighbours and aggressors [R06 E1].
- `uncertainty.thermal: {family, time_scale_fast, time_scale_slow, slow_share, transport_delay_s, crosstalk_scale,
  scaling_hypothesis: resistance | capacitance | independent}`. Under `resistance`, each τ_k scales by the same
  factor as its residue (τ = RC).
- Interim model until the FEM ROM of the default heater family exists (the `two_pole` path becomes three poles). It
  uses [R04]'s numbers because they are the only measured multi-pole set (imec, metal heater above the ring, no
  undercut); the heater family itself stays a hypothesis (P1-3):
  - self path: 3.2 µs (43 %) + 13.0 µs (57 %), plus a die and package tail of 1 to 10 ms carrying 2 to 10 %;
  - crosstalk: 3 % to the nearest neighbour (sweep 1 to 10 %) and a 1.5 % floor to every other ring (0.5 to 2.6 %),
    with first-order dynamics τ_xt = max(0.1 ms, 4.1 ms·(d/60 µm)²) up to 100 µm and 4 to 9 ms beyond;
    [2026-10-16, physics_thermal review: d is the heat-source-to-victim distance of [R04] §2.5 (60 µm for imec's
    first neighbour at a 100 µm pitch), i.e. the centre distance less both ring radii; the runtime passed centre
    distances until then. The crosstalk numbers are imec's natural-convection package; the FEM ROM's TEC-held package
    is the choice `uncertainty.thermal.stack: side_by_side_tec_held` (docs/CHANGELOG.md)]
  - the textbook's 10/300 µs and 0.1/0.03 crosstalk with the self-path dynamics become a pessimistic corner.
- A derived diagnostic, `rom_dc_scale = efficiency_measured/(Z_FEM·dλ/dT)`, per dλ/dT hypothesis. It is printed in
  VALIDATION.md, with a warning outside [0.5, 2].

**Tests.** Reciprocity, Z_ij = Z_ji, for like sources and observers. A neighbour-lock step test: locking ring j steps
ring i by the `cross_opt` residue.

**Why.**
- Each locked ring absorbs about 1 to 2 mW. That power switches on acquisition, on loss of lock and at line turn-on,
  a MIMO coupling the draft omits.
- Measured crosstalk is small, long-range and two to three orders of magnitude slower than the self path [R04 X1,
  X2]. A decoupler built on "same dynamics" is optimistic.
- Heater readback works as a thermometer only if T_heater sees the neighbours, the ambient and the aggressors.
- Scaling the DC gain while keeping τ fixed is physically inconsistent. τ_fast is the unknown that matters most:
  loop bandwidth, the lag budget and the tracking error all scale with it. Hiding the DC rescale hides the only
  evidence about the stack.

### P1-3. Thermal ROM extraction and its verification

Sources: [PHY], [VAL], [R02] §5, [R04] §2.

**Change.**
- `thermal_fem.py` uses the materials of [R02]: thin-film silicon k ≈ 56 to 66 W/(m·K) for 150 to 200 nm films
  (±25 %), not bulk 148 and not the smoke test's 100; k(T) for film and oxide; 2 µm BOX (printed for 45SPCLO [R01]).
- It reports the fit error in the frequency domain (|ΔH| < 0.3 dB and phase error < 3° from DC to 1 MHz) for the self
  and nearest-neighbour terms, plus max|ΔT|/ΔT_ss < 2 % from 10 ns to 10 ms (a verification test).
- It runs one full FEM of the 8-ring bank at the design pitch, with heater leads, vias, BEOL metal and the shared bus.
  It reports the superposition error of the distance-interpolated single-ring ROM, with a target below 10 % on the
  crosstalk residues.
- Heater family as a hypothesis: {in-ring resistive, no undercut (default; "heater and PN inside the ring" in the
  G26 schematic [R01]); GF-PDK-like silicide with undercut (four resistors, two inside and two outside the ring
  [R01, R04]); metal above}. Also a stack hypothesis {side-by-side (G26: SerDes beside the PIC), stacked EIC
  (M1000-like: −43 % efficiency, +44 % crosstalk in imec's 3D case [R04])}.
- Sweep the stack unknowns (heater depth and type, metal fill, substrate thinned or not, undercut) and write the
  resulting ranges into `uncertainty.thermal`, not a point.
- Use pyamg iterative solves under memguard, with graded meshes. Fall back to a 2.5D variant if 3D exceeds 3 GB.

**Why.** Loop stability depends on the phase near crossover (tens of kHz to 1 MHz). A step-response RMS hides the
phase error of diffusive cross terms, which act like dead time. Metal leads (k about 400 W/(m·K)) shunt about 10 % of
the oxide conductance between neighbours. The heater family moves the fast pole by 3 to 10 times: about 1 µs for an
interior heater, 5 to 10 µs for metal above, tens of µs with undercut [R04].

### P1-4. Disturbance paths, and the aggressor as power into the ROM with targets in resonance units

Sources: [CTRL], [VAL], [R01] D6, D7 and D13, [R04] §3, [R06] N6 and D2. Conflict D6.

**Change.**
- Every disturbance has its own path, and the four aggressor methods are never merged:
  - package or TEC base temperature (seconds to hours);
  - on-die aggressor heater, through the ROM `aggressor_*` columns;
  - XPU or ASIC power events (10 to 120 °C/s within about 30 ms at the PIC [MRVL]; about 100 °C/s about 100 ms after
    a 500 W step [R04]);
  - laser-wavelength sweep, optical only, with no thermal path (50 nm/s over 4 nm, trapezoid period ≈ 0.44 s [R01]).
- On-die aggressors: `scenario.aggressor: {site, waveform: parabolic | square | sawtooth | burst, period_s, duty,
  burst_on_s, burst_off_s, power_w}`. Targets are stated at the victim ring: `scenario.target: {units: resonance |
  temperature, trajectory}`, and dλ/dT is applied exactly once. The calibration step solves for the power waveform;
  the result is tagged `calibrated` and records the assumed thermal poles.
- Default targets in resonance units, using Lightmatter's own 15 GHz/°C:
  - A1: the slide-10 waveform — T from 25 to 113 °C, dT/dt a ±2200 °C/s sawtooth, period 130.5 ms, mean ≈ 1350 °C/s
    [R01 D7, D13];
  - A2: the paper's run, 808 °C/s read as ΔT per period (80.8 K in 100 ms), so the edges are at least 1600 °C/s
    [R01 D7]. It may be the same heater as A1.
  - The alternative hypotheses are `units: temperature` and a symmetric triangle. One 80 K edge at 808 °C/s alone
    takes 99 ms of the 100 ms period (checked).
- A replica `apparent_shift_fixed_code(cfg, T_grid)` models an ambient sweep at a fixed DAC code, including the TCR
  feedback: about ±12 % at P_FSR. Current drive raises the apparent sensitivity; voltage drive lowers it.

**Why.** Prescribing the ring temperature directly bypasses the network, so the sensors and the neighbours see the
wrong disturbance, and feed-forward or decoupling results mean nothing. If 808 °C/s was inferred from the resonance
shift through 15 GHz/°C, a °C target under H_solver understates the real disturbance by 1.2 to 1.3 times — a circular
result.

### P1-5. Coupling: the extinction-droop mechanism, and the under/over-coupled mirror

Sources: [PHY], [VAL], [R06] O5.

**Change.**
- Replace `coupling_dispersion_per_nm` with per-coupler κ²_through(λ) and κ²_drop(λ) from `extraction/coupling.py`.
  Use the real MRM cross-section (rib plus slab, 160 nm film) and the curved gap profile (a z-dependent gap and the
  phase mismatch between the bent and straight guides). Evaluate κ at each line wavelength, never at the resonance
  wavelength and never as a function of the heater code.
- Split `dlnqi_dt_per_k` into `dln_alpha_dT_ring_per_k` (applied to the mode-weighted ring temperature) and
  `dln_alpha_dT_heater_per_k` (applied to the ROM heater temperature, for a heater whose free-carrier absorption
  overlaps the mode).
- `uncertainty.coupling_regime: under | over`. `fit_coupling` returns every solution branch, written as alternatives
  (P1-14).
- `uncertainty.er_mechanism: dispersion | ring_t_loss | heater_t_loss` (mixtures allowed).
  [2026-10-16, params review: the CAL-1/CAL-2 fit ties the branch to the mechanism (dispersion under-coupled, the
  loss mechanisms over-coupled), so in the calibrated presets `coupling_regime` is a label entry of `er_mechanism`
  (follows + map), not an independent draw; docs/CHANGELOG.md]
- `mrm/calibration/er_vs_heater.py` emulates the G25 measurement (a low-power laser sweep at each heater voltage,
  all-pass or add-drop), fits on 0 to 1.2 V only (CAL-2) and predicts 1.4 and 1.5 V (VAL-2). The transfer from G25 to
  the G26 add-drop ring is stated explicitly.

**Tests.** `test_kappa_independent_of_heater_at_fixed_line`. `test_coupling_fit_returns_mirror`: both branches meet
CAL-1 and CAL-2, and their drop currents differ. `test_adjacent_order_discriminator`: at the same λ, order m + 1 has
the same ER under the dispersion mechanism and a lower ER under the loss mechanism. A ring-bus temperature mismatch
at the coupler is excluded with a reason: Δn_eff ≈ 0.02 at 100 K costs less than 1 % of κ².

**Why.**
- femwell (t02) gives d ln K²/dλ ≈ 1.3 to 1.5 %/nm. Over 13 nm of tuning that takes the notch from 28 dB only to
  about 25 to 26 dB, not 17 dB. Reaching 17 dB needs about 2.9 %/nm, or an extra loss term.
- The mechanisms differ under lock. At a fixed line, the resonance condition pins the ring temperature, so dispersion
  and ring-temperature loss give no ambient dependence under lock (only spread from ring to ring and order to
  order). A heater-temperature loss term does drift under lock, because T_heater falls as the ambient rises. A model
  that applies ER as a function of heater drive (the textbook's choice) invents or misses a closed-loop ER drift.
- Through-port magnitude cannot tell κ_i from κ_e. The two branches differ in drop current, best OMA side, and ER
  against temperature.

### P1-6. Phase model and FSR

Sources: [PHY], [VAL], [ARCH], [R01] D1, D2 and D10, [R02] §2. Conflicts D10, D17, D21.

**Change.**
- `extraction/modes.py` solves the bent MRM cross-section at the design radius for both film thicknesses (160 and
  200 nm) and records `r_ref_um`, the radius the solver's n_eff refers to. The runtime phase uses
  φ = 2π·n_eff,bent(λ, T, V)·2π·r_ref/λ with the same r_ref. Store n_eff(λ) to second order (n_g and dn_g/dλ), and
  L(T) with α_L.
- One silicon index model is used everywhere (Li 1980 for the temperature dependence, shifted to the Frey/Franta
  absolute level [R02 §2]). The two differ by 4.1e-3, which moves a design resonance by about 1.2 nm.
- `resonance_offset_nm` is defined at the cold cavity (P_in → 0), at a stated bias, with the heater off, at T_ref.
  G25's 1301.90 nm at 0 V is converted if it was measured at power.
- FSR is an epistemic hypothesis:
  - window B, 1.72 to 1.88 THz (default: the printed 1.8 THz design, the patent's FSR < 11.44 nm, BND-1);
  - window A, 1.32 to 1.48 THz;
  - stress "G25 cluster", 1.52 to 1.54 THz (8.7 to 8.8 nm from the wafer maps and the 15 µm diameter [R06 O8]); its
    next order sits 1.0 to 1.3 FWHM from another lane;
  - stress "PDK size", about 1.60 THz (R = 7.5 µm at n_g 3.9 to 4.04 [R01 §4.4]), with the next order on a line.
- Derived diagnostics: `next_order_distance_ghz`, the same in FWHM, and the IL and modulation crosstalk it implies
  (BND-1).

**Tests.** tidy3d and femwell bent n_eff agree once both refer to the same r_ref. Optional (D10): the FSR from the
phase model agrees with a converged cylindrical Meep run (at least three resolutions plus extrapolation) within
0.3 %.

**Why.** tidy3d (−0.0011 against straight) and femwell (+0.0014) disagree on the sign of the bend correction at
R = 7.5 µm. That is a radius-reference convention worth about 1 nm of resonance. FSR decides which lines fall near an
adjacent order, and therefore which wrong-order and aliasing events the model can produce. At 1.8 THz the aliases
sit exactly on the counter-propagating comb [R01 D10], so backscatter and interleaver leakage land on them.

### P1-7. Junction LUT completed; the physics of the bias actuator

Sources: [PHY] (twice), [R02] §7, [R06] M2. Conflict D5.

**Change.**
- `junction_<name>.json` adds the absolute `alpha_fca_per_m(V)`, `r_series_ohm(V)`, `fill_fraction` and `overlap`.
  The round-trip phase becomes 2π/λ·[n_eff·L + fill·L·Δn_eff(V)].
- Plasma dispersion: Nedeljkovic et al. 2011 at 1.31 µm as the default, with Soref-Bennett 1987 behind the same
  switch. At 1e18 cm⁻³ the newer fit gives a 10 % smaller |Δn| and a 13 % smaller Δα [R02 §0.5].
- Extend the voltage grid to −0.5 V forward bias (bias − Vpp/2 can reach it). An optional temperature axis (V_bi
  about −2 mV/K). The shift must follow a monotone depletion law Δλ = A·[(V + V_bi)^m − V_bi^m], m between 1/3 and 1/2
  [R06 M2].
- In LUT mode, `eff_pm_per_v_at_0v` and `dqi_dv_frac_per_v` become derived, report-only values.
- Calibrate `fill_fraction` to CAL-5 (an average of 52.8 pm/V over 0 to −1.6 V) after removing the self-heating
  relief at the measurement power. The `tx.bias` channel (P0-7) drives both Δn_eff(V) and α_FCA(V), hence η(V) and
  P_abs.

**Tests.** A +0.5 V reverse step at lock, heater held: an instant red shift of about 25 pm, then a blue relaxation on
the self-heating poles with amplitude G·ΔP_abs. Held-out VAL-3 and VAL-4.

**Why.** Without the absolute α(V) the model cannot compute η, Q_i(V) or the thermal side effect of the bias
actuator. Without R_s it cannot compute junction heating or the RC time. The DC pm/V under optical power is not the
RF pm/V. A LUT kept next to a flat pm/V and a dQi/dV over-specifies the junction.

### P1-8. Data-activity heat: junction CV² dissipation and the monolithic driver

Sources: [PHY], [R04] §3.3.

**Change.**
- The scenario and DisturbanceFrame add `transition_density (T, B, N)`, `swing_vpp (T, B, N)` (covering squelch, idle
  and driver-off) and `driver_enable (T, B, N)`.
- New keys: `driver.output_resistance_ohm`, `driver.static_power_mw`, `driver.energy_per_transition_fj` and
  `driver.site` (in `layout.sites`), plus the existing `ring.junction.series_resistance_ohm`.
- P_j = ½·C_j(V)·V_pp²·bitrate·transition_density·R_s/(R_s + R_drv) enters the ROM as source `junction`.
  P_d = P_static + E_tr·bitrate·transition_density enters as source `driver` at its site.

**Test.** A traffic-enable step at lock moves the hot resonance by the analytic amount within 5 %. Halving
`transition_density` halves P_j.

**Why.** Random NRZ draws ¼·C·V² per bit: 4 to 17 fJ/bit, or 0.2 to 0.85 mW at 50 Gb/s for C of 20 to 40 fF and 0.9
to 1.3 Vpp. Roughly half is dissipated in the ring's own doped series resistance, inside the mode region. At 350 to
880 pm/mW that is 40 to 300 pm (0.1 to 0.9 FWHM), switched within microseconds whenever traffic starts, squelches or
changes density. That is the most common fast disturbance a deployed loop sees, and checklist item E has no physical
path today.

### P1-9. Monitor: an identifiable lumped gain, the weak drop coupler, PD temperature and dark current

Sources: [PHY], [VAL], [R02] §10, [R06] S4, S6 and S8. Conflicts D7, D20.

**Change.**
- Add a derived `identifiable` block, recorded in every Trace and in VALIDATION.md: `monitor_codes_per_uw_at_drop`,
  `heater_pm_per_code(code)`, `fwhm_pm`, `disturbance_pm_per_k`, `selfheat_pm_per_mw_in`.
- CAL-8 (≈ 15 µA) calibrates only the product `monitor_codes_per_uw × P_line × lock factor`. The line power at the
  ring, κ_d, the responsivity, the TIA gain and the ADC LSB stay nuisance parameters, varied through
  `uncertainty.monitor.gain_scale [0.5, 2]`. The q_drop prior spans κ_d/κ from 0.4 % to about 18 % (the latter for a
  5 % per-pass drop coupling on a near-critically coupled ring, GF PDK [R01]).
- Extract the drop coupler at its own large gap, with its own dispersion and gap sensitivity. Its variation comes
  through the geometry sampler (P1-16).
- The PD temperature comes from the ROM's `monitor_pd` observer at its site. Dark current:
  I_d = I_d0·exp[−(E_a/k)·(1/T_pd − 1/T₀)], E_a 0.35 to 0.38 eV, I_d0 an uncertainty of 1 to 30 nA at 25 °C, with its
  shot noise. Rename `dark_doubling_k` to `dark_activation_ev`.
- Add `ring.monitor.background_fraction` (stray or substrate light; assumed; default 0, with a range) and an optional
  `ring.monitor.reference_tap` sensor channel with its own responsivity, coupling and dark current [R06 S8].
  [2026-10-16, physics_sensing review: the fraction is of the total bus power at the ring, all lines; the reference
  tap is not built in the engine yet and a config that enables it is refused (docs/CHANGELOG.md)]
- The drop current sums every line and every resonance order, so neighbour leakage (about 0.7 % per neighbour at
  400 GHz [R06 S6]) appears on its own.

**Tests.** `test_monitor_tracks_intracavity_energy` (I_mon·κ_abs/κ_d = R·P_abs). `test_dark_current_vs_heater`.

**Why.** The drop signal's gain is set by a weak, large-gap coupler. d ln K²/d(gap) ≈ −2 %/nm [R05 t02], so ±5 nm of
gap is ±10 % of gain, and about 1.2 to 1.5 %/nm of dispersion is about 20 to 25 % across the comb. Controllers that
lock to a fixed set-point mis-lock on that spread; fraction-of-peak controllers tolerate it. A 2.5 % laser power
drift moves a side-of-fringe lock by about 5 pm unless the monitor is normalized against a reference tap. A controller
maps codes to codes, so only lumped products can be identified from public data.

### P1-10. Converters and sensors modelled as hardware; fault injection

Sources: [CTRL] (twice), [R01] PAT-2, [R04] C11. Conflict D14.

**Change.**
- `temp_sensors: [{name, site | attach: ring_i | bank | die, type: diode | rtd | thermistor, sample_rate_hz,
  latency_s, noise_k_rms, flicker_corner_hz, resolution_k, offset_k_sigma, gain_error_frac_sigma, self_heating_k}]`
  replaces `temp_sensor`. Thermal coupling comes from the ROM's `temp_sensor` residues at the site.
- `ring.heater.readback: {present, adc_bits, v_full_scale_v, i_full_scale_a, rate_hz, latency_s}` feeds the channels
  `heater.readback_v/i`. This replaces `expose_heater_readback`, which has no Observation field today.
- ADC: `adc_inl_lsb`, `adc_dnl_lsb`, `pga_gains` with the `monitor.pga_select` actuator, `adc_ref_tempco_ppm_per_k`,
  `flicker_corner_hz`, and the baseline subtraction of the patent's integrating ADC. Heater: `dac_inl_lsb`,
  `dac_dnl_lsb`, `dac_max_update_hz`, `driver_compliance_v` (current-drive headroom as R rises through the TCR).
  [2026-10-16, physics_sensing review: the tempco acts at `ring.monitor.adc_ref_site`, default the package base
  temperature (the converter beside the PIC), not at each PD; rings that share a converter (adc_mux) share its DNL
  and INL (docs/CHANGELOG.md)]
- Faults, as scenario-bound channel masks: `adc_drop_mask`, `adc_stuck`, `adc_saturate`, `dac_glitch`,
  `controller_reset_at`. Observation flags: `monitor_valid`, `adc_overrange`.

**Why.** On-die sensors are slow converters, with millisecond conversion and an uncalibrated offset of a few kelvin.
The draft's sensor would make feed-forward look better than it will on silicon (textbook Ch13 sizes the feed-forward
residual by sensor lag). Heater V/I readback gives R(T) through the TCR: a per-ring thermometer at zero area cost, and
a credible novel sensing route. DNL and missing codes create dead zones for small dithers and hill climbing (Ch14).
PGA autoranging is what lets a monitor survive connector re-mates and laser power steps.

### P1-11. Actuator limits come from hardware, not from measured demand; the driver law is a hypothesis

Sources: [CTRL], [VAL], [R01] D11 and PAT-2, [R06] E2.

**Change.**
- Presets must not set `ring.heater.slew_codes_per_s` from `REF.heaterDacSlewCodesPerMs` (460 codes/ms is an
  envelope of demand; per lane the mean is 310 to 340 codes/ms and the peak about 700 codes/ms [R01 D11]). They must
  not set `code_min/code_max` from `REF.heaterDacCodeRange` (the range one test used).
- Defaults: codes 0 to 65535 and unlimited slew, or a hardware value tagged assumed. Metrics report the demanded peak
  slew and the code range used (P0-8).
- `uncertainty.heater.driver_mode: voltage | current | power | dac_pwm`. `dac_pwm` is the patent's 8-bit DAC plus
  8-bit PWM over a 2.56 µs period, about 16 bits in total, with N/N+1 code dithering [R01 PAT-2]. It is modelled as
  the period average plus a ripple term.
- The heater TCR is tied to the heater family: doped Si 1.0 to 1.8e-3 /K, TiN about 0.5e-3 /K, silicides and W about
  2 to 4.5e-3 /K [R02 §9]. The textbook's fitted 1e-3 is at the low end.

**Why.** A demand-based limit makes the actuator only just sufficient, so the fast edges (about 1.5 to 2.3 times the
mean demand) fail by construction. In voltage or current drive P ∝ code², so dP/dcode varies 2.53 times across 19,500
to 49,300, and the TCR adds about ±10 %; in power drive it is flat. That decides whether a fixed-gain PI suffices or
gain scheduling is needed. The G26 code range cannot confirm dλ/dT: depending on the full scale it implies 71 to
88 pm/K [R06 E2].

### P1-12. WDM acquisition and arbitration

Sources: [CTRL], [ARCH], [R01] PAT-1 and PAT-3.

**Change.**
- The `tx.assignment` action (P0-7) and `metrics.allowed_assignments: fixed | cyclic | any`. `metrics.wrong_line`
  judges each ring against the declared map.
- `array.heater_total_power_max_w` and `array.heater_supply_current_max_a`, enforced by the plant (it clips and raises
  a flag).
- Scenario events: `laser_line_enable (T, B, L)`, laser failover (a step in line wavelength and power, scenario B2),
  and a fibre-loss step (scenario A6, connector re-mate).
- `channel_plan.lines` becomes a table {freq, direction, source_laser, pol}. It covers G26 (16 lines, bidirectional)
  and L20 (1311/1331 nm).
- ControllerSpec gives the bus order and the nominal grid only.

**Why.** Lightmatter's own patent remaps every ring by one channel (a barrel shift, with the wrap-around ring moving
to another order) when heater power becomes excessive [R01 PAT-1]. Its initialization ramps the heater and parks at
the monitor maximum [R01 PAT-3]. A fixed-assignment metric would fail correct controllers. Sweeping 8 heaters at once,
at 25 to 34 mW per FSR each, can exceed a die supply limit that a real arbiter must sequence around.

### P1-13. Noise and random streams

Sources: [CTRL], [ARCH], [VAL].

**Change.**
- Generate shot, TIA and AFE noise by exact discretization: the discrete-time Gramian of the filter, and a variance
  of S/(2T_w) for an integrate-and-dump window. Never use PSD × bandwidth.
- A counter-based RNG in numba (Philox or splitmix64), keyed by (seed, stream_id, b, ring, tick).
  `experiment.seeds: {device, noise, scenario}`. Stream ids come from a fixed enum (variation, uncertainty,
  laser_rin, laser_drift, adc_<channel>, shot_noise, scenario_<name>, controller).
- `Plant.reset(noise_seed)` never resamples the devices.

**Tests.** The monitor-noise σ at the ADC stays within 5 % for dt from 0.25 to 2 µs. Element b of a B = 64 run equals
the B = 1 run with seed b, for any thread count. Toggling `fidelity.shot_noise` leaves the ADC-noise draws unchanged.
Two controllers on one seed see the same noise (common random numbers).

**Why.** prange with per-thread `np.random` makes results depend on thread scheduling, which breaks the replay of a
single Monte Carlo case and paired controller comparisons. Noise that scales with dt silently makes a fine-step
plant noisier than a coarse one.

### P1-14. Provenance that is enforced and propagates

Sources: [ARCH], [VAL]. Conflicts D2, D3.

**Change.**
- Presets may not contain bare numeric leaves. Only overrides may, and they are tagged `override` automatically.
- A child preset that changes `value` must restate provenance and generation.
- `alternatives: {name: value}` on a leaf, selected with `--alt path=name`. `experiment.kind: robust` sweeps them.
- Derived values are written as `@derived(out, inputs=[…])` and inherit the weakest input provenance (rank: printed >
  read-off > calibrated > inferred > solver > literature > assumed). Solver outputs carry their inputs, for example
  `solver(assumed-geometry)`. `mrm params explain <path>` prints the chain.
- Generation is a closed enum {G25, G26, EVK100, L20, M1000, Guide, OCP, GF, PAT, none}; `literature` moves to
  provenance. `source` must be a key in `mrm/params/sources.yaml`, which mirrors the textbook's tags. Optional fields
  `textbook: REF.<field>` and `disagrees: <reason>` generate the table of disagreements with the textbook.
- Rule 2 in §6 is reworded: "Measured Lightmatter data beats solver output, solver output beats literature, and
  literature beats assumption. When a printed value is not a measurement, or violates a physical bound computed from
  materials data, the bound-respecting value is the default and the printed value is a required robustness
  alternative. VALIDATION.md states the gap."
- Answers that Lightmatter gives under NDA go to a git-ignored `mrm/params/presets/private/`, with provenance
  `private`.

**Why.** For a controller designer, provenance answers which plant properties the controller must be robust to.
That works only if derived quantities carry their inputs' weakest provenance, and if disputed values can be swept
without forking presets.

### P1-15. A strict schema and a unit-suffix grammar

Sources: [ARCH].

**Change.**
- `mrm/params/schema.py` (dataclasses, or pydantic v2 as a core dependency) lists every key, its type and allowed enum
  values, and which keys are required or forbidden under each fidelity choice. For example, `thermal_model=rom`
  requires `rom_file` and forbids `tau_fast_s`. Unknown keys are an error with a did-you-mean suggestion.
- A suffix parser for keys of the form `<name>[_<qualifier>]_<unit>`, with unit tokens from one table, combined with
  `per` and `rt`. `_c` is affine; `_dbm` and `_db` are nonlinear.
- Renames: `wavelength_error_ghz_sigma` → `wavelength_error_sigma_ghz`; `resonance_offset_global_nm_sigma` →
  `resonance_offset_global_sigma_nm` (and the local one); `laser_power_db_sigma` → `laser_power_sigma_db`;
  `capacitance_ff_at_0v` → `capacitance_at_0v_ff`; `eff_pm_per_v_at_0v` → `eff_at_0v_pm_per_v`. Split
  `ring.heater.full_scale` into `_v`, `_a` and `_w`. Drop `unit` from the leaf mapping.
- State that `_hz` and `_ghz` are always cycles per second, and say whether `backscatter_ghz` is the mode splitting or
  the coupling rate.

**Why.** The draft breaks its own suffix rule about eight times. An override typo (`r_cold_ohms`) silently runs the
nominal model 20 times and looks like a robustness result.

### P1-16. Variation keyed by path, and sampled from correlated geometry

Sources: [ARCH], [PHY].

**Change.**
- `variation:` entries are keyed by config path: {dist, rel_sigma or die_sigma/ring_sigma, level: die | ring | line,
  corr_length_um}, truncated to the leaf's range.
- Add `variation.geometry: {width_nm_sigma, thickness_nm_sigma, slab_nm_sigma, gap_through_nm_sigma,
  gap_drop_nm_sigma}`, each with a global/local split and a correlation length. Extraction stores the Jacobians of
  {λ_res, n_g, K_e², K_d², α_rad, dλ/dT} with respect to {w, t, slab, gap}. `realize()` draws the geometry and maps it
  through the Jacobians. The direct sigmas remain as overrides.
- The fabrication offsets (6 nm global, 2 nm local) are Berkeley figures (literature), not Lightmatter's. The G26
  lane spread of 7.5 to 9.5 k codes [R01] is a calibration target for them (CAL-9).

**Test.** The sampled correlation of (offset, K_d²) matches the Jacobian prediction.

**Why.** One width or thickness error moves the resonance (about 1 to 2 nm per nm), the FSR and the coupling
together. Independent draws create impossible rings that controllers can overfit to. Monitor gain, the most variable
per-ring quantity, is absent from `variation.*` today.
[2026-10-16, params review: in g26 the geometry now moves the couplers through the coupling Jacobian (the R4
anchors describe the nominal-geometry device of each world; a width error also closes both gaps) and the FSR
through n_g at the mask radius of that device; the bend radiation follows its exponential law. The test runs on
g26 (tests/verification/test_params_review.py; docs/CHANGELOG.md)]

### P1-17. Experiment files, registries and run manifests

Sources: [ARCH].

**Change.**
- New top-level sections: `controller: {name, params}`, `scenario: {name | stack, params}`, `metrics: {…}`,
  `experiment: {kind: run | sweep | mc | compare | robust | acceptance | quasi_static, duration_s, batch, grid, n_mc,
  seeds}`.
- Registries `mrm.control.registry` and `mrm.scenarios.registry`, keyed by name or by `module:Class`.
- Two entry points: `mrm run exp.yaml --set a.b=c`, and `mrm.run(mrm.params.load('exp.yaml'))`.
- Each run writes `runs/<UTC>_<cfghash8>/`:
  - `experiment.resolved.yaml` (every leaf with its provenance);
  - `manifest.json` (git commit and dirty-diff hash; numpy, scipy, numba and llvmlite versions; platform; argv; seeds;
    the sha256 of every artefact used);
  - `metrics.json`;
  - `trace.h5` (h5py becomes a core dependency).
- `mrm rerun runs/<id>`. Ship `experiments/acceptance/*.yaml` for the textbook's acceptance table.

**Why.** Comparisons must be reproducible from a file, and exchangeable with Lightmatter. Otherwise each student
writes a private driver script.

### P1-18. A pure-NumPy reference of the kernel

Sources: [ARCH].

**Change.** Keep a NumPy reference of the advance kernel. Add a parity test with `NUMBA_DISABLE_JIT=1`, which also
lets students step through the kernel in pdb. Add a test that two configs run in one process give the expected
different outputs (a guard against frozen globals).

**Why.** A frozen global gives a plant that ignores a swept parameter, with no error.

### P1-19. A controller API a student team can use

Sources: [ARCH].

**Change.**
- `BaseController` (a no-op reset, a params dataclass filled from `controller.params`, state arrays allocated by
  shape).
- `PerRing(ctor)` and `PerDie(ctor)` adapters, so that a scalar algorithm is vectorized over B and N.
- The runner validates shape, dtype and range on every tick, applies `on_out_of_range: clip | error`, and counts
  saturation.
- Templates in `examples/controllers/` — dither lock-in on the bias, drop-port set-point PI, ramp-and-maximize
  acquisition (PAT-3), bit statistics — and a tutorial notebook. [R04] §4 already gives the update laws.

**Why.** Most capstone controllers will start as per-ring loops. The adapters remove the (B, N) barrier and keep the
batched path for Monte Carlo.

### P1-20. Controller telemetry and supervisory metrics

Sources: [CTRL].

**Change.** `Controller.telemetry() -> dict[str, (B, N) array]` (state-machine state, lock flag, estimated detuning,
integrator value), recorded at the trace decimation. Metrics: missed loss of lock, false alarms, declared against
true time to lock, and integrator saturation time.

**Why.** Production controllers publish telemetry (Ch17; the lock-excursion counter in Hot Chips). A controller that
sits on the wrong line while reporting lock is worse than one that re-acquires.

### P1-21. Fixed-point toolkit, golden vectors and HDL co-simulation

Sources: [CTRL], [ARCH].

**Change.**
- `mrm/control/fxp.py`: declared Q formats; int64-backed operations with explicit wrap or saturate and a rounding
  mode; a checked mode; usable from numba. `mrm/control/lfsr.py` for controller randomness and dithers.
- `mrm.sim.record_vectors(run, path)` writes versioned `.npz` and `.hex` files for `$readmemh`. In `tests/hdl/`,
  cocotb (with Icarus or Verilator) replays the observation vectors and asserts bit-exact actions.
- Closed-loop co-simulation: Verilator builds a `.so` with `ctrl_init` and `ctrl_tick(const int32_t* obs, int32_t*
  act)`, loaded through cffi inside the numba kernel.
- `bridge.py` uses binary, versioned, lockstep frames generated from the ChannelSpecs ({u64 tick, u16 adc[N], i16
  temp[S], u8 flags} → {u64 tick, u16 dac[N], u16 bias[N]}), never wall-clock time.
- Equivalence: float against fixed point within bounds (Ch17); fixed point against RTL, bit-exact.

**Why.** Python integers never overflow, and numba int64 wraps at 64 bits, not at the 18 to 40-bit widths of the RTL.
Ch17 shows that the integrator's fraction bits are the usual failure. A text protocol at 10 to 50 µs per round trip
limits co-simulation to millisecond windows. All the tools are free (Verilator, Icarus, cocotb, Yosys).

### P1-22. A batched RL environment

Sources: [CTRL] P1, [ARCH] P2. Conflict D18.

**Change.** `MRMEnv` as a `gymnasium.vector.VectorEnv` over the plant batch, with no subprocess workers, and spaces
generated from the ChannelSpecs. frame_skip = the controller period. Actions are bounded delta codes, plus the bias.
The observation is the last K sample blocks, applied codes and temperature codes, normalized. The reward is computed
in the kernel from Truth, which appears only in `info['truth']`. Each reset draws a new world and device (domain
randomization over `uncertainty` and `variation`). Target: at least 1e5 env steps/s at B = 256. A fixed-point MLP
inference helper in `fxp`.

**Why.** RL needs 1e7 to 1e8 steps. At about 35 µs of Python per tick, a B = 1 environment gives at most about 3e4
steps/s before any plant work. A policy that cannot be quantized is not an FPGA controller.

### P1-23. Test pyramid and generated validation report

Sources: [ARCH], [VAL]. **Change:** see section 4. CI blocks on verification and closure tests only.

---

## 3. P2 changes

**P2-1. Exclude TPA, FCA and Kerr with a computed bound.** [PHY]. Keep `fidelity.nonlinear_optics` off. Add
`tests/test_nonlinear_bounds.py` at the highest preset power (+9 dBm at the ring) and Q = 4200, with β_TPA from 0.5 to
2 cm/GW [R02 §8]. It asserts: TPA loss below 0.2 dB/cm against α_i ≈ 110 dB/cm; a TPA free-carrier shift below 5 pm
at τ = 1 ns (below 0.5 pm with junction sweep-out); a Kerr shift below 1 pm. Mark them excluded-with-reason in
PARAMETERS.md and remove `tpa_beta_m_per_w`, `n2_m2_per_w` and `mode_area_um2` from the runtime keys. *Why:* at
+4 dBm the ring circulates about 22 mW in about 0.15 µm², which is 3 to 4 orders of magnitude below the thermal pull.
Thermo-optic bistability (checklist H) is covered by P0-1.

**P2-2. Backscatter per order, and an external-reflection scenario.** [PHY], [R01] D10. Default `backscatter_ghz` of
about 1 to 10 GHz, with an independent random complex β per longitudinal order. Derive r_b = 2πβ/FSR in `derive()`,
so that the exact ring and TCMT agree. Route the CCW drop output to the add-port end, not to the monitor. Add
`scenario.external_reflection: {r_db, phase_drift_rad_per_s}` at the bus output (connector re-mates). *Why:* β/κ ≈
0.02 to 0.15, so through and drop distortion stays below about 1 %. Reflections of −17 to −34 dB matter for BiDi
crosstalk and laser feedback, and at FSR = 1.8 THz the aliases sit on the counter-propagating comb.

**P2-3. Parasitic transverse mode families.** [PHY]. `ring.mode_families: [TE0, TM0 | TE1]`, each with its own n_eff,
n_g, dn/dT, Q, coupling and `input_fraction`, summed incoherently. Test: a heater sweep shows weak extra dips with a
different FSR. *Why:* a wrong-lock trap that is separate from FSR aliasing.

**P2-4. Optics stored in solver-independent terms; a complex TCMT sum over orders.** [PHY]. Store per-pass K_e², K_d²,
the round-trip loss and β. `derive()` produces both the exact-ring and the TCMT parameters from them. TCMT sums orders
in complex amplitude. Test: exact and TCMT agree to about 1/F² near resonance. Document that the TCMT Lorentzian tail
underestimates the adjacent-400 GHz tail by about 20 to 30 %. *Why:* the finesse is only about 25 to 30.

**P2-5. Temperature-dependent conductivity and the heater's electrothermal gain.** [PHY], [R02] §5, [R06] N8. Optional
`thermal.k_temp_model` using [R02]'s film fit (−10 % from 300 to 375 K for a 200 nm film; bulk k ∝ T^−1.3 for the
substrate), applied as a quasi-static scaling of the residues. Document the electrothermal loop gain P·TCR·R_th,H ≈
25 mW × 1e-3 × 9 K/mW ≈ 0.23: current drive raises the gain by about 29 % at full drive, and voltage drive lowers it.
*Why:* heater power runs from about 0 to about 25 mW across the ambient range, which moves loop gain by tens of
percent (10 to 25 % [R06 N8]).

**P2-6. A data-modulation LUT in normalized coordinates, with a validity domain.** [PHY], [ARCH]. Keep the
mark-density averaging (correct, because the thermal and AFE time scales are far longer than a 20 ps bit).
`fidelity.data_modulation=lut` corrections are keyed by (detuning/HWHM, κ_e/κ, swing in linewidths, mark and
transition density, rise time). Store the domain, raise an error outside it, and record the config hash. Test: the
correction to the mean drop power is below 3 % for PRBS at τ_p ≈ 3 ps and a 20 ps UI. *Why:* a LUT keyed on raw
parameters is invalid for every sampled device.

**P2-7. Multi-rate composition and scheduling.** [CTRL]. `controller_interface.schedule: simultaneous | round_robin`,
with a per-ring update phase; `CompositeController([(fast_lock, 100e3), (supervisor, 1e3)])`; an optional
`tick_jitter_s`. *Why:* real controllers time-multiplex one datapath across 8 to 16 rings.

**P2-8. Optional sensing and actuation paths.** [CTRL], [R04] C12. `ring.monitor.location: junction_photocurrent`
(in-resonator photoconductance); a sigma-delta heater driver; `laser.tune` bound to the controller (the channel
exists from P0-7). *Why:* sensorless locking and joint laser-ring control are credible novel directions.

**P2-9. A fixed-point plant export and FPGA resource counts.** [CTRL]. Express the optics as a relative phase
dφ = (dφ/dT)·dT + (dφ/dV)·dV − 2π·n_g·L·dν/c around each ring's anchor, re-anchored once per FSR; a LUT in dφ; the
ROM as Q-format IIR filters; a bit-exact Python reference; Yosys LUT and DSP counts as `metrics.fpga_resources`.
*Why:* the absolute round-trip phase is about 475 to 565 rad, and float32 resolves it only to about 0.05 pm.

**P2-10. A generic optical element contract.** [ARCH]. Conflict D19. An S-matrix per element over (B, L, 2[, 2])
with TE/TM, a netlist, and the Redheffer star product, added when the second element (interleaver or polarization
controller) arrives. *Why:* the BiDi limiters (TX-to-RX reflection; the interleaver penalty at ±20 GHz, with two
interleavers in the path) need reflections between elements.

**P2-11. Pinned environment and hashed artefacts.** [ARCH]. A lock file with hashes (uv lock), with numba, llvmlite
and numpy pinned together. `schema: 'thermal_rom/1'` and a sha256 in every sidecar, verified by `realize()`. Pin
tidy3d and use only its local mode solver (its FDTD is paid cloud compute). Meep through a conda-forge
`environment-meep.yml` (the smoke runs used `~/photonics-simulations/.meep`), or drop it. A `control` extra with
casadi and nlopt. *Why:* an unpinned numba or numpy upgrade changes floating-point results and invalidates the golden
tests.

**P2-12. CLI and generated docs.** [ARCH]. `mrm params show | table | explain | diff | validate`; `mrm run | sweep |
mc | compare | robust | acceptance`; `mrm rerun`; `mrm report`. Generate `docs/PARAMETERS.md` from `mrm params
table`, with a test that fails when the committed file is stale.

---

## 4. Validation plan

The model's claim is "a calibrated family of plausible plants", not a digital twin [VAL]. The plan therefore keeps
four things apart: code correctness, consistency of the published numbers, fitting, and prediction.

### 4.1 Test classes (rewrite of accuracy rule 4 in §6)

| Directory | Question it answers | Blocks CI |
|---|---|---|
| `tests/verification/` | Is the code right? (against analytic limits) | yes |
| `tests/closure/` | Are the published numbers consistent through `derive()`? Known conflicts are `xfail(strict=True, reason='VALIDATION.md#<id>')` so they stay visible | yes |
| `tests/calibration/` | Fit residuals of the calibration targets (`pytest -m calibration`) | no, reported |
| `tests/validation/` | Held-out targets, bounds, third-party replications; slow; writes VALIDATION.md | no, reported |
| `tests/golden/` | Has a refactor changed the plant that earlier controller results were scored on? | yes |
| `tests/perf/` | Speed targets of P0-9 | no |

Solver extraction tests run under `tools/memguard.py` and are skipped when the `extract` extras are missing.

### 4.2 Verification tests (check the code, not the hardware)

- Energy balance per line: |t|² + |d|² + |r|² + A = 1 to 1e-12.
- The exact ring against closed forms (notch depth, drop peak, FSR periodicity, the Lorentzian limit). TCMT against
  the exact ring within 1 % for Q > 1000 within ±0.3 FSR.
- Reciprocity of the ring, and of the thermal ROM residues.
- The ZOH step response equals Σ r_k·(1 − e^(−t/τ_k)) for any dt; the ZOH coefficients against `scipy.linalg.expm`.
- Self-heating: the fold points against the analytic cubic (2 % of the HWHM); the implicit 1 µs solve against a 10 ns
  reference; no spurious oscillation; dt against dt/8.
- Relations hold to 1e-12 after `realize()`; the index-weight identity holds to 1e-4.
- The timing probe impulse, latency invariance against rate, FIFO depths.
- Batch invariance, thread-count invariance, seed reproducibility. Metrics within 1 % between dt = 1 µs and 0.25 µs,
  with self-heating on and off. Decimation does not alias the metrics.
- numba against the NumPy reference; noise σ invariant to dt; the nonlinear bounds (P2-1).
- Solver cross-checks: femwell against tidy3d at one r_ref; the ROM against the FEM in the frequency domain; devsim
  against the analytic depletion width (t04: the simulated width is about 5 to 8 % below the approximation); Meep
  cylindrical, optional and converged.

These test the code, not the device. Two mode solvers agree to about 1e-4 on one assumed geometry, while ±10 nm of
width moves n_eff by about 1e-2, or about 3 nm of resonance.

### 4.3 Calibration targets (`calibration.yaml`; only these may be fitted)

| ID | Target | Value (σ) | Conditions, generation, source | What it may identify |
|---|---|---|---|---|
| CAL-1 | Loaded Q | 3500 (printed, talk) and 3900 to 4200 (read-off), fitted as an interval | low power, cold, 0 V; G25 [LM-T] | Q_L, through the replica `loaded_q_from_3db_width` |
| CAL-2 | Notch depth against heater V | 27.9, 29.8, 29.8, 24.4, 22.1, 19.8 dB at 0.0, 0.2, 0.5, 0.8, 1.0, 1.2 V (±0.5 dB) | G25, low power; read-off [R01] (REF lists 0.6 V for the third point) | the coupling branch (with CAL-1) and the ER-mechanism mixture |
| CAL-3 | Heater P_FSR and V_FSR | medians 25.45 mW and 1.195 V; ranges 24.9 to 34.1 mW and 1.16 to 1.55 V for the variation model | G25 wafer, 60 dies, 25 °C | heater efficiency in FSR/mW, and heater R |
| CAL-4 | Heater V → λ curve | resonance positions 1301.89 … 1313.48 nm at 0 to 1.4 V (±0.03 nm) [R01]; droop of Δλ/V² of 20 to 30 % | G25 | only y = α·R_th,H and the driver-law shape |
| CAL-5 | Resonance shift against reverse bias | shape over 0 to −1.6 V; average 52.8 pm/V | G25, at measurement power (remove the self-heating relief) | `fill_fraction` and the junction LUT |
| CAL-6 | ~15 GHz/°C | 86 pm/K | G26 [LM26-P] introduction | not fitted: it is the H86 hypothesis |
| CAL-7 | Blue offset | 1301.90 nm at 0 V against a line near 1309.5 nm, so 7.6 nm | G25/G26 | `resonance_offset_nm` at the cold cavity |
| CAL-8 | Drop monitor current at lock | ≈ 15 to 16 µA | G26, closed loop, Fig. 3(b) | the lumped monitor gain only |
| CAL-9 | Heater codes on the slide-10 aggressor | 20 to 22.3 k codes per lane over about 88 °C; lane offsets 7.5 to 9.5 k codes | G26 [R01] read-off | the DAC LSB, only under an assumed driver law and full scale; the per-ring offset variation |
| CAL-10 | EO bandwidth at −1 V | 30.9 ± 2 GHz | G25 | R_s·C (waveform mode) |
| CAL-11 | Responsivity | 0.88 to 1.04 A/W over 500 parts | G25 | variation range |
| CAL-12 | Laser | ±20 GHz spec, ±15 GHz demonstrated; 0.02 nm and 2.5 % power per 24 h | G26, Guide | scenario parameters |
| CAL-13 | Aggressor shape | slide-10 T(t) and the 808 °C/s run (P1-4) | G26 [R01] | the aggressor power waveform (a scenario calibration, not a plant one) |
| CAL-14 | Stabilized relOMA at ≈ 1.05 V | ≈ −2.1 dB, relOMA = OMA/P_in (D9) | G26, slide 9 | one lumped modulation depth (swing × efficiency/FWHM) |

Rules. A leaf may be tagged `calibrated` only when its 95 % profile interval (`scipy.optimize.least_squares` plus a
profile likelihood) is at most half its prior range; otherwise it keeps `assumed` with a range. Each calibrated leaf
records its target ids, its residual and whether the fit is unique. Read-off targets carry their digitization σ and
record the digitizer and the crop.

### 4.4 Held-out validation targets (`validation.yaml`, frozen now; tolerances fixed now)

| ID | Target | Value | Tolerance | What it tests |
|---|---|---|---|---|
| VAL-1 | Tuning range at 1.5 V | 13.0 nm (1301.9 → 1314.9 nm) | ±0.7 nm | the TCR fit extrapolated past 1.4 V. Semi-independent: it comes from the same sweep as CAL-4 |
| VAL-2 | Notch depth at 1.4 and 1.5 V | 18.0 and 16.9 dB | ±1.5 dB | extrapolation of the ER mechanism |
| VAL-3 | EO bandwidth at −2 V | 35.4 GHz (34.1 in the caption) | ±3 GHz | the C(V) from the CAL-5 fit |
| VAL-4 | Stabilized relOMA across bias and temperature | −3.0 dB reached near 3.15 V at 0 °C and near 3.4 V at 105 °C; spread below 1 dB [R01] | ±0.3 dB on the spread, ±0.3 V on the contour position | the shape of dλ/dV(V), η(V), self-heating, the generation transfer |
| VAL-5 | Loop-open relOMA contour asymmetry at 2.25 V | −16 dB at 62 and 74.5 °C; −24 dB at 57 and 79 °C [R01] | sign of the asymmetry, ±2 °C | the self-heating line shape (weak: the map interpolates 5 °C samples) |
| VAL-6 | Stabilizer code excursion over 0 to 105 °C | from the slide-10 data | set when known | usable only once the DAC LSB and driver law are known (Q3); until then it is CAL-9 |
| VAL-7 | Third-party pipeline replication, optics | de Cea et al., Opt. Express 27, 24274 (2019), 45 nm MRM: n_g 2.73, Γ 0.65, τ_self ≈ 0.4 µs (as reported by [VAL]; re-check before freezing, including whether the 2.4 MHz thermal rate is angular) | ±3 % n_g, ±0.05 Γ, ±50 % τ | `modes.py` and `thermal_fem.py` on a geometry that is actually published |
| VAL-8 | Third-party pipeline replication, thermal | imec metal-heater ring: heater node τ ≈ 5 µs, waveguide τ ≈ 10 µs; two-pole fit 3.2/13.0 µs [R04: CO22-JLT, CO22-TCPMT] | ±30 % on τ | `thermal_fem.py` and the Foster fit on a published stack |

### 4.5 Bounds (from the fact that the G26 link works)

- **BND-1, adjacent orders.** For every ring, the IL and the modulation crosstalk that its other orders impose on the
  other TX lines must fit the link margin. The interleaver keeps the RX comb off the TX bus (Fig. 1 caption), apart
  from its finite isolation. At 2 FWHM the penalty is 0.26 dB IL and about −13 dB crosstalk relative to the victim's
  own OMA. At 1.0 to 1.3 FWHM it is 0.6 to 1.0 dB IL and −5 to −9 dB crosstalk on every lane (checked). With FWHM ≈
  60 GHz, the windows that keep ≥ 2 FWHM are 1.32 to 1.48, 1.72 to 1.88, 2.12 to 2.28 and 2.52 to 2.68 THz (checked).
  Examples: a 1.8 THz design gives 3.3 FWHM; 8.8 nm (1.54 THz) gives 1.0 FWHM; R = 7.5 µm at n_g 3.9 to 4.04 (1.58 to
  1.63 THz) gives about 0.5 FWHM.
- **BND-2, no order hop at the hot end.** For the worst lane:
  Δλ_blue(T₀) − ∫ dλ/dT dT (T₀ → 105 °C) − Δλ_sh,lock − Δλ_bias − δ_lock ≥ 0.
  With 7.6 nm and T₀ = 25 °C, H86 leaves 0.72 nm for self-heating, bias and the lock offset, and H_solver leaves
  about 2.0 nm (checked). [PHY]'s lock-point self-heating estimate (0.6 to 1.3 nm at +4 dBm) nearly uses up the H86
  margin. BND-2 therefore couples dλ/dT, the self-heating gain and the power at the ring. Caveats: 7.6 nm is a G25
  number; its reference temperature is unknown; G26 may stagger its rings. A continuous BER below 1e-9 over the soak
  suggests no hop, but 2 s BER sampling could hide one.
- **BND-3, aggressor feasibility.** Some causal controller, at a plausible rate and latency, holds the aggressor with
  the monitor inside its quiet band (about ±0.5 µA, read-off). This bounds τ_fast and the aggressor path together.
- **BND-4, oracle flatness.** The oracle relOMA at a fixed bias varies by less than 1 dB over 0 to 105 °C. Only a
  heater-temperature loss mechanism produces a closed-loop ER drift, so BND-4 bounds `dln_alpha_dT_heater_per_k`.

### 4.6 Closure and conflict table (generated into VALIDATION.md; `xfail(strict=True)`)

| Quantity | Conflicting evidence |
|---|---|
| dλ/dT | 86 pm/K (printed aside) · 62.5 (Lightmatter bench, inferred) · 62.54 (imec, measured, O-band) · ≈ 66 to 69 at 25 °C and ≈ 74 to 77 at 105 °C (FEM plus expansion) · materials ceiling 71 to 80 · 50 (textbook, refuted) |
| FSR | 10.3 nm (1.8 THz design) · < 11.44 nm (patent) · 8.7 to 8.8 nm (15 µm diameter, wafer maps) · 9.0 to 9.3 nm (R = 7.5 µm at the 160 nm film's n_g) · 11.2 nm (P_FSR × 0.44) · ≥ 14.1 nm (unsupported read-off) |
| Waveguide thickness | 200 nm (SMTAI render) · 160 nm (45SPCLO photonics layer) |
| Q | 3500 (talk) · 3900 to 4200 (linewidth read-off) |
| Heater efficiency | 0.44 nm/mW (printed) · 0.405 (median at 10.3 nm) · 0.348 (swept device at 1.195 V) |
| R_th | 4.7 K/mW under H86 · ≈ 6.1 K/mW under H_solver · 2.9 to 11.4 K/mW across heater families [R04] |
| relOMA level | −2.1 to −3.0 dB printed · −5.9 to −8.1 dB for a G25-like drive (checked) · requires 170 to 230 pm at Q 3500 [R06] |
| relOMA curvature | k ≈ 0.67 dB/°C² implies FWHM/(dλ/dT) ≈ 6 to 8 K only [R06 R4], in tension with Q 3500 and every dλ/dT hypothesis unless the swing or Q differ |
| Aggressor rate | 808 °C/s × 100 ms = 80.8 K (a mean rate) · slide-10 peak ±2200 °C/s · a symmetric triangle would need 1600 °C/s |
| G25 TX topology | all-pass in the paper schematic · input/thru/drop micrograph on later slides [R01 D14] |
| EO bandwidth | 35.4 against 34.1 GHz at −2 V · 66.8 GHz (peaked, single device, generation unclear [R01 D4]) |
| rom_dc_scale | reported per dλ/dT hypothesis (P1-2) |

### 4.7 Excluded, with reasons (`excluded.yaml`)

- The polarization-scramble codes (7,000 to 41,500) and the 7.7 dB swing: these are RX-path monitors (Fig. 2(e)
  caption), and the low code is an ADC offset, not a light level [R01 D8].
- All BER traces, as evidence of plant accuracy: the receive link budget dominates, and pass/fail at 1e-9 cannot tell
  plants apart. They are used only for feasibility (BND-2, BND-3).
- The Fig. 3(b) temperature waveform: it is plot-sampled and aliased; identical ramps peak anywhere from 95 to 113 °C.
- The 2.85 °C half-width and 0.67 dB/°C²: both are interpolated from 5 °C samples (kept in the closure table only).
- The ≥ 14.1 nm FSR bound: the traces are about 1 nm wide windows [R01 D2].
- "0.9 Vpp" as the junction swing: it is the pattern-generator swing into the driver [R01 D3].
- G25 modulation IL of 4.9 dB "at about +4 dBm": the operating detuning is not stated; a sanity check only.
- Textbook-derived values (4.7 K/mW, 0.67 dB/°C², 9.0 K/mW heater self-heating, the 29 µs loop lag): arithmetic on
  the same public numbers. Using them as targets would be circular.

### 4.8 The uncertainty set, and how controllers are judged

Initial `uncertainty:` entries (one draw per world, shared by all rings):

| Entry | Kind | Values | Basis |
|---|---|---|---|
| thermo_optic | hypothesis | H_solver (default), H86, H86_sensor | D2 |
| waveguide_thickness_nm | hypothesis | 160 (default), 200 | D21 |
| thermal.family | hypothesis | in-ring no undercut (default), GF-PDK silicide with undercut, metal above | P1-3 |
| thermal.time_scale_fast, time_scale_slow | interval | 0.5 to 2 each (on top of the family) | [R04] |
| thermal.package_tail | interval | 1 to 10 ms, 2 to 10 % of DC gain | [R04] |
| thermal.transport_delay_s | interval | 0 to 2 µs | [VAL] |
| thermal.crosstalk | interval | nearest 1 to 10 %, floor 0.5 to 2.6 %; textbook 10 %/same-dynamics corner | [R04] |
| thermal.stack | hypothesis | side-by-side (default), stacked EIC | [R04] |
| self_heating.eta (via the loss budget) | interval | 0.5 to 0.95; stress corner 0.3 | D1 |
| self_heating.z_ratio | interval | 1 to 2 | [PHY], [R06 H2] |
| power_at_ring_dbm | interval | 0 to +8 (nominal +4) | G25; 10 dBm/λ laser [R01] |
| q_loaded | interval | 3500 to 4200; low-Q corner 2600 | CAL-1, [R06] |
| coupling_regime | hypothesis | under, over | P1-5 |
| er_mechanism | hypothesis | dispersion, ring_t_loss, heater_t_loss | P1-5 |
| fsr | hypothesis | window B (default), window A; stress 1.53 and 1.60 THz | P1-6 |
| heater.driver_mode | hypothesis | voltage, current, power, dac_pwm | P1-11 |
| heater.tcr | interval | 0.5e-3 to 4.5e-3 /K, tied to the family | [R02] |
| monitor.gain_scale, monitor.dark_i0 | interval | 0.5 to 2; 1 to 30 nA | D7, D20 |
| loop latency | interval | 0.5 to 2 times the declared total | [CTRL] |
| aggressor.referencing, aggressor.waveform | hypothesis | resonance (default) or temperature; slide-10 parabolic (default) or symmetric triangle | D6 |

Corner presets are thin `extends:` files that select alternatives only: `g26_slow_thermal`, `g26_fast_thermal`,
`g26_undercut`, `g26_h86`, `g26_h86_sensor`, `g26_overcoupled`, `g26_heater_loss`, `g26_hot_selfheat`,
`g26_low_eta`, `g26_fsr_window_a`, `g26_fsr_edge`, `g26_textbook_thermal`.

Judging a controller. `experiment.compare` reports the nominal result, the worst named corner, quantiles over 256 to
512 Latin-hypercube worlds with metrics-only recording (to stay inside 16 GB), and the Kendall-τ stability of the
ranking across worlds. The primary acceptance metrics are the OMA penalty against the oracle at the same instant, the
lock-excursion count and duration, the wrong-line flag and the time to lock. Absolute relOMA is reported. A claim is
published only if it holds over the set. The first line of every controller report says that performance is
conditional on the assumed thermal dynamics until Q1 is answered.

### 4.9 Reports, golden traces and reproducibility

- `python -m mrm.validation.report` regenerates `docs/VALIDATION.md` with fixed sections:
  1. the target table (id, role, measured ± σ, predicted nominal and band, z-score, tolerance, pass/fail);
  2. the closure and conflict table (4.6);
  3. identifiability: the Jacobian of the calibration replicas, its SVD, the profile likelihoods, and the half-prior
     rule;
  4. uncertainty propagation: a tornado chart per entry for each controller metric, and Sobol indices for the top 8
     dimensions (SALib, MIT licence);
  5. controller rank stability across worlds;
  6. the generation mix of each derived quantity (for example, g26 takes Q, ER, junction and heater from G25, and the
     monitor, DAC and grid from G26);
  7. the textbook disagreements, generated from the `disagrees:` fields.
- Golden traces: `tests/golden/<preset>__<scenario>.npz`, metrics only, for static lock, the fast aggressor and the
  compressed soak. They change only through `mrm golden --update`, with a CHANGELOG line.

---

## 5. Questions for Lightmatter, ranked by expected reduction in controller-relevant model error

Answers go to a git-ignored `mrm/params/presets/private/` with provenance `private`, so the public validation stays
reproducible.

1. **Heater step response at the lock point.** A small heater-code step, with the monitor ADC trace at full rate from
   1 µs to 100 ms. Repeat for the adjacent ring's heater and for the on-die aggressor heater. Also: the heater type
   and position (doped Si in the ring, silicide resistors, metal above), undercut or not, and whether the G26 ring
   derives from the GF PDK ring. *Pins:* τ_fast, τ_slow, the slow share, the transport delay, crosstalk and its
   dynamics, the aggressor path, the heater family. All are assumed today, and they span about 10 times in
   achievable loop bandwidth.
2. **One open-loop static heater sweep of the TX drop monitor.** ADC code against DAC code across more than one FSR,
   swept up and down, at 2 to 3 laser powers and a known stage temperature, repeated at +10 °C. *Pins:* FWHM and FSR
   in codes, the lumped monitor gain, the driver-law curvature, self-heating (asymmetry and hysteresis against
   power), dλ/dT in codes per K (H_solver against H86 in controller units), and the coupling branch.
3. **Controller I/O facts.** The loop update rate; the monitor ADC type (the patent's dual-slope integrator?), its
   integration window and sample rate, and any multiplexing; the ADC LSB in µA (or the code reading at the 15 µA of
   Fig. 3(b)); the heater DAC law and full scale (DAC plus PWM as in the patent?); code limits and slew; the sensing,
   compute and actuation latencies; whether the loop dithers, and on what; the lock set-point (peak, fraction of
   peak, blue edge); temperature sensors (type, rate, location); heater V/I readback; whether the loop also commands
   the bias. *Pins:* quantization, delay, the 2.5-times gain variation, and which controller classes are realistic.
4. **relOMA and the drive.** Confirm the reference plane ("1 mW of laser into the MRM", Hot Chips slide 16), the
   optical power per line at the MRM input, the junction swing (the driver supply), the G26 modulation efficiency,
   and what the 1.0 to 3.75 V bias axis is. *Pins:* the modulation depth (about 55 against 170 to 230 pm), the
   self-heating power, and whether CAL-14 is a real calibration.
5. **How "~15 GHz/°C" and "808 °C/s within the MRM" were measured.** Fixed code or locked; TEC, on-die sensor, or
   inferred from the stabilizer code or the resonance shift; the ring temperature at the time; what "period of
   100 ms" means; whether the 808 °C/s and > 2000 °C/s results use the same heater. *Pins:* the disturbance scale
   (1.2 to 1.3 times between H86 and H_solver) and the slope (up to 2 times).
6. **The G26 TX ring itself.** SOI thickness (160 or 200 nm), strip or rib and the slab, the radius at the waveguide
   centre, the measured FSR, the loaded Q, under- or over-coupled, the drop coupling, and the doping level. *Pins:*
   the FSR window, η, the coupling branch, and the transfer of the ER mechanism from G25.
7. **The raw 5 °C × bias samples behind the OFC 2026 slide-9 relOMA maps,** and the heater code used for the
   loop-open map. This is the only held-out dataset that jointly tests dλ/dT, the FWHM and the modulation depth.
8. **Layout and crosstalk.** TX ring pitch; positions of the heaters, the aggressor heater, the monitor PDs, the
   drivers and the temperature sensors; any measured neighbour crosstalk in pm/mW, with and without a stacked EIC.
   *Pins:* the ROM cross terms, the data-activity heat path, the PD temperature.
9. **Laser and traffic.** Whether lines from one module drift together; RIN; power behaviour during squelch; the
   laser turn-on sequence; whether the TX feed is polarization-maintaining; whether traffic start/stop, idle and
   training patterns are known disturbances to the loop; the junction capacitance and series resistance. *Pins:*
   self-heating disturbances, data-activity heating, the reset protocol.
10. **Distributions of the lock code at 25 °C across lanes and dies,** the designed resonance stagger, and the heater
    R spread within a die. *Pins:* the variation model and the worst lane in BND-2.
11. **Bring-up policy.** Sweep range and direction (which matters with self-heating bistability), whether calibration
    is stored, whether order hops or barrel-shift remaps (patent PAT-1) occur over 0 to 105 °C, and how wrong-line
    locks are detected. *Pins:* the acquisition scenarios, the wrong-line metric, BND-2.
