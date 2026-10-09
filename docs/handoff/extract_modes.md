# Handoff: extract_modes (`mrm/extraction/common.py`, `modes.py`, `coupling.py`)

Status: see the numbers section (filled from the committed artefacts). Nothing is committed.

## Files

- `mrm/extraction/common.py`: cross-section hypotheses `XSECS` (rib160, rib200, strip400x200, decea2019), materials
  from `mrm.materials` only (`indices`, `group_indices`, `dn_dt`), gmsh meshing, the FEM mode solver `FemProblem`
  (femwell's formulation generalized to diagonal anisotropic eps and mu: exact bend transform, femwell's isotropic map
  for comparison, complex-coordinate PML), the Tidy3D local-solver wrapper `tidy3d_mode`, Richardson extrapolation,
  `write_artefact` (data + `<name>.sidecar.json`).
- `mrm/extraction/modes.py`: `python -m mrm.extraction.modes [--quick] [--xsec ...] [--study yes|no|only] [--out DIR]`.
  Writes `modes_<xsec>.json` (schema `modes/1`) and `modes_study_bend.json` (schema `modes_bend_study/1`).
- `mrm/extraction/coupling.py`: `python -m mrm.extraction.coupling [--quick] [--xsec ...] [--out DIR]`. Writes
  `coupling_<xsec>.json` (schema `coupling/1`). Needs `modes_<xsec>.json` for r_ref.
- `tests/verification/test_extraction_modes.py`: fast artefact tests + `@pytest.mark.slow` re-runs.
- `--quick` writes to a temporary directory unless `--out` is given; it never overwrites the committed artefacts.

## Conventions every consumer must know

1. **Bend formulation.** n_eff of the bent mode is referenced to r_ref (the ring centreline) and computed with the
   EXACT transformation-optics map (eps_t = eps rho, eps_s = eps/rho, mu_t = rho, mu_s = 1/rho). femwell's
   `compute_modes(radius=R)` uses eps (1 + x/R)^2 and overestimates the bend shift about 3.7 times; never use it for
   r_ref-referenced phases. The runtime phase is phi = 2 pi n_eff(lambda, T) 2 pi r_ref L(T)/L / lambda.
2. **r_ref** is self-consistent with the 1.8 THz design FSR: c / (2 pi r_ref n_g,bent(r_ref)) = 1.8 THz.
3. **dlambda/dT (H_solver)** = (lambda/n_g)(dn_eff/dT + n_eff alpha_L(T)) at lambda0 = 1310 nm (a locked ring sits
   at its line), alpha_L = substrate-clamped Si (Okada & Tokumaru). Table at 0, 25, 50, 75, 105, 125 C.
4. **Index weights** are dn_eff/dn_i by central differences (the perturbation integral agrees to < 2e-4). The power
   fraction is NOT the thermo-optic weight; dn_eff/dn_Si > Gamma_power for these guides.
5. **coupler_phase_rad** is the ring-side self-coupling phase DELAY relative to the uncoupled bent guide over the same
   arc (-arg t_ring for fields exp(j(wt - beta z))), so it adds to phi (`docs/handoff/plant_optics.md`, psi).
6. **kappa2** is per pass (power). Q_e = omega T_rt / (-ln(1 - K^2)), T_rt = 1/FSR. Evaluate at each line:
   K^2(lambda_l) = K^2_0 exp(dln * (lambda_l - lambda0)), or interpolate `kappa2_table`.
7. **Jacobians** use the names of `mrm.params.realize.JACOBIAN_TARGETS` / `_JAC_DIMS`. They are partial derivatives
   at fixed r_ref (modes) and at fixed edge-to-edge gaps (coupling). A lithographic width bias delta widens both
   guides and closes each gap by delta: total d ln K^2 = (d/dw - d/dgap) delta.

## Numbers: modes (lambda0 = 1310 nm, T_ref = 25 C, TE0 bent at r_ref)

| | rib160 (default) | rib200 | strip400x200 (textbook) |
|---|---|---|---|
| geometry (nm): width / film / slab | 400 / 160 / 60 | 400 / 200 / 80 | 400 / 200 / 0 |
| r_ref for 1.8 THz (um) | 6.7270 | 6.6076 | 6.0948 |
| n_eff (bent) | 2.43993 | 2.62968 | 2.49200 |
| n_g (bent) | 3.9405 | 4.0116 | 4.3492 |
| n_g geometry spread (RSS, width +-45, film +-10, slab +-20 nm) | +-0.11 | +-0.04 | +-0.05 |
| dn_g/dlambda (1/nm) | -1.05e-3 | -7.1e-4 | +1.4e-4 |
| dlambda/dT 0/25/50/75/105/125 C (pm/K) | 62.59 / 65.08 / 67.38 / 69.51 / 71.84 / 73.26 | 64.37 / 66.91 / 69.26 / 71.43 / 73.80 / 75.24 | 63.30 / 65.80 / 68.11 / 70.25 / 72.59 / 74.02 |
| of which expansion (pm/K, 25 C) | 2.1 | 2.2 | 1.9 |
| sigma from Si dn/dT (pm/K, 25 / 105 C) | 2.3 / 3.2 | 2.3 / 3.3 | 2.3 / 3.3 |
| beta of s(T)=s0(1+beta(T-T0)) (1/K, at 25 / 105 C) | 1.47e-3 / 1.02e-3 | 1.46e-3 / 1.01e-3 | 1.46e-3 / 1.02e-3 |
| dn_eff/dn: Si / BOX / clad | 0.9684 / 0.1021 / 0.1489 | 1.0167 / 0.0679 / 0.1100 | 1.0857 / 0.0679 / 0.1666 |
| identity sum n_g,i w_i - n_g | -2.2e-7 | -1.6e-7 | -2.8e-7 |
| Poynting power in Si (rib + slab) | 0.690 (0.654 + 0.036) | 0.775 | 0.807 |
| electric-energy fraction in Si | 0.903 | 0.933 | 0.918 |
| mode area (Poynting, um^2) | 0.105 | 0.108 | 0.078 |
| alpha_rad at r_ref (dB/cm), Q_rad | 0.119, 6.9e6 | 1.03, 8.1e5 | < 1e-6 (noise floor) |
| alpha_rad vs slab 50/60/70/80/90/100 nm (dB/cm) | 0.0013 / 0.119 / 4.7 / 79 (approx.) / - / lost | 7.8e-7 / 2.1e-4 / 0.023 / 1.03 / 19.3 / 172 | - |
| Si level alternative (Li): Delta n_eff, Delta lambda_res | -0.00395, -1.31 nm | -0.00415, -1.36 nm | -0.00443, -1.34 nm |
| R = 7.5 um: n_g, FSR, dlambda/dT, alpha_rad | 3.9415, 1.614 THz (9.24 nm), 65.09, 0.022 | 4.0125, 1.586 THz (9.08 nm), 66.93, 0.19 | 4.3488, 1.463 THz (8.37 nm), 65.80, 0 |
| TM0 | straight 1.624 < TE slab 1.946: leaks; not guided | straight 1.915 < TE slab 2.166: leaks | guided: 1.8975 / bent 1.9000, n_g 4.29, alpha_rad 0.08 dB/cm |
| TE1 | straight 1.9515 (just above the slab), bend 3000 dB/cm: not guided | straight 2.173, bend 7500 dB/cm: not guided | straight 1.496 (near cut-off), not guided in the bend |

Jacobians (per nm, at fixed r_ref; inputs width, film, slab):

| output | rib160 | rib200 | strip400x200 (width, film) |
|---|---|---|---|
| lambda_res_nm_per_nm | 0.402, 1.390, 0.951 | 0.368, 0.913, 0.787 | 0.652, 1.190 |
| n_g_per_nm | 3.2e-4, 5.9e-3, -4.8e-3 | 1.1e-4, 2.7e-3, -1.2e-3 | -2.4e-3, 5.8e-4 |
| dlambda_dt_pm_per_k_per_nm | 0.018, 0.061, 0.027 | 0.013, 0.025, -0.041 | 0.026, 0.038 |
| alpha_rad_db_per_cm_per_nm | -0.006, -0.024, 0.236 | -0.042, -0.116, 0.964 | 0, 0 |

**alpha_rad is exponential in the slab**: the linear Jacobian is valid only within about +-5 nm. Use the
`slab_sweep` table (log-interpolate) for larger excursions. A slab of 80 nm on the 160 nm film would radiate
~80 dB/cm at r_ref, most of the 107-128 dB/cm intrinsic budget (07 D1), so the slab is bounded at about 70 nm if
Q_i is 7000-8400 and the loss is mostly absorption (`P0-2` loss budget: `alpha_bend_radiation_db_per_cm`).

Convergence (Richardson over h_core 30/20/15/10 nm, recorded in each sidecar): error of n_eff 1.4e-5 to 2.3e-5, of n_g
5e-6 to 2.7e-4, of dn_eff/dT 1e-9 (5e-6 relative), of the Si index weight 7e-6 to 3e-5. PML loss spread over four PML
and window settings: 0.09 % (rib160), 0.03 % (rib200).

Cross-checks:
- Tidy3D local solver (interface-aligned grid, 10/5/2.5 nm, Richardson): n_eff delta (FEM - Tidy3D) -1.6e-4
  (rib160), +2.5e-4 (rib200), -1.9e-4 (strip). Bend shift at r_ref: FEM 0.003032 / 0.003896 / 0.000575 against
  Tidy3D 0.003020 / 0.003877 / 0.000580 (finest grid). femwell's isotropic map gives 0.006398 / 0.007885 / 0.002147.
- Bend formulation (`modes_study_bend.json`): exact transform vs analytic 2-D Bessel eigenvalue < 1e-7 (isotropic
  off by 3.6e-3 to 4.4e-3 at R = 3 um); Tidy3D +-R mean equals the exact shift (0.000378 vs 0.000378 at 7.5 um);
  cylindrical Meep (3-D vectorial, averaged, 30-120 px/um, first order, Richardson) 1.30758 um vs FEM exact
  1.30750 um and isotropic 1.30954 um at R = 3 um, m = 36. Verdict: the exact transform; femwell's isotropic map is
  wrong by 3.7x at 7.5 um.
- PML radiation loss vs the exact complex-frequency (Hankel) solution of a 2-D ring: Q_rad within 0.2 % for
  Q_rad > 2e4 (R = 4 and 5 um). Tidy3D's local solver with bend + PML returned only PML modes (tested, not used).
- 07 P0-3 test_dldt_perturbation_vs_resolve: sum w_i dn_i/dT agrees with the re-solved dn_eff/dT to < 1e-4.

## Numbers: couplers (lambda0 = 1310 nm, ring at r_ref, straight bus of the ring's cross-section)

Design targets: Q_i = 7700 (7000-8400), kappa_d/kappa = 3 % (prior 0.4-18 %), FSR 1.8 THz. The through gap is the
critical-coupling gap (kappa_e = kappa_i + kappa_d), which gives Q_L = 3619 (inside 3500-4200). The mirror branches
are solved at Q_L = 3850 with the G25 notch depth 27.9 dB: kappa_i' + kappa_e = kappa, |kappa_i' - kappa_e| = r kappa.

| | rib160 | rib200 | strip400x200 |
|---|---|---|---|
| through gap (nm), K^2 per pass | 183.1, 0.1045 | 161.8, 0.1045 | 50.5 (extrapolated below the table), 0.1034 |
| under / over-coupled gap (nm) at Q_L 3850, 27.9 dB | 190.4 / 184.7 | 168.9 / 163.3 | 55.8 / 51.6 (extrapolated) |
| drop gap (nm) for kappa_d/kappa 0.4 / 3 / 18 % | 525 (extrapolated) / 381.9 / 255.1 | 499 / 357.8 / 232.7 | 285.9 / 189.5 / 102.9 |
| drop K^2 per pass (nominal) | 0.0066 | 0.0066 | 0.0066 |
| d ln K^2 / d lambda (through / drop), %/nm | 0.767 / 1.226 | 0.661 / 1.058 | 0.712 / 1.247 |
| d ln K^2 / d gap (through / drop), %/nm | -1.346 / -1.406 | -1.367 / -1.424 | -1.678 / -2.080 |
| coupler_phase_rad (delay; through / drop) | -0.026 / -0.0036 | -0.032 / -0.0050 | +0.321 / +0.015 |
| d ln K^2 through per nm of width / film / slab | -0.0145 / -0.0221 / +0.0260 | -0.0139 / -0.0163 / +0.0142 | -0.0169 / -0.0100 / - |
| d ln K^2 drop per nm of width / film / slab | -0.0189 / -0.0368 / +0.0481 | -0.0182 / -0.0269 / +0.0375 | -0.0223 / -0.0169 / - |
| power conservation of the CMT over all gaps | 0.9983-0.9998 | 0.9991-1.0000 | 0.9999-1.0000 |

Method validation:
- Convergence (rib160, through gap): K^2 = 0.10434 / 0.10450 on interface-aligned 10 / 5 nm overlap grids, 0.10449
  with dz halved; FEM mesh 20 / 15 / 10 nm: 0.10472 / 0.10450 / 0.10443. (A 7.5 nm grid, not aligned with the
  interfaces, gives 0.1018: pixelation of the cores, not a convergence trend.)
- 2-D full-wave check (Meep FDTD of a ring arc and bus in the rib160 effective-index model, 30-80 px/um, the two
  finest averaged): K^2 = 0.2128 +- 0.0006 at 150 nm and 0.02596 at 300 nm. Method A (this CMT) / Meep = 0.994 and
  1.024; method B (supermode + curved gap, centreline phase) / Meep = 0.962 and 0.949. Method A is used.
- Method B in 3-D at the design gaps: 0.0985 (through, A 0.1045) and 0.00613 (drop, A 0.00660); its dispersion
  0.780 %/nm agrees with A's 0.767 %/nm. Ignoring the bent/straight phase mismatch changes B's K^2 by +3 %.

- Strip (oxide gap, high contrast): the same 2-D check gives A/Meep = 0.815 at 100 nm and 0.937 at 200 nm (Meep
  converged to 0.2 %: 0.0998 and 0.0129). Method A underestimates K^2 by 6-18 % there, i.e. the strip's design gaps are
  about 4-10 nm too small. Method B is worse (0.86 and 0.54): its quadratic fit of ln kappa(g) is extrapolated far
  along the curved gap. The strip needs a ~50 nm gap (or a pulley bus) to reach critical coupling at this Q_L, which
  is itself a reason to prefer the rib hypothesis.

What it means for 07 P1-5: the coupler dispersion on the rib160 cross-section is 0.77 %/nm (through), half of the
femwell smoke number (1.3-1.5 %/nm, strip with an oxide gap) and about a quarter of the 2.9 %/nm that would explain
the G25 extinction droop by dispersion alone. The ER droop therefore needs a loss mechanism (`er_mechanism`
ring_t_loss or heater_t_loss), unless the real coupler differs strongly (pulley bus, no slab in the gap). The two
mirror branches are only ~6 nm of gap apart, well inside fabrication spread, so the coupling regime cannot be
inferred from a drawn gap.

## VAL-7 replication: de Cea, Atabaki, Ram, Opt. Express 27, 24274 (2019)

Re-checked in the paper (Table 1, section 3): ring outer radius 10 um, 1.7 um wide, "roughly 100 nm thick", zero-change
45 nm SOI, lambda0 1545.2 nm; n_g = 2.73 is derived from the measured FSR (the radius used is not stated);
Gamma_ring = 0.6515 is a FEM value with the Poynting normalization of Robinson et al. 2008 (= dn_eff/dn_Si);
gamma_th = 2.4 MHz. Model (`modes_decea2019.json`; BOX 145 nm on air below, oxide above, both ASSUMED):

| Si thickness (nm) | n_g (centreline 9.15 um) | n_g referred to the 10 um outer radius | dn_eff/dn_Si |
|---|---|---|---|
| 80 | 3.040 | 2.782 | 0.640 |
| 90 | 3.201 | 2.929 | 0.709 |
| 100 (printed, nominal) | 3.338 | 3.055 | 0.768 |
| 110 | 3.454 | 3.161 | 0.819 |
| 120 | 3.551 | 3.249 | 0.862 |

Honest verdict: at the printed 100 nm the pipeline does NOT reproduce the paper (n_g +12 % with the outer-radius
convention, +22 % at the centreline; Gamma +0.12; VAL-7 tolerances are 3 % and 0.05). Both printed numbers are
reproduced together at one thickness, about 77-82 nm, if the paper's FSR-derived n_g refers to the outer radius. The
pipeline is consistent with the paper only under that reading; the thickness is the dominant unknown.

## Assumptions (all ASSUMED unless stated; sweep or Jacobian given)

- Rib width 400 nm (render 400, BU 45CLO 410), slab 60 nm on 160 and 80 nm on 200 (same as `junction.py`),
  vertical sidewalls, BEOL oxide (n 1.450) above, thermal BOX (1.4472) below, substrate omitted (2 um BOX: leakage
  < 1e-14). Undoped silicon (the junction's index change is in `junction_<xsec>.json`).
- Ring length expands with the substrate-clamped Si alpha_L; cross-section expansion and stress-optic terms are
  neglected (< 0.5 % of dn_eff/dT from the Jacobians).
- Couplers: straight bus of the ring's own cross-section, slab continuous through the gap (rib), drop identical to
  the through coupler on the opposite side. A pulley bus or a slab-free gap would change K^2(gap) and its dispersion.

## Open issues

1. The slab depth is the dominant optical unknown: it moves lambda_res by ~1 nm/nm, n_g by -0.005/nm and
   alpha_rad exponentially (0.12 -> 79 dB/cm from 60 to 80 nm on 160). The loss budget bounds it (above).
2. VAL-7 fails at the printed thickness (see above); passes only at ~80 nm with the outer-radius n_g convention.
3. Tidy3D's local solver cannot cross-check bend radiation (PML modes only); the FEM-PML is validated in 2-D instead.
4. The coupler phase (-0.026 rad through) is a CMT quantity with non-orthogonality corrections; it shifts the
   resonance by ~0.04 nm and is absorbed by the calibrated resonance offset. Method B gives a similar magnitude.
5. coupling K^2 is evaluated at 25 C. A ring-bus temperature mismatch is excluded with a reason in 07 P1-5 (< 1 %).
6. No sidewall-angle sensitivity was computed (nominal 90 deg); `XSec.sidewall_deg` supports it in the FEM, not in
   the Tidy3D wrapper.

## Handoff requests

- **params / presets (07 P0-3, P1-6):** H_solver s0 at 25 C = 65.08 pm/K (rib160; 66.91 rib200), beta = 1.47e-3 /K
  around 25 C (1.02e-3 around 105 C), from `extracted/modes_<xsec>.json:dlambda_dt_pm_per_k` (table node, `at:
  {t_c: ...}`). `ring.geometry.radius_um` = r_ref_um; `ring.optics.n_eff` = n_eff, `n_g`, `dn_g_dlambda_per_nm`;
  `ring.geometry.jacobian` = `jacobian`; `ring.geometry.jacobian_coupling` = coupling `jacobian`.
- **loss budget (07 P0-2):** `alpha_bend_radiation_db_per_cm` = `alpha_rad_db_per_cm` (0.12 dB/cm rib160, 1.0 rib200).
  Use `slab_sweep` rather than the linear Jacobian for slab excursions beyond +-5 nm.
- **coupling (07 P1-5, P1-9):** `dln_kappa2_through_dlambda_per_nm` = 0.0077 /nm (rib160), drop 0.0123 /nm;
  `dln_kappa2_dgap` -0.0135 /nm (so +-5 nm of drop gap is +-7 % of monitor gain, not +-10 %).
- **plant_optics:** `coupler_phase_rad` follows your psi convention (delay, adds to phi).
- **realize:** the coupling Jacobian is partial at fixed gap; a correlated width bias needs d/dw - d/dgap.
- **validation (VAL-7):** use the table above; VAL-7 as frozen (n_g 2.73 +-3 %, Gamma 0.65 +-0.05) fails at 100 nm.
- **docs (07 D10, 05 section 5.4):** the femwell-vs-Tidy3D bend disagreement is settled in favour of the exact
  transform (three independent checks); cylindrical Meep with subpixel averaging converges at first order and
  agrees after Richardson extrapolation.
- **tools/memguard.py (owner unknown):** the heavy lock has no queue; with 5-6 waiters one agent's sequential jobs
  can starve others for an hour. A FIFO ticket would help.
