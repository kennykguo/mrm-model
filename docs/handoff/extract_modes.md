# Handoff: extract_modes (`mrm/extraction/common.py`, `modes.py`, `coupling.py`)

Status (second resumed run, 2026-10-09): finished from the results that existed when a session restart stopped the
first run. The data files are committed (0a9941e, df1f69e; `modes_study_bend.json` in 812fd1c) and are unchanged.
Neither resumed run started a solver (no femwell, Tidy3D, scikit-fem, DEVSIM or Meep). The second resumed run filled
sidecar fields from stored data only (see "Memory cap" and "Sidecar fields added"), added CLI flags for the first-design
meshes, and added tests; none of that is committed yet. A review pass (also no solver) then corrected the stated
uncertainty of the Tidy3D and Meep cross-checks, labelled the assumed values that have no range, fixed a CLI check and
made one test robust to re-extraction (see "Review corrections"). Parts of the original task that were NOT done are
listed in "Not done", each with the command that would produce it and the runtime fallback used meanwhile. Fast tests:
`pytest tests/verification/test_extraction_modes.py -m "not slow"` gives 85 passed (also with `-W error`). Every
number below was re-read from the committed artefacts.

## Files

- `mrm/extraction/common.py`: cross-section hypotheses `XSECS` (rib160, rib200, strip400x200, decea2019), materials
  from `mrm.materials` only (`indices`, `group_indices`, `dn_dt`), gmsh meshing, the FEM mode solver `FemProblem`
  (femwell's formulation generalized to diagonal anisotropic eps and mu: exact bend transform, femwell's isotropic map
  for comparison, complex-coordinate PML), the Tidy3D local-solver wrapper `tidy3d_mode`, Richardson extrapolation,
  `write_artefact` (data + `<name>.sidecar.json`).
- `mrm/extraction/modes.py`: `python -m mrm.extraction.modes [--quick] [--xsec ...] [--study yes|no|only] [--out DIR]
  [--mesh-scale S] [--conv-scales S1 S2 ...] [--meep-pred-scale S]` (defaults 1.5; 3.0 2.0 1.5 1.0; 1.0).
  `production_mesh_bias` (pure arithmetic) writes the bias of the reported production-mesh values into the sidecar.
  Writes `modes_<xsec>.json` (schema `modes/1`) and `modes_study_bend.json` (schema `modes_bend_study/1`).
  `--assemble-study QUICK_STUDY_JSON MEEP_CYL_JSON` writes the study from stored results with no solver run
  (`assemble_bend_study`); `meep_series_verdict` is the pure Richardson analysis of the cylindrical Meep rows that both
  paths use.
- `mrm/extraction/coupling.py`: `python -m mrm.extraction.coupling [--quick] [--xsec ...] [--out DIR] [--mesh-scale S]
  [--dx-nm D]` (defaults 1.5 and 5 nm; the convergence rows follow the chosen levels). Writes
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
   fraction is NOT the thermo-optic weight; dn_eff/dn_Si > Gamma_power for these guides. The keys are `si`, `box`
   and `clad` (plus `air` for de Cea), not the `sio2` of the ARCHITECTURE 5.1 example: the BOX and the cladding have
   different indices and dn/dT. For a common oxide index change use `box + clad` (0.251 rib160, 0.178 rib200, 0.234
   strip).
4a. **n_g and the polynomial.** `n_g` and `dn_g_dlambda_per_nm` are local derivatives at lambda0 (finite differences
   with a Richardson step); `n_eff_poly_lambda` is a quadratic fit over 1260-1360 nm, so c0 - lambda0 c1 differs from
   `n_g` by 2.6e-4 (rib160). The runtime (`mrm/plant/phase.py`) builds n_eff(lambda) from n_eff, n_g and dn_g/dlambda,
   which agrees with c1 and c2 to 0.02 % and 0.3 %.
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
| n_g geometry spread (RSS of Jacobian x half-width; width +-45 (strip +-20), film +-10, slab +-20 nm) | +-0.11 | +-0.04 | +-0.05 |
| same with the slab half-range of `ranges_nm` (rib160 40-100: +-30; rib200 50-120: +-35) | +-0.16 | +-0.05 | - |
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

**alpha_rad is exponential in the geometry, and its Jacobian entry is a secant.** `alpha_rad_db_per_cm_per_nm` is
the central difference over the stored steps (width +-10, film +-5, slab +-10 nm), not the local slope. From the same
stored evaluations (`jacobian.evaluations`, no new solve), d ln(alpha_rad)/dp and the local slope alpha_0 d ln/dp are:

| | width | film | slab |
|---|---|---|---|
| rib160: d ln alpha_rad / dp (1/nm) | -0.0485 | -0.1785 | +0.4104 |
| rib160: local slope (dB/cm/nm); secant / local | -0.0058; 1.05 | -0.0212; 1.14 | +0.0488; 4.83 |
| rib200: d ln alpha_rad / dp (1/nm) | -0.0398 | -0.1075 | +0.3356 |
| rib200: local slope (dB/cm/nm); secant / local | -0.0409; 1.03 | -0.1105; 1.05 | +0.3449; 2.79 |

The linear offset can give a negative radiation loss: on rib160 a film draw of +8.4 nm (2 sigma of the preset's die
spread) gives 0.119 - 0.024 x 8.4 = -0.085 dB/cm, where the exponential gives +0.027 dB/cm. The effect on the total
loss (~110 dB/cm) is below 0.1 %, but the sign is unphysical (handoff request to realize below). For slab excursions use
the `slab_sweep` table (log-interpolate) or exp(d ln alpha/dp x delta).

A slab of 80 nm on the 160 nm film would radiate ~80 dB/cm at r_ref, most of the 107-128 dB/cm intrinsic budget
(07 D1), so the slab is bounded at about 70 nm if Q_i is 7000-8400 and the loss is mostly absorption (`P0-2` loss
budget: `alpha_bend_radiation_db_per_cm`).

Convergence (Richardson over h_core 30/20/15/10 nm, recorded in each sidecar): error of the FINEST (10 nm) level, which
is what `discretization_error` in the data files holds: n_eff 1.4e-5 to 2.3e-5, n_g 5e-6 to 2.7e-4, dn_eff/dT 1e-9
(5e-6 relative), Si index weight 7e-6 to 3e-5. The reported values come from the 15 nm production mesh, whose bias
against the Richardson limit is 3-4x larger on the ribs (`validation.production_mesh_bias` in each sidecar):

| bias f(15 nm) - f0 | rib160 | rib200 | strip400x200 | decea2019 |
|---|---|---|---|---|
| n_eff (bent) | +7.6e-5 | +5.4e-5 | +1.6e-5 (p not found) | -1.6e-5 (p not found) |
| lambda_res at fixed r_ref (nm) | +0.025 | +0.018 | +0.005 | -0.008 |
| n_g (bent) | -3.6e-5 | -2.0e-5 | -3.3e-4 (p 0.46) | -6.3e-5 (p not found) |
| dn_eff/dT (1/K) | +5.2e-9 | +4.0e-9 | -9.4e-9 | -3.5e-9 |
| Si index weight | +3.2e-5 | +2.5e-5 | -4.6e-5 | -1.9e-5 |

All are far below the geometry uncertainty (lambda_res moves ~1 nm per nm of slab; n_g spread +-0.04 to +-0.11) and the
lambda_res bias is absorbed by the calibrated resonance offset. "p not found": the three finest levels are not in the
asymptotic range, so f0 is the 10 nm value and the bias is the 15-to-10 nm change; that is a lower estimate, because
the 10 nm value itself carries an error of about the same size (up to ~2x the tabulated bias). PML loss spread over four PML
and window settings: 0.09 % (rib160), 0.03 % (rib200). Limits of this evidence: the strip's n_g series is not yet
asymptotic (observed order 0.46; error estimate 2.7e-4, the largest of the set); alpha_rad has PML/window convergence
but no mesh-refinement series; the Jacobians use one finite-difference step each (no step-size study).

Cross-checks:
- Tidy3D local solver (interface-aligned grid, 10/5/2.5 nm, Richardson): n_eff delta (FEM - Tidy3D limit) -1.6e-4
  (rib160), +2.5e-4 (rib200), -1.9e-4 (strip). **These deltas are not the accuracy of the cross-check.** Tidy3D
  converges at about first order here (fitted order 1.18 / 1.01 / 1.07), so its finest grid (2.5 nm) is still
  1.9e-3 / 2.2e-3 / 4.3e-3 from its own limit (0.64 / 0.72 / 1.28 nm of lambda_res), and a first-order fit of the two
  finest grids moves the limit by 5.1e-4 / 1.7e-5 / 4.2e-4 (delta +3.5e-4 / +2.7e-4 / +2.3e-4). So the FEM n_eff
  agrees with Tidy3D within the Tidy3D extrapolation uncertainty, a few 1e-4 (~0.1-0.2 nm of lambda_res) by the order
  sensitivity and < 4.3e-3 by the conservative bound; it is not verified at the 2e-4 level by this check
  (`validation.crosscheck_tidy3d_uncertainty` in each sidecar; the FEM's own Richardson limit gives deltas of
  -2.4e-4 / +2.0e-4 / -2.1e-4). The sharper check is the bend shift (bent - straight on the same grid, which converges
  fast): FEM 0.003032 / 0.003896 / 0.000575 against Tidy3D 0.003020 / 0.003877 / 0.000580 (finest grid; +0.4 / +0.5 /
  -1.0 %). femwell's isotropic map gives 0.006398 / 0.007885 / 0.002147.
- Bend formulation: see "Bend-formulation study" below. Verdict: the exact transform; femwell's isotropic map
  overestimates the bend shift 3.7x at 7.5 um.
- PML radiation loss vs the exact complex-frequency (Hankel) solution of a 2-D ring (rib160 effective-index model):
  Q_rad 27527 (FEM, four PML/window settings within 2e-4) against 27468 (exact), +0.21 %, at R = 4 um. Only R = 4 um
  is stored; the interrupted run's statement "R = 4 and 5 um" is not backed by a stored number. Tidy3D's local solver
  with bend + PML returned only PML modes (tested, not used).
- 07 P0-3 test_dldt_perturbation_vs_resolve: sum w_i dn_i/dT agrees with the re-solved dn_eff/dT to < 1e-4.

## Bend-formulation study (`modes_study_bend.json`, assembled from stored results)

The full study (`--study only`) was waiting for the heavy lock when the session restarted, so it never ran. The resumed
run assembled the artefact with `python -m mrm.extraction.modes --assemble-study <quick study JSON> <Meep rows JSON>`
from: the `--quick` study (Bessel and PML-Hankel benchmarks), the stored cylindrical Meep runs (9 runs, 30-120 px/um),
the committed `modes_<xsec>.json` (production FEM and Tidy3D at r_ref; the strip at 7.5 um) and the Tidy3D +-R check
of 05 (`05_tools_smoke/verify/v01_tidy3d_bend_asym.json`). The sidecar lists every input with its sha256 (the two
scratchpad inputs are embedded in the artefact, so the artefact is self-contained).

| check | exact transform | femwell isotropic map |
|---|---|---|
| 2-D ring vs analytic Bessel eigenvalue (R = 3 um, h = 10 nm, both polarizations) | max error 6.4e-8 | 3.6e-3 / 4.4e-3 |
| 3-D bend shift vs Tidy3D +-R mean, strip at r_ref 6.095 um | 0.000575 vs 0.000580 (extrapolated 0.000576) | 0.002147 |
| strip at 7.5 um (Tidy3D: 05 V1, constant indices, 20/10/5 nm) | 0.000379 vs 0.000378 | 0.001418 (3.74x) |
| rib160 / rib200 at r_ref | 0.003032 / 0.003896 vs 0.003020 / 0.003877 | 0.006398 / 0.007885 |
| largest exact-vs-Tidy3D difference of the shift (5 rows) | 1.0 % | ratio >= 2.03 |
| isotropic excess shift, r_ref vs 7.5 um | ratio 1.5139 against 1/R^2 = 1.5143 | |
| cylindrical Meep, R = 3 um, m = 36 (3-D vectorial, subpixel averaging, observed order 0.86-1.02) | 1.30750 um vs Meep 1.30750 +- 0.23 nm (narrow) / +- 2.1 nm (order spread) | 1.30954 um: 2.04 nm away; needs order 1.19 between the two finest runs, outside the observed 0.86-1.02 |

Meep: the estimate is the mid-range of three Richardson extrapolations of the averaged series (free order on 120/80/60
px/um: 1.30727; first order on 120/80: 1.30759; first order on 80/60: 1.30772); the +-0.23 nm is their half-range.
That number does NOT include the order uncertainty (review correction): the finest run (120 px/um) is still 10 nm
above the estimate, and 0.1 in the order moves the limit by 1.2 nm. Free-order fits of every consecutive triple give
orders 0.97 / 0.86 / 1.02 and limits 1.30727 / 1.30489 / 1.30918 um, a half-range of 2.1 nm, about the size of the
2.04 nm exact-isotropic difference (`validation.meep_order_sensitivity` in the study sidecar). The exact prediction
lies inside that range and needs order 0.99 between the two finest runs; the isotropic one lies 0.36 nm outside and
needs 1.19. So the Meep check is consistent with the exact transform and disfavours the isotropic map, but it is not
the "9x the uncertainty" result stated before; the verdict rests on the analytic Bessel benchmark (6.4e-8 vs 3.6e-3)
and the Tidy3D bend shifts (within 1 % vs a ratio >= 2). A 160 px/um run would narrow the order spread (not run, per
Kenny).
Without averaging the series is not monotone (staircasing) and is not used. The FEM predictions of the Meep resonance
(exact 1.30750, isotropic 1.30954 um) come from the interrupted run (recorded in this handoff; the output file was not
kept) and were not recomputed. They are consistent with the stored data: the isotropic excess shift at 7.5 um, scaled by
(7.5/3)^2 and converted with n_g 4.142 (05 t01), predicts a 2.049 nm difference against the recorded 2.040 nm (0.4 %).

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
- rib200 and strip400x200 have the same overlap-grid and FEM-mesh convergence rows in their sidecars (rib200: 0.10426 /
  0.10449 on 10 / 5 nm grids, 0.10461 / 0.10449 / 0.10440 on FEM 20 / 15 / 10 nm; strip: 0.10321 / 0.10339 and
  0.10348 / 0.10339 / 0.10318). The overlap grid has only two interface-aligned levels (10 and 5 nm); the FEM mesh has
  three. rib200 has method B but no 2-D full-wave check (`validation_2d` is null; the code runs it for rib160 and the
  strip only).

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

## Not done (state at the end of the second resumed run, 2026-10-09)

The resumed runs were not allowed to start solver jobs. `MG` below stands for
`.venv/bin/python tools/memguard.py --heavy --` (one heavy job at a time, 6 GB default cap; do not add `--limit-gb 3`,
an explicit limit overrides the default).

| part of the original task | what exists instead | command that would produce it | runtime fallback meanwhile |
|---|---|---|---|
| full bend study: strip rows at R = 3 and 5 um with Tidy3D, Bessel at R = 6.5 um and h = 5 nm, PML-Hankel at R = 3 and 5 um, stored FEM predictions of the Meep resonance | `modes_study_bend.json` assembled from stored results (above) | write `meep_cylindrical.rows` of `modes_study_bend.json` to a file F, then `MRM_MEEP_CYL_JSON=F MG .venv/bin/python -m mrm.extraction.modes --study only` (FEM + Tidy3D; Meep rows reused) | none needed: the runtime does not read the study; every `modes_<xsec>.json` already uses the exact transform |
| more Meep resolution (e.g. 160 px/um) to tighten the first-order extrapolation | 9 stored runs, 30-120 px/um; estimate +-0.23 nm (narrow), +-2.1 nm with the order spread | `MG .venv/bin/python -m mrm.extraction.modes --meep-cyl-only F2` (edit `resolutions` in `meep_cyl_runs` first); not run, per Kenny | Meep is a check only; nothing at runtime |
| 2-D full-wave (Meep) check of the rib200 couplers | rib160 check (same slab-gap topology: A/Meep 0.994 and 1.024) and method B in 3-D for rib200 | add `rib200` to the two name filters in `coupling.py` (`extract_coupling`, `--meep2d-only`), then `MG .venv/bin/python -m mrm.extraction.coupling --xsec rib200 --meep2d-only F3` and `MG .venv/bin/python -m mrm.extraction.coupling --xsec rib200 --meep2d-json F3` | the presets use `coupling_rib160` in every world (the 200 nm world keeps the 160 nm couplers) |
| a third interface-aligned overlap-grid level (2.5 nm) for K^2 | 10 / 5 nm aligned + FEM mesh 20 / 15 / 10 nm | `MG .venv/bin/python -m mrm.extraction.coupling --xsec rib160 --no-meep --dx-nm 2.5 --out <tmp>` (rows 10 / 7.5 / 5 / 2.5 nm; estimated peak ~4 GB, needs the 6 GB cap) | committed K^2 (5 nm grid; 10 -> 5 nm changes it by 0.15 %) |
| mesh-refinement series for alpha_rad; step-size study of the Jacobians | PML/window spread (0.09 %, 0.03 %); one step per Jacobian; log-derivatives in this handoff | alpha_rad: run the extraction at three meshes, `MG .venv/bin/python -m mrm.extraction.modes --study no --xsec rib160 --mesh-scale S --conv-scales ... --out <tmp_S>` for S = 2.0, 1.5, 1.0 (one process per cross-section). Jacobian steps: edit `Settings.jac_steps_nm` (no CLI flag) | secant Jacobian applied as a linear offset by `realize` |
| first-design meshes (production h_core 10 nm, convergence down to 7.07 nm), cut under the former 3 GB cap | 15 nm production, 30 / 20 / 15 / 10 nm series; bias of the reported values in `validation.production_mesh_bias` (see "Memory cap") | `MG .venv/bin/python -m mrm.extraction.modes --study no --xsec rib160 --mesh-scale 1.0 --conv-scales 2.0 1.4142 1.0 0.7071 --out <tmp>` (one process per cross-section; estimated peak 3.1 GB rib160, 3.8 GB rib200), then the couplers with `--mesh-scale 1.0`. Replacing the committed modes files changes their sha256, which `junction_rib160.json` and the coupling sidecars record: re-extract those too | committed 15 nm values (lambda_res bias 0.025 nm, n_g bias 3.6e-5 on rib160) |
| Tidy3D cross-check of TM0 / TE1 and of the de Cea geometry | FEM only (`mode_families`; `modes_decea2019.json` crosscheck is null) | needs a family selector in `tidy3d_block` (not written) | TM0 / TE1 are not used at runtime (07 P2-3 optional); VAL-7 uses the FEM |
| sidewall-angle sensitivity | nominal 90 deg; `XSec.sidewall_deg` works in the FEM, not in the Tidy3D wrapper | add `sidewall_deg` to `jac_steps_nm` and re-extract (no CLI flag) | none; `realize` has no sidewall dimension |
| n_eff(lambda, T) cross term (dlambda/dT per line) | n_eff(lambda) at 25 C and dn_eff/dT(T) at 1310 nm | not written | s(T) at 1310 nm for every line |
| coupler K^2 versus temperature | K^2 at 25 C | not written | K^2 at 25 C (07 P1-5 excludes a ring-bus mismatch, < 1 %) |
| VAL-7 replication of de Cea 2019 at the printed thickness | done, but it fails (see VAL-7) | - | `replicas.py` reads `modes_decea2019.json` |
| the full extraction at R = 7.5 um (TE0 over 1260-1360 nm and 0-125 C, Jacobians, PML spread, Tidy3D) | `alt_radius`: single values at 1310 nm and 25 C (n_eff, n_g, dn_eff/dT, dlambda/dT, FSR, alpha_rad) | not written: `extract_xsec` has no radius override (r_ref is always the 1.8 THz radius); a `--radius-um` flag that skips `find_r_ref` would produce it | R3 derives the radius from the FSR and n_g; at the `pdk_size` FSR (R ~ 7.5 um) the runtime keeps the r_ref values: n_g differs by +1.0e-3 (rib160), dlambda/dT by +0.02 pm/K, alpha_rad 0.022 vs 0.119 dB/cm (negligible against the ~110 dB/cm budget) |
| a range and a sensitivity for the sidewall angle and the cladding stack | labelled in `inputs.geometry.ranges_not_assigned` of every sidecar (review) | sidewall: as in the sidewall row above; cladding: not written | nominal vertical walls and an all-oxide cladding; the runtime uncertainty omits both |

## Memory cap (former 3 GB per job; now 6 GB for the single heavy job)

Question: did the old 3 GB cap make this task's results less accurate? Evidence: the first run's job logs (memguard
"killed" and "peak RSS" lines, session scratchpad, not in the repo), its transcript, the code and the sidecars.
`git log -p` shows no history for these files (the code was committed once, in its final form). The mesh counts below
come from gmsh meshes generated without solving (peak 0.14 GB). Each sidecar now holds the result in `mesh.memory`.

| # | what was reduced | evidence | accuracy cost (from stored convergence numbers) | does 6 GB allow the intended setting? |
|---|---|---|---|---|
| 1 | Modes: production mesh h_core 10 -> 15 nm, and convergence levels 20 / 14.1 / 10 / 7.07 -> 30 / 20 / 15 / 10 nm, for all four modes artefacts. The couplers use the same 15 nm mesh. | Changed before the first full run, right after a 5-solve probe at 10 nm (rib160, windows up to 128k dofs) peaked at 2.61 GB, 87 % of the 3 GB cap. The transcript does not state the reason. Solve time (about 3x per solve) was a second factor. The 7.07 nm level has about 2x the elements of the 10 nm level. | The reported values are the 15 nm values. Their bias is 3-4x the recorded `discretization_error` (the table under "Convergence"): rib160 n_eff +7.6e-5 (lambda_res +0.025 nm), n_g -3.6e-5 (r_ref +9e-6 relative), dn_eff/dT +5.2e-9 (+0.002 pm/K). The strip's n_g series is not asymptotic (p 0.46), and a 7.07 nm level would have tested it. The couplers' K^2 at 15 nm is +0.06 % (rib160), +0.08 % (rib200) and +0.20 % (strip) from the 10 nm value, against a 1-3 % CMT method error. None of this matters at runtime, because the geometry uncertainty and the calibrated resonance offset are 30-100x larger. | Yes. Estimated peaks of the first design: rib160 3.1 GB, rib200 3.8 GB. Basis: the largest pair of live problems at 7.07 nm (23666 + 21038 and 25168 + 22658 triangles, 314k and 336k dofs); an RSS model of 0.18 GB + 6.25 kB per live dof, fitted to the probes; plus the overhead measured in the production runs (1.0 and 1.5 GB). Uncertainty is about 30 %, so the worst case is about 4.9 GB. That exceeds 3 GB and fits in 6 GB. The run time would be about 3-4x (rib160 production took 30 min). The strip (about 2.5 GB) and de Cea (about 2.8 GB, scaled from its 10 nm meshes; not stored in its sidecar) estimates are below 3 GB. |
| 2 | Bend study: the FEM mesh of `meep_check`'s predictions of the cylindrical-Meep resonance, 7.07 -> 10 nm. | The light job ran under a self-chosen 1.5 GB cap and was killed at 1.52 GB. The rerun at 10 nm peaked at 2.37 GB under 2.5 GB. That was before the gc fix, and the Meep subprocess counts toward the RSS. | About 0.005 nm (the strip's n_eff changes by 1.6e-5 from 15 to 10 nm), against the Meep uncertainty (+-0.23 nm narrow, +-2.1 nm with the order spread). The exact-minus-isotropic difference (2.04 nm) is taken on one mesh, so the bias mostly cancels. The verdict is unchanged. | Yes. The 7.07 nm mesh adds 31k dofs (about 0.2 GB) and fits even the 3 GB light default (`--meep-pred-scale 0.7071`). |
| 3 | Process structure only, no accuracy effect. A run that extracted several cross-sections in one process was killed twice (3.03 and 3.04 GB) because memory built up across cross-sections. The fix was one process per cross-section, a 2-problem FEM cache, `gc.collect()` every 4 solves, and freeing the eigen matrices. The coupling `--quick` run hit 2.78 GB under a 1.5 GB light cap in skfem's point locator; matplotlib's trapezoid-map finder replaced it and gives the same results. | `modes_full.log`, `modes_full3.log`; transcript | none: same meshes and solves. The strip artefact was written before the kill and was not affected. | n/a |

Not caused by the cap (checked):
- The Tidy3D grids 10 / 5 / 2.5 nm were the design. The 2.5 nm level adds about 0.8 GB.
- The coupling overlap grids 10 / 5 nm were the design. A 2.5 nm level was never planned. It would need about 4 GB (the sampled-field cache grows from 0.3 to 1.3 GB and the probe matrices by about 0.3 GB), which fits only in 6 GB (`--dx-nm 2.5`).
- The coupling dz was changed from 0.02 to 0.04 um to save run time. Halving dz changes K^2 by 5e-5 relative.
- The Meep runs stayed below 0.85 GB.
- The 160 px/um Meep run was skipped because Kenny asked for that, not because of memory.

Peak RSS of the committed runs (memguard): modes rib160 2.17 GB, rib200 2.74 GB, decea2019 2.00 GB (the strip's peak
is not recorded, but it was below 3.0 GB); coupling rib160 2.03 GB, rib200 2.43 GB, strip 1.90 GB; Meep cylindrical
0.19-0.23 GB; Meep 2-D 0.74-0.85 GB.

**Does 6 GB limit accuracy or any other goal of this task? No.** Every intended setting fits, with the largest estimate
3.8 GB (about 4.9 GB at the +30 % bound). Running the first design again would refine lambda_res by about 0.02 nm, so
the runtime model does not need it. The commands are in "Not done" (row "first-design meshes"). Run each cross-section
as its own heavy job. Do not pass `--limit-gb 3`, because an explicit limit overrides the 6 GB default. The coupler
re-run with `--mesh-scale 1.0` was not estimated separately: its FEM convergence levels become 20 / 15 / 7.07 nm
(the 7.07 nm level is the same isolated-mode problem as in the modes first design, 3.1-3.8 GB estimated) plus the
overlap-grid caches (~0.3 GB at 5 nm); combining it with `--dx-nm 2.5` (~4 GB on its own) could approach the 6 GB
cap, so run those two refinements as separate jobs. The modes CLI now refuses a `--mesh-scale` that is not one of the
convergence scales before any solve (previously it failed in `production_mesh_bias` after the convergence solves).

## Sidecar fields added in the second resumed run (data files unchanged, no solver run)

- `validation.production_mesh_bias` (modes): f(production mesh) - f0 for every convergence quantity, the
  lambda_res-equivalent bias and the r_ref relative bias. `modes.production_mesh_bias` computes it from the stored
  series, and future extractions write it too.
- `mesh.memory` (modes, coupling) and `mesh.meep_fem_prediction` (study): the cap that applied, the peak RSS, what was
  reduced and why, the mesh counts, the 6 GB estimate, and a re-run command.
- `inputs.geometry.provenance` = "assumed" (the per-field dict moved to `provenance_by_field`) and
  `inputs.materials.provenance` = "literature", the junction sidecar convention. `mrm.params.config.resolve_from` now
  labels values from these artefacts `solver(assumed-geometry, literature-materials)`, which ranks as assumed. Before
  this change it labelled them bare `solver`, which ranked too strong. The presets set explicit provenance on every
  leaf they load from these files, so the config hash does not change. `common.geometry_input` and
  `common.materials_input` write the same fields in future runs.
- `assembly.full_study_command` of the study no longer pins `--limit-gb 3`.

## Review corrections (no solver run; data files unchanged)

- Tidy3D cross-check: the deltas (-1.6e-4 / +2.5e-4 / -1.9e-4) were reported as if they were the agreement; Tidy3D's
  own extrapolation uncertainty is larger (see "Cross-checks"). New sidecar field
  `validation.crosscheck_tidy3d_uncertainty` (`modes.tidy3d_crosscheck_uncertainty`, pure arithmetic, written by future
  extractions too; null for de Cea, which has no Tidy3D run).
- Cylindrical Meep: the +-0.23 nm left out the order uncertainty (+-2.1 nm with it); the "9x" statement is withdrawn
  (see the bend study). New sidecar field `validation.meep_order_sensitivity` (`modes.meep_order_sensitivity`), written
  by both study paths.
- Assumed values without a range (the sidewall angle of every cross-section, the cladding stack, de Cea's 145 nm BOX)
  are now listed in `inputs.geometry.ranges_not_assigned` (`common.unranged_assumptions`, also in future sidecars).
- `inputs.r_ref_rule` (modes): r_ref is the 1.8 THz radius, except de Cea (fixed 9.15 um centreline; its sidecar's
  `fsr_design_thz` is not used).
- n_g geometry spread: the label now says which half-widths were used (the strip's width range is +-20 nm, and the
  rib slab ranges in `ranges_nm` are wider than +-20 nm: +-0.16 / +-0.05 with them).
- `test_memory_record` skips artefacts without a `mesh.memory` record (the extraction code does not write it, so any
  re-extraction would have failed the test) and accepts a 3 or 6 GB cap.
- New fast tests: `test_tidy3d_crosscheck_uncertainty_recorded`, `test_bend_study_meep_order_sensitivity`,
  `test_unranged_assumptions_labelled`, `test_cli_rejects_production_scale_outside_convergence_series`.

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
7. Memory cap: the reported mode values come from the 15 nm mesh that the former 3 GB cap shaped. Their bias is
   3-4x the recorded `discretization_error` (rib160: lambda_res +0.025 nm, n_g -3.6e-5), recorded in
   `validation.production_mesh_bias`. The first design (10 nm, series to 7.07 nm) fits the 6 GB cap (estimated
   3.1-3.8 GB) but was not re-run (no solver runs allowed). See "Memory cap".
8. Cross-check strength (review): the Tidy3D n_eff cross-check is limited by Tidy3D's first-order convergence (a few
   1e-4 by the order sensitivity, < 4.3e-3 conservatively), and the cylindrical Meep check by its order spread
   (+-2.1 nm against the 2.04 nm exact-isotropic difference). The bend-formulation verdict stands on the Bessel
   benchmark and the Tidy3D bend shifts. A finer Tidy3D grid (1.25 nm) or Meep 160 px/um would sharpen both.
9. The sidewall angle, the cladding stack and de Cea's BOX are assumed without a range
   (`inputs.geometry.ranges_not_assigned`); the runtime uncertainty does not include them.

## Handoff requests

- **params / presets (07 P0-3, P1-6):** H_solver s0 at 25 C = 65.08 pm/K (rib160; 66.91 rib200), beta = 1.47e-3 /K
  around 25 C (1.02e-3 around 105 C), from `extracted/modes_<xsec>.json:dlambda_dt_pm_per_k` (table node, `at:
  {t_c: ...}`). `ring.geometry.radius_um` = r_ref_um; `ring.optics.n_eff` = n_eff, `n_g`, `dn_g_dlambda_per_nm`;
  `ring.geometry.jacobian` = `jacobian`; `ring.geometry.jacobian_coupling` = coupling `jacobian`.
- **loss budget (07 P0-2):** `alpha_bend_radiation_db_per_cm` = `alpha_rad_db_per_cm` (0.12 dB/cm rib160, 1.0 rib200).
  For slab excursions use `slab_sweep` (log-interpolate), not the linear Jacobian (a secant, see the Jacobian note).
- **realize (07 P1-16):** `alpha_rad_db_per_cm_per_nm` is applied as a linear offset, which can make the bend
  radiation negative (rib160: -0.085 dB/cm at a +8.4 nm film draw). Either clamp `alpha_bend_radiation_db_per_cm` at
  0 after the Jacobian, or add a log-scale target (e.g. `ln_alpha_rad_per_nm`, "lnscale", +1) and use the
  d ln alpha_rad/dp values of the Jacobian note (rib160: -0.0485, -0.1785, +0.4104 per nm of width, film, slab). I did
  not add them to the artefacts: an unknown `of` name would make `realize` raise, and rewriting `modes_rib160.json`
  would break the modes sha256 that `junction_rib160.json` and the coupling sidecars record.
- **params_content (presets):** `modes_rib200.json` and `coupling_rib200.json` exist. The `uncertainty.film_thickness`
  "200" choice still uses the strip400x200 artefacts as a stand-in "until modes_rib200 exists", and the note of
  `strip400x200_q_anchor` asks for a recompute when the rib n_g lands (rib160 n_g 3.9405, rib200 4.0116). The
  `ring.geometry.jacobian` fallback note ("until modes_rib160 exists") is stale. `ring.geometry.jacobian_coupling` is
  not wired to `coupling_<xsec>.json:jacobian` yet. Five fast tests of `tests/closure/test_presets.py` fail because
  the loss budget is still balanced for the fallbacks (n_g 4.0, bend radiation 1.0 dB/cm) while the artefacts now
  resolve (n_g 3.9405, 0.119 dB/cm): realize of `g25_test_vehicle` gives a 26.46 dB notch (test: 27.9 +- 0.5);
  overriding only n_g = 4.0 gives 29.86 dB, only the radiation = 1.0 gives 25.08 dB, both give 27.91 dB. Rebalance the
  non-heating loss term so that Q_i (not alpha_i) is kept, as the `strip400x200_q_anchor` note describes.
- **extract_junction:** `test_tidy3d_cross_check[rib160]` fails (delta n_eff -0.0177). The cause is the extrapolation,
  not the solvers: the junction's Tidy3D runs use uniform grids without subpixel averaging and a forced first-order
  Richardson over 20 / 10 / 6.7 nm (2.4596 / 2.4462 / 2.4372 -> 2.4192). The finest value 2.4372 agrees with the
  junction's femwell 2.4368, and the modes artefact's independent Tidy3D run on interface-aligned 10 / 5 / 2.5 nm grids
  extrapolates the same rib160 straight guide to 2.43706 against the FEM 2.43690 (delta 1.6e-4). Fitting the order
  (as `mrm.extraction.common.richardson` does) or aligning the grid with the interfaces would remove the failure.
- **coupling (07 P1-5, P1-9):** `dln_kappa2_through_dlambda_per_nm` = 0.0077 /nm (rib160), drop 0.0123 /nm;
  `dln_kappa2_dgap` -0.0135 /nm (so +-5 nm of drop gap is +-7 % of monitor gain, not +-10 %).
- **plant_optics:** `coupler_phase_rad` follows your psi convention (delay, adds to phi).
- **realize:** the coupling Jacobian is partial at fixed gap; a correlated width bias needs d/dw - d/dgap. When
  `ln_kappa2_*_per_nm` is wired to Q_t/Q_d with `lnscale, -1`, note that d ln Q_e / d ln K^2 = -K^2 / ((1 - K^2)
  (-ln(1 - K^2))) = -1.057 at the through K^2 0.1045 (-1.003 for the drop), so the -1 mapping understates the
  through coupler's geometry sensitivity of Q_t by 5.7 %.
- **plant_optics / params (phase.py):** the runtime s(T) = s0 (1 + beta (T - T0)) with beta at 25 C overestimates
  the solver table by 0.5 / 1.25 / 1.9 % at 75 / 105 / 125 C (the integrated shift from 25 C by 0.18 / 0.44 /
  0.68 %), because s(T) is concave (beta 1.47e-3 at 25 C, 1.02e-3 at 105 C). This is below the Si dn/dT sigma
  (3.2 pm/K at 105 C); a quadratic term or the `dlambda_dt_pm_per_k` table would remove it.
- **integrator (ARCHITECTURE 5.1):** `index_weights` holds `si`, `box`, `clad` (not `sio2`); amend the example or
  note it in docs/CHANGELOG.md (convention 4 above).
- **validation (VAL-7):** use the table above; VAL-7 as frozen (n_g 2.73 +-3 %, Gamma 0.65 +-0.05) fails at 100 nm.
- **docs (07 D10, 05 section 5.4):** the femwell-vs-Tidy3D bend disagreement is settled in favour of the exact
  transform (decisive: the analytic Bessel benchmark and the Tidy3D +-R mean bend shifts; supporting: cylindrical
  Meep, which converges at first order and, after Richardson extrapolation, is consistent with the exact transform
  but resolves the 2 nm difference only marginally once the order uncertainty is included; `modes_study_bend.json`).
- **params (docs/PARAMETERS.md):** `tests/verification/test_params_cli.py::test_parameters_md_is_current` fails.
  `docs/PARAMETERS.md` was last generated before the modes and coupling artefacts existed, so it still shows the
  fallbacks (n_g 4, n_eff 2.4, bend radiation 1 dB/cm). Regenerate it with `mrm params table g26 --out
  docs/PARAMETERS.md` after the loss-budget rebalance above. The sidecar provenance change of this run does not move
  the config hash, because the presets set provenance explicitly.
- **tools/memguard.py (owner unknown):** the heavy lock has no queue; with 5-6 waiters one agent's sequential jobs
  can starve others for an hour. A FIFO ticket would help.
