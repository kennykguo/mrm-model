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

## Numbers

(filled in below after the production run)
