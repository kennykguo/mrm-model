# Handoff: materials (`mrm/materials.py`, `mrm/data/materials.json`)

Status: done. `tests/verification/test_materials.py` has 42 tests. They run in about 3 s and need no solver.
Run `python -m mrm.materials` to print the materials table for the docs.

## For every agent that uses material data

- Import from `mrm.materials`. Do not copy numbers out of `docs/research/02_materials.*`. All coefficients live in
  `mrm/data/materials.json` (schema `materials/1`). That file is a cleaned copy of the 02 data with the
  verification-log corrections applied.
- Units. Wavelength `lam_m` is the vacuum wavelength in m. Temperature `t_k` is in K. Doping and carrier
  densities `*_per_cm3` are in cm⁻³. A unit suffix in a function name gives the unit of its result.
  - If you pass a wavelength in nm or µm, you get a `ValueError`.
  - If you evaluate a model outside its source range, you get a `MaterialRangeWarning`.
- Record the materials model in your sidecar: put `mrm.materials.materials_provenance()` (schema and sha256) under
  `inputs.materials`.

## Requests and notes, by consumer

1. **modes (07 P1-6)**
   - Use `n_si(lam_m, t_k)`, the Li 1980 temperature shape on the Frey/Franta absolute level, together with
     `n_sio2(lam_m, t_k, kind="thermal"|"pecvd")`. Both are vacuum-referenced.
   - Take n_g,i from the same models: `n_g_si`, `n_g_sio2`. At 1310 nm and 300 K they give 3.690 and 1.4621.
     Use these for the identity Σ n_g,i ∂n_eff/∂n_i = n_g.
   - `level="li"` is the alternative index hypothesis. It is 4.085e-3 lower at 1310 nm.
   - For the uncertainty of dλ/dT, use `dn_dt_si_sigma_per_k(t_k)` and `material_uncertainties()`.
   - The materials bound on dλ/dT is `dlambda_dt_ceiling_pm_per_k`: 70.8 pm/K at 300 K with the one-model n_g, and
     71.1 pm/K on the Li level.
   - Expansion: `alpha_l_si_per_k`, which the substrate clamps. The exact length change is `expansion_strain_si`.
2. **thermal_fem (07 P1-3)**
   - The text in 07 P1-3, "thin-film k ≈ 56 to 66 W/(m·K), ±25 %", is **superseded** by the 02 correction (log #14).
     Use `k_si_film_w_per_m_k(t_k, film_nm)`:
     - 160 nm: 71.8 W/(m·K) at 300 K and 64.2 at 375 K;
     - 200 nm: 80.0 and 70.1.
     The uncertainty is ±15 %. The valid range is 150 to 300 nm.
   - `thermal_properties(material, t_k, film_nm)` covers every stack material (`THERMAL_MATERIALS`).
   - The metal/dielectric boundary resistance is 2e-7 m²·K/W, equal to about 0.25 µm of oxide. It is not negligible
     for `metal_above`.
   - `rho_c` of low-k is an assumption with no source: 1.3 MJ/(m³·K), range 0.9 to 1.8.
3. **junction**
   - `plasma_dispersion(ne, nh, model="nsm2011"|"soref1987")` takes absolute carrier densities.
   - For depletion, use `plasma_dispersion_change(ne0, nh0, ne1, nh1)`, that is, f(N1) − f(N0). Do not use a signed
     power of ΔN.
   - α is returned in 1/m, as an intensity absorption.
   - `vbi_v`, `dvbi_dt_v_per_k` and `ni_si_per_cm3` use the Misiakos n_i law. devsim or tidy3d charge models have
     their own n_i and E_g, for example tidy3d `Si_MultiPhysics` uses a constant E_g = 1.11 eV. Record which one the
     LUT used.
4. **plant/electrical and params**
   - `heater_tcr_per_k(family, corner)` accepts a ROM family name or a material name:
     - `inring_noundercut` → doped Si, 1.4e-3 (range 1.0 to 1.8e-3);
     - `silicide_undercut` → silicide, 3e-3 (range 2 to 4e-3, assumed);
     - `metal_above` → TiN, 0.5e-3. The family may also be W (2 to 4.6e-3).
   - The textbook's G25 fit of 1e-3 /K is the low end of doped Si.
5. **plant/sensing (07 P1-9, D20)**
   - Use `ge_pd_dark_current_a(i0_a, t_k, ea_ev=...)`. E_a is 0.37 eV (range 0.35 to 0.38). The stress value is
     0.5 eV, from `ge_pd_dark_ea_ev("stress")`.
   - I_d0 is 1 to 30 nA at 25 °C (`GE_PD_DARK_I0_A_RANGE`).
   - The responsivity limit at 1310 nm is 1.0566 A/W. The responsivity is constant in T, with an assumed ±3 %.
6. **validation (P2-1 bound tests)**: use `tpa_beta_m_per_w(corner)`, which is 0.9 cm/GW (range 0.5 to 2.0), and
   `kerr_n2_m2_per_w(corner)`, which is 4e-14 cm²/W (range 3 to 6e-14).
7. **params/table.py**: `material_table()` and `material_table_markdown()` give the materials section of
   PARAMETERS.md.

## For the integrator (files I do not own)

- ARCHITECTURE §2 says "Coefficients from docs/research/02_materials.json". They are now read from
  `mrm/data/materials.json`, the cleaned copy that this task requested. The test fails if `02_materials.json`
  changes without a re-sync. Please update the line and add a `docs/CHANGELOG.md` note.
- 07 P1-3 thin-film k (56 to 66 W/(m·K), ±25 %): superseded, see item 2.
