# Handoff: extract_thermal (`mrm/extraction/thermal_fem.py`, `mrm/extraction/mor.py`)

Status (2026-10-09, resumed run): finished from the committed results. This pass ran no solver. It checked the
artefacts against ARCHITECTURE 5.3 and the runtime reader, completed the sidecars from numbers they already held,
added the VAL-8 sidecar, fixed the fast tests, and recorded what is not done ("Not done") and what the former 3 GB
memory cap reduced ("Memory cap").

- All six `thermal_rom_<family>_<film>` artefacts exist; `mrm.plant.thermal.load_rom` accepts every one (schema,
  sha256, site offsets, the four 07 P1-3 numbers).
- Time and frequency contracts: all six pass (self and nearest-neighbour terms).
- **Superposition (07 P1-3): FAILS.** The 8-ring bank error is 13.9 % against the 10 % target. The split into 3-D
  discretization and multibody effects is pending (the re-run was interrupted).
- VAL-8: 8a, 8b and 8c fail, 8d passes (validation tests, reported, not blocking).
- Fast tests: `tests/verification/test_thermal_rom.py` passes (one strict xfail for the superposition target).
- Review pass (same day, no solver): corrections are marked "[review]" below. In short: the mesh error of the 10 %
  self-heating time was understated (bound 0.3-2.2 %, not <= 0.29 %); every assumed value now carries a range in the
  sidecar (`inputs.assumption_ranges`); the `g25_consistency` diagnostic still used the old dlambda/dT proxy and is
  refreshed; the superposition note now says what is compared; two inferences in the superposition section were not
  supported by the data; the stacked-EIC stack hypothesis of 07 P1-3 was missing from "Not done"; runtime notes for
  plant_thermal and params are added.

## Files

- `mrm/extraction/thermal_fem.py`: axisymmetric P2 heat FEM (scikit-fem on gmsh meshes), coarse 3-D P1 model (single
  ring and 8-ring bank), exact layered analytic solution (independent check), femwell TE0 weights, extraction
  pipeline, stages, artefact I/O, CLI.
- `mrm/extraction/mor.py`: rational Krylov (Galerkin) ROM, modal Foster form, common real-pole vector fitting,
  variable-projection polish, DC-exact residue fit, error metrics.
- `mrm/data/extracted/thermal_rom_<family>_<film>.npz` + `.json` sidecar (schema `thermal_rom/1`), families
  `inring_noundercut` (default), `silicide_undercut`, `metal_above`, films 160 and 200 nm.
- `mrm/data/extracted/thermal_val8_imec.json` (schema `thermal_val8/1`, 07 VAL-8) + `thermal_val8_imec.sidecar.json`
  (new in this pass: schema, sha256 of the data file, inputs, tool versions, mesh, validation, created_utc).
- `tests/verification/test_thermal_rom.py` (fast: artefact identities, recorded verification, runtime-reader check,
  sidecar consistency; slow: re-run), `tests/validation/test_thermal_replication.py` (VAL-8).

Re-run, each job through `.venv/bin/python tools/memguard.py --heavy -- ...` (6 GB default cap for heavy jobs):
`python -m mrm.extraction.thermal_fem [--family F] [--film N] [--stage extract|uncertainty|analytic|3d|val8|propagate]
[--h3d-levels H...] [--h3d-bank H] [--quick] [--out DIR]`. `--quick` writes to a temporary directory unless `--out`
is given. `--stage propagate` runs no solver: it refreshes `rom_dc_scale` from the modes artefacts, copies the shared
records (superposition, analytic check reference) into every sidecar and rewrites the VAL-8 sidecar.

Changes in this pass (code): `--h3d-levels` / `--h3d-bank` (defaults reproduce the committed 3-D record); a 3-level
3-D study now records a Richardson estimate; the lead effect is computed at equal mesh when possible;
`superposition_record()` writes the correct note (the old note said "L1 residue error" for the step metric);
`write_val8_sidecar()`; `rom_dc_scale` records the sha256 of the modes artefact it used.

[review] Code changes of the review pass (none changes an artefact; the `.npz` files and their sha256 are unchanged):
- `write_artefact` keeps every `validation.superposition_*` key on a re-extraction (it dropped `superposition_target`
  and `superposition_pass`, which would have broken `test_superposition_recorded_consistently` after any re-run).
- `conservative_mesh_bound()` (recorded as `mesh.convergence.conservative_bound`): `richardson()` clamps the observed
  order to >= 1, which understates the error when the changes shrink slowly; the bound sums the remaining changes as
  a geometric series (or takes the largest of the last two changes for noise-level or oscillating changes).
- `assumption_ranges()` (recorded as `inputs.assumption_ranges`): every geometry, boundary and site value with its
  provenance, its range and where its sensitivity is recorded (Jacobian 1-sigma, sweep, radial table, or "not swept"
  with a plausible range that no run has explored).
- `derive_sidecar_fields()` in `--stage propagate`: the two records above, `rom.taus_note` (`rom.taus_vf_s` are the
  vector-fitting poles before the polish; the final poles are `taus_s` in the `.npz`), the `g25_consistency` numbers
  re-evaluated with the current dlambda/dT hypotheses (they still used the 67.5 pm/K proxy for the 160 nm films and
  the strip-400x200 value, 65.8 pm/K, for the 200 nm films), and `stages.crosscheck_analytic.disagreements_above_1pct`.
- `stage_three_d` (for the pending re-runs): the bank-solve cache key now hashes the config, mode weight, materials
  file and mesh (it was keyed by ring count, pitch and h only, so a stale bank could be reused); the run keeps the
  `memory_cap` record (and `derived_equal_mesh` when the meshes are unchanged), which `update_sidecar` would otherwise
  drop; `distance_interpolation` also records the linear-in-d scheme the runtime uses.
- `superposition_record()` note: states that the comparison model is the axisymmetric FEM at the exact bank distances
  (comparison variant), so the Foster-fit and distance-interpolation errors come on top.
- `write_val8_sidecar()`: records VAL-8e `rel_err_tau` and states that its `materials` block is read when the sidecar
  is written (the data file does not store the materials sha256; `mrm/data/materials.json` is unchanged since commit
  0a9941e, before the VAL-8 run, so the block is correct now).

## Artefact format (for `mrm/plant/thermal.py`; follows its handoff section 4)

`.npz` (no pickle): `taus_s` (K,) strictly increasing; `sources` = (heater, waveguide_absorption, junction, driver,
aggressor); `observers` = (ring_mode_weighted, heater_resistor, monitor_pd, temp_sensor); `self_res` (5, 4, K) K/W;
`cross_r_um` (D,) increasing, all > 2R; `cross_res` (5, 4, D, K) K/W; `site_offset_names` (n,) and
`site_offsets_um` (n, 2) for monitor_pd, temp_sensor, driver, aggressor; extras `dc_self_k_per_w` (5, 4),
`dc_cross_k_per_w` (5, 4, D) (exact FEM DC; the residues sum to them to 1e-12), `family`, `film_nm`, `schema`.

Sidecar `<stem>.json` (read by `mrm.plant.thermal.load_rom` and `mrm.params.config.sidecar_candidates`; the plant
tests rely on this name): `schema`, `sha256`, `created_utc`, `family`, `film_nm`, `inputs` (every geometry
assumption with its provenance, the material values used, `materials_provenance()`, the femwell mode info),
`tool_versions`, `mesh` (model sizes, the convergence study, `memory_cap`), `rom` (fit details), `validation` (full-FEM
errors; flat keys `step_max_rel_err`, `freq_max_db_err`, `freq_max_deg_err`, `superposition_rel_err`,
`superposition_target`, `superposition_pass`), `summary` (key numbers), `rom_dc_scale` (07 P1-2 diagnostic table),
`reciprocity`, `site_offsets_um` (dict), `k_temp_scaling` (07 P2-5, see below), `stages.uncertainty` (Jacobians,
sweeps, G25 consistency), `stages.crosscheck_analytic`, `stages.crosscheck_3d` (default sidecar: the full record plus
`derived_equal_mesh`, `memory_cap`, `pending`; other sidecars: a reference).

Conventions:
- Step response of a pair: sum_k r_k (1 - exp(-t/tau_k)); sources normalized to 1 W; observers are weighted
  averages (ring: n dn/dT |E|^2 of the femwell TE0 mode, bent at R; heater: volume average; PD and sensor: disks at
  the film mid-plane).
- Cross distance: source centre (ring centre, or site centre for driver/aggressor) to observer centre (ring centre for
  ring_mode_weighted/heater_resistor, site centre for monitor_pd/temp_sensor). Radially symmetric approximation.
- `k_temp_scaling` (top-level sidecar key): `{t_k: [300, 325, 350, 375, 400], dc_ratio: {pair: [...]},
  tau_ratio: {pair: [...]}}` with pairs `heater->ring_mode_weighted`, `heater->heater_resistor`,
  `waveguide_absorption->ring_mode_weighted`, `heater->ring_mode_weighted@pitch`. Quasi-static use (07 P2-5): residues
  times dc_ratio(T), taus times tau_ratio(T), T the mean path temperature.

## Results (all six artefacts; checked against the sidecars in this pass)

Axisymmetric P2 FEM, production mesh 50 nm in the core (scale 1.0, ~46-49k DOF per model); convergence over four levels
(100, 71, 50, 35 nm); Foster fit against the Krylov ROM (142-147 + 103 states, error against exact solves < 4e-6 of DC);
verification against the FULL FEM (exact complex solves at 16 frequencies, TR-BDF2 time stepping 0.25 ns .. 20 ms
with a step-halving error estimate ~3e-5 of DC). Numbers are DC gains in K/mW and 63 % times; ± is the 1-sigma
geometry spread from the Jacobian propagation (assumed parameter sigmas, see `stages.uncertainty.jacobian`).

| artefact | Z heater->ring | Z heater->heater | Z opt->ring | z ratio | t63 heater->ring us | t63 heater self us | opt->ring t10 / t63 | xt at 50 um (DC / t63) | K | self: dB, deg, time | nearest neighbour | mesh err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inring_noundercut 160 | 2.85 (±23 %) | 7.53 | 5.02 | 1.76 | 4.97 (±30 %) | 1.48 | 10 ns / 1.76 µs | 1.02 % / 0.12 ms | 22 | 0.14 dB, 1.6°, 0.24 % | 0.05 dB, 0.2°, 0.02 % | 0.25 % |
| inring_noundercut 200 | 2.86 (±22 %) | 7.44 | 5.01 | 1.75 | 4.98 (±30 %) | 1.53 | 11 ns / 1.78 µs | 1.02 % / 0.12 ms | 20 | 0.20 dB, 1.3°, 0.19 % | 0.16 dB, 0.8°, 0.10 % | 0.29 % |
| silicide_undercut 160 | 8.12 (±29 %) | 8.47 | 10.69 | 1.32 | 21.5 (±48 %) | 18.8 | 50 ns / 15.8 µs | 0.69 % / 0.11 ms | 20 | 0.21 dB, 2.5°, 0.16 % | 0.11 dB, 0.7°, 0.10 % | 0.12 % |
| silicide_undercut 200 | 8.07 (±29 %) | 8.40 | 10.61 | 1.32 | 21.5 (±48 %) | 18.8 | 55 ns / 15.7 µs | 0.70 % / 0.11 ms | 24 | 0.16 dB, 0.7°, 0.08 % | 0.05 dB, 0.4°, 0.03 % | 0.23 % |
| metal_above 160 | 3.57 (±16 %) | 10.32 | 5.12 | 1.43 | 4.73 (±24 %) | 1.21 | 10 ns / 1.73 µs | 0.96 % / 0.11 ms | 24 | 0.09 dB, 1.4°, 0.31 % | 0.03 dB, 0.4°, 0.02 % | 0.17 % |
| metal_above 200 | 3.56 (±16 %) | 10.39 | 5.12 | 1.44 | 4.82 (±25 %) | 1.20 | 9 ns / 1.72 µs | 0.97 % / 0.11 ms | 20 | 0.23 dB, 2.5°, 1.35 % | 0.13 dB, 0.8°, 0.12 % | 0.08 % |

Contract limits: time < 2 % of steady state (10 ns .. 10 ms), < 0.3 dB and < 3° (DC .. 1 MHz) for the self and
nearest-neighbour terms: all six pass. Cross terms use the band where |H| >= 1e-3 |H(0)| (below that the diffusive
response is under -60 dB and its phase is meaningless); far pairs (1.5-2 mm) reach up to 2.9 dB / 10.4° at their
band edge; every one of the 440 pairs per artefact is < 1.4 % in time. Every pole set spans 0.7-7 ns .. 68 ms.

Physics read-outs:
- The film (160 vs 200 nm) changes the thermal ROM by < 1 %; the heater family changes it by 3x (DC) and 4x (tau).
- Self-heating path (07 P0-1): the waveguide-core step response reaches 10 % at ~10 ns, 27 % at 100 ns and 54 % at
  1 us (no-undercut families); 5 % / 12 % / 25 % at 10 ns / 100 ns / 1 us with the undercut. The heater->ring
  response is 21 % at 1 us and 81 % at 10 us (no undercut), 9 % / 42 % with the undercut. z_ratio = Z_opt/Z_heater =
  1.32-1.76 (>= 1, as expected for an external heater). (Pole-by-pole DC shares are not unique between fits; quote
  step fractions.)
- Crosstalk at 50 um pitch: 0.7-1.0 % of the self DC, first-order-like with a 63 % time of 0.11-0.12 ms (the d^2 law
  of 04 gives 0.1-0.3 ms at 15-30 um; consistent). The long-range floor is the package: P x R_pkg = 0.002 K/mW
  (R_pkg 2 K/W for the assumed TEC-held package), i.e. 0.07 % of Z_heater->ring per ring; imec's 1.3-2.6 % floor was
  measured with natural convection and an EIC on top (much larger R_pkg). `sub_um` 100 um raises the 50 um crosstalk
  by 43 %; Dirichlet instead of the package Robin lowers it by 9 %.
- Boundary and substrate choices move Z_heater->ring by <= 0.5 % and t63 by <= 1.1 % [review: was "< 1 %"; `sub_um`
  100 um gives t63 +1.05 %]. Removing the BEOL metal fill raises Z_heater->ring by 6 % (in-ring), 7 % (metal above)
  and 38 % (undercut, where the BEOL carries the lateral heat), and t63 by 14 %, 18 % and 97 % [review: t63 added].
  The undercut cavity's air k (1e-4 instead of 0.026 W/(m K)) moves Z_heater->ring by 1.4 % and the 10 % self-heating
  time by 3.1 %.
- k(T) (07 P2-5, top-level `k_temp_scaling`): from 300 to 400 K Z_heater->ring falls 9 % (oxide k rises with T),
  t63 rises 19 % (rho c), the crosstalk FRACTION rises 38 % (substrate k falls); the absolute cross DC
  (`dc_ratio["heater->ring_mode_weighted@pitch"]`, the factor the runtime applies to cross residues) rises 26 %
  [review: the 38 % is the fraction, not the residue factor].

## Convergence evidence

- Axisymmetric production models (every number in the table): four levels, 100 / 71 / 50 / 35 nm in the core
  (`mesh.convergence`). Several metrics converge non-monotonically at the 1e-4 level (gmsh re-meshing noise), so the
  error estimate is the larger of the Richardson estimate and the 50 -> 35 nm change. Worst metric per artefact
  0.08-0.29 % (`max_rel_err_est`); Z_heater->ring changes by <= 0.10 % from 50 to 35 nm; the conservative bound
  (largest of the last two changes, any metric) is <= 0.26 %.
  [review] That bound does not hold for metrics whose changes shrink slowly. `richardson()` clamps the observed order
  to >= 1, and the 10 % self-heating time t10(opt->ring) converges with observed order 0.1-0.7 (changes 0.29 %,
  0.09 %, 0.08 % per level for inring 200). Summing the remaining changes as a geometric series
  (`mesh.convergence.conservative_bound`, arithmetic on the stored levels) gives, per artefact, worst metric:
  inring 160 0.34 % (t10), inring 200 2.2 % (t10), metal 160 0.57 % (t10) and 0.53 % (t63 of the crosstalk at the
  pitch, a steady +0.02 % per level drift), metal 200 0.12 %, silicide 160 0.12 %, silicide 200 0.26 %. The DC gains
  and the heater time constants stay below 0.3 % (tested). The t10 error (<= 2.2 %) is far below its geometry
  spread (sigma_ln 0.27-1.2), so no number changes, but "every metric < 0.3 %" is not supported for t10.
- Full FEM reference: time-stepping discretization ~3e-5 of DC (step halving); Krylov ROM vs exact solves < 4e-6.
- VAL-8 replication: three levels (100, 71, 50 nm); last-level change 0.02 % (tau heater), 0.05 % (tau wg), 0.12 %
  (efficiency).
- Analytic check: three structured levels (scale 2, 1, 0.5); FEM mesh change <= 5e-6 at DC for every pair.
- 3-D single ring: TWO levels only (0.5, 0.35 um); Z_heater->ring changes 1.0 % between them. Bank and silicide arcs:
  ONE level. See "Not done" and "Memory cap".
- Jacobians and sweeps (`stages.uncertainty`): one level (100 nm). They are differences at the same level; the
  re-meshing noise (~1e-4 .. 1.6e-3 relative per run, from the 4-level study) is small against the reported
  sensitivities except for parameters with |d ln| x step < ~2e-3 (contact_offset, via_ratio), whose contributions
  to sigma are negligible anyway.

## Independent checks

- **Analytic (exact layered solution, `stages.crosscheck_analytic`).** `LayeredAnalytic` solves the laterally
  infinite multilayer exactly (Hankel transform in r, transmission-line admittances in z) for annular sources; it
  reproduces the half-space disk formula 8P/(3 pi^2 k a) to 0.08 %. Against the axisymmetric FEM on a structured,
  interface-aligned mesh (not gmsh), with the full stack (substrate, BOX, continuous 160 nm Si film, cladding,
  anisotropic BEOL, passivation), Dirichlet and Robin bottoms: self and heater-annulus responses agree to <= 0.14 %
  from DC to 1 MHz (0.43 % at 10 MHz); victim annuli at 20-100 um to 0.26-0.59 % at DC; every difference is below
  8.2e-4 of the self DC. The victim at 300 um differs by +3.4 % at DC (+2.7 .. 2.8 % at 1 kHz; 1.25e-5 of the self DC; FEM
  mesh change 1.4e-6, so the gap is not FEM discretization; cause not resolved). Points where both responses are
  < 1e-5 of the self DC (far victims above 10 kHz) are quadrature noise and carry no information.
  [review] The last sentence does not cover every disagreement: three more points above 1 % have responses
  >= 1e-5 of the self DC: the 50 um victim at 100 kHz (+19 %, response 6.4e-5 of the self DC, 4.6e-3 of its own DC,
  i.e. inside the band where the ROM's nearest-neighbour frequency check runs), the 100 um victim at 10 kHz (+1.3 %)
  and the 20 um victim at 1 MHz (+1.2 %). The FEM mesh change there is 0.024 %, 0.004 % and 0.008 %, so the
  disagreement is not FEM discretization; whether the Hankel quadrature or the FEM is wrong is not resolved. Every
  disagreement above 1 % is <= 1.25e-5 of the self DC in absolute terms (`disagreements_above_1pct`), so no runtime
  number is affected; but above about 10 kHz the independent check does not confirm the FEM cross-term magnitude at
  50 um.
- **3-D formulation (single ring, no leads).** 3-D vs axisymmetric at 0.35 um: Z_heater->ring -2.5 %,
  Z_heater->heater -4.9 %, t63 -0.17 %. With two levels the formulation and discretization parts are not separable;
  extrapolation with order 1..2 leaves a formulation gap of -0.2 .. -1.6 % (Z_hr) and +0.8 .. -2.6 % (Z_hh)
  (`stages.crosscheck_3d.derived_equal_mesh`).
- **Heater leads.** -4.0 % on Z_heater->ring, -4.7 .. -4.8 % on Z_heater->heater at equal mesh (interpolated). The
  recorded `lead_effect_*` (-4.3 %, -5.5 %) compares 0.4 um with leads against 0.35 um without and is superseded.
- **Silicide arcs.** Four 140-degree arcs vs two full annuli (same power): +1.0 % on Z_heater->ring, +7.8 % on
  Z_heater->heater, +0.8 % on t63 (same mesh, 0.35 um).
- **Full FEM vs ROM.** The Krylov ROM matches exact solves to < 4e-6 of DC; the final Foster model matches the
  time-stepped FEM within the contract (table above); the TR-BDF2 discretization error is ~3e-5 of DC.
- **Reciprocity** (07 P1-2): Z(heater -> ring weight) = Z(ring weight -> heater) to 1e-15 (`reciprocity`).

## Superposition (07 P1-3): 13.9 % against a 10 % target (FAIL)

8-ring bank at 50 um pitch, 3-D with heater leads, M1 lines and the bus, Dirichlet bottom under 200 um of substrate,
mesh 0.4 um at the rings (110,832 nodes, 634,996 tets). Metric: max over pairs and over 10 ns .. 10 ms of
|y_bank - y_ax| / y_ax(inf). [review] y_ax is the axisymmetric single-ring FEM of the same comparison variant
(Dirichlet bottom, 200 um substrate) at the exact bank distances (its Krylov ROM), not the production artefact
(Robin, 775 um) and not the distance-interpolated Foster table. The Foster-fit error (< 1.4 % of DC, every pair) and
the distance-interpolation error of the `cross_r_um` grid (0 at the 50-300 um grid points, -2.5 % at 350 um with
log-log interpolation) come on top; the 13.9 % assumes the 3-D/axisymmetric ratio carries over from the comparison
variant to the production boundary. The 3-D bank DC values are lower than the axisymmetric ones: -12.6 .. -13.1 %
(50 um), -13.3 .. -13.8 % (100 um), -10.3 .. -10.6 % (150 um), -7.7 .. -7.8 % (200 um), -5.3 % (250 um), -3.6 %
(300 um), -2.3 % (350 um) [review: the 50 and 200 um ranges were incomplete]. The error changes by <= 0.5 % between
end and interior positions of the driver or the victim at the same distance [review: was "<= 0.4 %, so intervening
rings are not the cause"; every pair at a given distance has the same number of intervening rings (d/pitch - 1), so
this comparison cannot separate their effect from the effect of distance; that inference is withdrawn].

What is known without a new run:
- Driven ring self term in the bank vs the axisymmetric ROM: -6.6 % = 3-D vs axisymmetric at 0.4 um without leads
  (-2.8 %; [review] this is the 3-D discretization error PLUS the formulation gap of -0.2 .. -1.6 %, not separable
  from two levels) x its own leads (-4.0 %) x multibody (+0.03 .. +0.10 %). Neighbours do not change the driven ring.
- The cross terms sample a much coarser 3-D mesh than the self term (1.6 um at the film level between rings, about one
  element across the 2 um BOX, 6-12 um in the substrate), so a discretization share of several percent is plausible.
  Victim leads and keep-outs (physical multibody effects that the radial table ignores) are roughly estimated at
  ~2 % or less (scaled from the -4 % self-term lead effect). Neither share is measured.

The split is pending (see "Not done", item 1). Until then the sidecars carry `superposition_pass: false`, the runtime
loader warns, and `test_superposition_meets_target` is a strict xfail.

## rom_dc_scale (07 P1-2) and the G25 efficiency

rom_dc_scale = (FSR/P_FSR)/(Z_FEM x dlambda/dT), P_FSR = 25.45 mW (G25 median). With FSR 1.8 THz (0.405 nm/mW).
Refreshed in this pass: H_solver now comes from `modes_rib160.json` (65.08 pm/K) and `modes_rib200.json` (66.91 pm/K)
for every family (before: a proxy 67.5 pm/K for two 160 artefacts, and the strip 400x200 artefact for the 200 films).

| family | H86 (85.9 pm/K) | materials ceiling (70.65) | H_solver (65.1 / 66.9) | bench 62.5 |
|---|---|---|---|---|
| inring_noundercut | 1.65-1.66 | 2.00-2.01 | 2.19 (160) / 2.12 (200) | 2.27-2.28 |
| silicide_undercut | 0.58 | 0.71 | 0.77 / 0.75 | 0.80 |
| metal_above | 1.32 | 1.60-1.61 | 1.74 / 1.70 | 1.81-1.82 |

No family reproduces the G25 efficiency on the printed 45SPCLO stack: the no-undercut families are 1.3-2.3x too
weak, the GF-PDK-like undercut family 1.25-1.7x too strong. The `g25_consistency` sweep (heater gap, BEOL fill,
keep-out) reaches at most 4.2 K/mW in-ring and 4.4 K/mW metal-above (G25 needs 4.7 (H86) to 6.5 (bench) K/mW). The
G25 number therefore points to a partial undercut, a heater inside the ring's own doped slab (our `junction` source
location gives Z = 4.2-4.4 K/mW), or a thicker oxide than the printed 2 um BOX (dlnZ/dBOX = +0.36 per um). This is the
calibration lever (CAL-3): `ThermalScales.dc_scale` under "resistance" scaling also scales tau.

[review] Two definitions are in use; do not mix them. This table is the 07 P1-2 diagnostic with a constant dlambda/dT
at 25 C. `mrm.params.realize` computes its own `rom_dc_scale` = mean Z_heater->ring from relation R2 (fsr_integral,
with the beta(T) rise of dlambda/dT over the FSR sweep) / Z_FEM: 1.98 for g26 nominal on inring_noundercut_160
(checked in the review with `fidelity.thermal_model: rom`), against 2.19 here. The `g25_consistency` variants
(`stages.uncertainty.g25_consistency`) used the 67.5 pm/K proxy (160 nm) and the strip-400x200 value 65.8 pm/K
(200 nm); they are now re-evaluated with the hypotheses above (`dlambda_dt_hypotheses` recorded there).

## VAL-8 (imec metal-heater ring; `thermal_val8_imec.json`; frozen targets VAL-8a..e)

Replicated with OUR materials (validation-owner rule) on the printed geometry and boundary conditions (15 um ring,
W heater above, M1/M2 Cu, top 10 W/(m^2 K), isothermal bottom, 16-diameter domain); dimensions imec does not print
are assumed (220 nm Si, 450 nm rib, 70 nm slab, heater 0.6 x 0.1 um at 0.45 um, Cu fill 0.15 outside 2 um). Our own
dlambda/dT for the imec C-band rib (femwell, bent, mrm.materials): 75.4 pm/K (72.9 thermo-optic only; imec used
dn_eff/dT 1.95e-4, i.e. ~74 pm/K).

| target | value | predicted | rel | tolerance | result |
|---|---|---|---|---|---|
| VAL-8a heater tau (63 %) | 5.0 us | 1.13 us | -77 % | 30 % | FAIL |
| VAL-8b waveguide tau | 10.0 us | 3.12 us | -69 % | 30 % | FAIL |
| VAL-8c efficiency, no UCUT | 222 pm/mW | 291 pm/mW | +31 % | 25 % | FAIL (just) |
| VAL-8d efficiency, full UCUT | 843 pm/mW | 676 pm/mW | -20 % | 30 % | pass |
| VAL-8e two-pole fit (cross-check) | 3.18 us (43 %) + 13.0 us (57 %) | 3.5 us (87 %) + 32.7 us (13 %) | | | reported |

Diagnosis (honest): with imec's own Table I materials the times are the same (0.93 / 3.2 us), so the gap is geometry
content imec does not print, not material data; our FEM time scaling is verified by the analytic check. Heater width
dominates the heater-node time: a bounded search over plausible unprinted dimensions (heater width 0.6-3 um, height
0.3-0.8 um, Cu keep-out, BEOL thickness; 24 stacks) spans tau_heater 0.5-4.5 us, tau_wg 1.9-6.4 us and 134-347
pm/mW; the best stack (3 um wide W, 0.8 um up) gives 4.5 us (-10 %), 6.4 us (-36 %) and 273 pm/mW (+23 %). No
plausible stack in the search reaches the 10 us waveguide time: imec's device holds more heat capacity and lateral
conduction near the ring (M1/M2 Cu electrodes inside and around the ring, Fig. 9 of the paper; imec states that the
metals change the efficiency by 18 to 52 %) than a keep-out effective-medium BEOL represents. Conclusion: the FEM
pipeline reproduces the undercut efficiency and the order of the time constants, but on this published stack it
predicts a 2-3x faster ring than imec measured. Treat `time_scale_fast` up to x2-3 as plausible (07 4.8 already
samples 0.5 to 2).

## Not done

Each item: what is missing, the command that produces it (run from the repo root), and the runtime fallback meanwhile.
`G = .venv/bin/python tools/memguard.py --heavy -- .venv/bin/python -m mrm.extraction.thermal_fem`.

1. **Split of the 13.9 % superposition error into 3-D discretization and multibody effects.** The re-run that adds
   `pairs[].dc_3d_single`, `max_discretization_dc_rel_all`, `max_multibody_dc_rel_all` and
   `max_multibody_step_err_frac_all` was interrupted (session restart, 16:28 UTC). Command:
   `G --stage 3d --family inring_noundercut --film 160` then `.venv/bin/python -m mrm.extraction.thermal_fem --stage
   propagate` (about 15-20 min, peak ~2.3-2.6 GB). Fallback: the radial `cross_res` table as is (it is 2-14 % above the
   3-D bank at 50-350 um); `load_rom` warns; `uncertainty.thermal.crosstalk_scale` (g26: loguniform 0.33-1.73)
   covers the gap.
2. **3-D at the intended resolution** (single-ring levels 0.5 / 0.35 / 0.25 um with Richardson, bank at 0.25 um). Cut
   by the former 3 GB cap ("Memory cap"). Command: `G --stage 3d --family inring_noundercut --film 160 --h3d-levels
   0.5 0.35 0.25 --h3d-bank 0.25`, then `--stage propagate` (estimated peak 4.7-5.4 GB; about 1 h). It also does
   item 1 at the finer mesh. A bank convergence study needs a second bank level (for example a separate run with
   `--h3d-bank 0.35`); the stage stores only the last bank record, so copy the first one aside, or extend
   `stage_three_d` to loop over bank levels. Fallback: as item 1.
3. **07 P1-3 superposition target (< 10 %).** Not met (13.9 %). Depends on items 1-2: if the error is mostly
   discretization, the finer run may pass; if it is multibody (victim leads, keep-outs), the radial table needs a
   multibody correction or the target needs a 3-D bank ROM. Fallback: as item 1. Test:
   `test_superposition_meets_target` (strict xfail).
4. **3-D bank at other pitches** (the pitch is unknown: 50 um assumed). The radial table covers 20-2000 um, so the
   runtime handles any pitch; the 3-D superposition check exists at 50 um only. Command: none yet; `stage_three_d`
   takes the pitch from `default_config(...).pitch_um` (add a `pitch_um=` argument there and a stage key per pitch).
   Fallback: radial table + `crosstalk_scale`.
5. **3-D bank for `silicide_undercut` and `metal_above`.** Only the default family has a bank run; the other sidecars
   copy its superposition number. The undercut family moves heat laterally through the BEOL (crosstalk sigma_ln 0.51,
   dominated by the keep-out), so its superposition error can differ. Command: none yet (`stage_three_d` fixes the
   family to `inring_noundercut`; generalize the loop). Fallback: the copied number + `crosstalk_scale`.
6. **Self-heating-time Jacobian, 200 nm films and the undercut family.** sigma_ln of t10(opt->ring) is 0.27 for the
   160 nm no-undercut families but 0.7-1.2 elsewhere, dominated by `slab_nm` (one-sided +20 nm step with a re-solved
   femwell mode; d ln t10 / d slab = +0.04 per nm) and, for silicide 160, by `keepout_um` (physical: t10 is a fraction
   of a DC that the keep-out raises). The `selfheat_tau_scale_2sigma` suggestions (0.09-11x) are therefore not
   trustworthy: the linear propagation fails at sigma_ln ~ 1, and a mode-selection jump at slab 80 nm is not excluded.
   Command: `G --stage uncertainty --family inring_noundercut --film 200` after changing the `slab_nm` step to a
   two-sided difference in `JAC_COMMON`. Fallback: use the 160 nm value (sigma_ln 0.27; 2-sigma factor 0.58-1.71).
   [review] This fallback is for the no-undercut families only. For `silicide_undercut` 160 the t10 sigma_ln of 0.71 is
   dominated by `keepout_um` (physical), with a `slab_nm` share of 0.33 that is suspect in the same way; removing that
   share leaves about 0.6, so use about 0.6 for the undercut family, not 0.27.
7. **VAL-8a, 8b, 8c** fail (see VAL-8). No command fixes them: the missing content is unprinted imec geometry.
   Fallback: `time_scale_fast` interval (07 4.8).
8. **Analytic check at 300 um**: +3.4 % at DC, cause not resolved (see Independent checks). Command: none (the
   comparison would need a finer Hankel quadrature study in `stage_analytic`). Fallback: none needed at runtime
   (the term is 1.25e-5 of the self DC).
9. **Slow tests** were not re-run in this pass (they run solvers): `.venv/bin/python tools/memguard.py --heavy --
   .venv/bin/python -m pytest -m slow tests/verification/test_thermal_rom.py tests/validation/test_thermal_replication.py`.
10. Meep: not part of this task (no Meep study was planned for the thermal extraction).
11. [review] **Stack hypothesis of 07 P1-3** ({side-by-side (default), stacked EIC: M1000-like, imec's 3-D case
    -43 % efficiency and +44 % crosstalk}). Only the side-by-side stack was extracted; there is no stacked-EIC ROM.
    Command: none yet (`ThermalConfig` needs an EIC die with a bump/underfill layer above the passivation and a top
    boundary through the EIC; then `G --stage extract --family F --film N` per artefact with a stack suffix in the
    file name, which ARCHITECTURE 5.3 does not define yet). Fallback: the interim model's `stacked_eic` alternatives
    (g26 `ring.thermal.xt_*_frac`, literature CO24-JOM) under `fidelity.thermal_model: interim`; in ROM mode the
    stack switch has no effect (see the params notes below).
12. [review] **Distance interpolation as the runtime does it.** The sidecar's `distance_interpolation` check is
    log-log; `mrm.plant.thermal` interpolates the residues linearly in d. A leave-one-out test on the stored DC tables
    (doubled spacing, so an upper estimate) gives max errors of 10-15 % (linear in d), 6-13 % (linear in ln d) and
    2-8 % (log-log) between 25 and 2000 um; at the geometric midpoints of the real grid, linear and log-log differ by
    0.4-1.9 % (4.6 % in the 20-25 um gap, where the 20 um value is near-field dominated). `stage_three_d` now also
    records the linear error at the bank distances (next 3-D run). Fallback: none needed at the grid points (50, 100,
    150 um ... are grid points); see the plant_thermal request below.

## Memory cap

The artefacts were produced under a 3 GB per-job cap; `tools/memguard.py` now allows 6 GB for a `--heavy` job. Sources:
the code, `git log -p`, the sidecars, the earlier session log, and a mesh-only node/tet count made in this pass (gmsh,
no assembly, no solve; the counts reproduce the stored 3-D meshes exactly). Memory per 3-D tetrahedron is scaled from
the recorded 2.28 GB peak of the 3-D stage at h 0.4 um (635k tets): 2.8 KB/tet with a 0.5 GB base, or 3.6 KB/tet with
no base (upper bound).

| what the 3 GB cap changed | intended | used | accuracy consequence | 6 GB enough? |
|---|---|---|---|---|
| Axisymmetric production level (all six artefacts) | 35 nm (finest level) | 50 nm; 35 nm kept as the finest convergence level | Negligible: every tracked metric changes by <= 0.16 % from 50 to 35 nm (Z_heater->ring <= 0.10 %), inside the stated mesh error 0.08-0.29 %. The first default extraction at 35 nm was killed at 3.10 GB (09:02 UTC), before the Krylov-basis and SuperLU-ordering memory fixes. | Yes (estimate): 35 nm = 77-83k DOF per model, 1.7x production; production jobs peaked at 1.1-2.3 GB. Gain < 0.2 %: not worth a re-run. |
| 3-D single-ring levels (formulation check) | 0.35, 0.25 um | 0.5, 0.35 um | Two levels only, no Richardson: the -2.5 % (Z_hr) / -4.9 % (Z_hh) 3-D vs axisymmetric gap cannot be split into formulation and discretization (formulation gap -0.2 .. -1.6 % / +0.8 .. -2.6 % by extrapolation). | Yes: 0.25 um = 166,716 tets (no leads), 291,178 tets (leads, bus, victim margin): ~1.0-1.3 GB. |
| 8-ring bank mesh | 0.25 um | 0.4 um | One bank level: the 13.9 % superposition error (FAIL) contains an unmeasured discretization share; no bank convergence. | Yes, with little margin: ~1.51M tets (extrapolated; the same rule predicts the meshed 0.4 um bank to +0.6 %), estimated peak 4.7-5.4 GB. A further level at 0.18 um (~2.8M tets, ~8-10 GB) does not fit. |
| Silicide arcs (3-D) | 0.25 um | 0.35 um | Arcs vs annuli (+1.0 %) is a same-mesh ratio: insensitive. Full 3-D vs axisymmetric (-7.0 %) carries the discretization error. | Yes (single ring). |
| Lead effect | equal meshes (bank h = finest single level) | 0.4 um with leads vs 0.35 um without | Recorded -4.3 % / -5.5 %; -4.0 % / -4.8 % at equal mesh (interpolated). Corrected here. | Yes (any of the above). |

[review] Checked against the earlier session log: the memguard kills at 3.41 GB (quick 3-D run), 3.10 GB (the first
default extraction, after its four convergence levels had finished) and 3.97 GB (3-D stage start) are there, and the
3-D coarsening was decided before any 3-D run ("to stay well under the 3 GB limit for the 8-ring bank"). The 8-ring
count at 0.25 um (1,512,020 tets = 1 ring + 7 x (2 rings - 1 ring)) and the 4.7-5.4 GB range (2.8 KB/tet + 0.5 GB,
or 3.6 KB/tet) reproduce from the stored counts. One more consequence of the cap: the 35 nm level is only a
convergence level, but this is not what limits the slowest-converging metric: the 10 % self-heating time of inring
200 changes by 0.09 % and 0.08 % over the last two refinements, so a 35 nm production level (fits in 6 GB) would
lower its 2.2 % bound by only about 0.1 %. Settling it needs a finer core level (25 nm: 146,225 P2 DOF for the
inring 200 ring model, a gmsh-only count made in the review that reproduces the stored 46,477 / 78,579 DOF at
50 / 35 nm; 3.1x production, so whether a convergence level there fits in 6 GB is not established, because the
direct-LU memory grows faster than the DOF) or a check of how t10 is read at 10 ns (the convergence study's Krylov
ROM uses 2 shifts per decade). The open question is whether the convergence evidence is right, not memory. The "Not done" 1 command with default settings repeats the 0.5 / 0.35 / 0.4 um meshes, so its lead effect is
again at unequal mesh; `stage_three_d` now keeps `derived_equal_mesh` in that case.

Memory kills that did NOT reduce accuracy (algorithmic fixes): a 3.41 GB kill in a quick 3-D run (pyamg operator
complexity on mass-dominated Krylov shifts -> Jacobi-preconditioned CG for those shifts; CG to rtol 1e-11, ROM DC
error <= 1.2e-9) and a 3.97 GB kill at the 3-D stage start (skfem's probe locator -> a memory-bounded exact locator,
reconstruction error 2e-16). The femwell mode solve was moved to a short-lived subprocess (no numerical change).
Not memory-driven: the Jacobians at 100 nm (cost), the analytic-check die size (gmsh speed).

Answer to "does 6 GB limit accuracy?": not for the axisymmetric ROMs (they are converged to < 0.3 % already; [review]
except the 10 % self-heating time, bound up to 2.2 %, which memory does not limit, see above). For the
3-D bank, 6 GB allows the intended 0.25 um mesh and a three-level bank study (0.5 / 0.35 / 0.25 um), but no finer
level. If the 0.25 um run still shows a large discretization share, a targeted mesh (refine the BOX and substrate
between the rings, where the cross-term heat flows, instead of the ring-centred refinement) is a better route than
more memory. Recorded in the sidecars as `mesh.memory_cap` (every artefact) and `stages.crosscheck_3d.memory_cap`
(default artefact, with the mesh counts).

## Requests and notes, by consumer

1. **plant_thermal (`mrm/plant/thermal.py`)**
   - Format follows your section 4 (taus increasing, `site_offset_names`, flat validation keys, `.json` sidecar).
     `superposition_rel_err` is the 8-ring 3-D bank result of the default family, copied into every sidecar with
     `superposition_target` 0.10 and `superposition_pass` false. Your loader warns on it, as it should.
   - `k_temp_scaling` is the top-level sidecar key you asked for (format above).
   - Residues: driving-point pairs (heater->heater_resistor, waveguide_absorption->ring_mode_weighted) are fitted
     non-negative; poles are kept >= 1.5x apart, so max |r_k| <= 8 |H(0)| on self pairs and <= 25 |H(0)| on cross
     pairs up to 300 um (`rom.residue_cancellation`). Collocated self pairs are therefore passive pole by pole; mixed
     signs occur only on non-collocated pairs, as the contract allows.
   - K = 20 to 24 poles (the fit ladder adds poles until the contract passes); every set spans ~1 ns .. 68 ms, with
     poles below 1 us and above 1 ms.
   - The package floor (die-level pole, P x R_pkg) is part of the ROM: a TEC/package base filter in the scenario must
     not double count it (it is 0.07 % of Z_heater->ring per ring at the assumed R_pkg = 2 K/W).
   - [review] `_rom_lags` interpolates `cross_res` linearly in d. The cross terms fall roughly like d^-1 .. d^-2, so
     log-log interpolation of the DC with the shape interpolated in ln d is closer: at the geometric midpoints of the
     grid, linear-in-d is 0.4-1.9 % above log-log (4.6 % in the 20-25 um gap). Request: interpolate in ln d (residues
     linear in ln d keeps mixed signs) or scale the interpolated residues to the log-log DC. "Not done" 12.
   - [review] With `fidelity.thermal_model: rom` on g26 (checked in the review: `realize` and `assemble` accept the
     artefact; Z_opt from the ROM, ratio 1.763), the layout puts the aggressor at (0, 30) um from ring 0, which is
     5 um from the ROM's default monitor-PD site (0, 25): that pair is clamped to the 20 um grid end (a warning). The
     FEM aggressor is a 10 um radius disk, so a PD 5 um from its centre is inside it; move the aggressor or set the PD
     site explicitly.
2. **params (schema, presets, realize)**
   - `ring.thermal.family` values: `inring_noundercut` (default), `silicide_undercut`, `metal_above`;
     `ring.thermal.rom_file`: `thermal_rom_<family>_<film>.npz`. g26 still runs `fidelity.thermal_model: interim`.
   - `uncertainty.thermal` ranges from the geometry Jacobians (1-sigma, ln): Z_heater->ring 0.16-0.29, tau_fast
     (t63 heater->ring) 0.24-0.48, crosstalk at pitch 0.27-0.51, self-heating tau 0.27 for the 160 nm no-undercut
     families (do not use the 0.7-1.2 values of the other artefacts: "Not done", item 6). Per-artefact
     `stages.uncertainty.uncertainty_thermal_suggestion` gives +/-2 sigma factors (in-ring 160: Z x0.63-1.60,
     time_scale_fast x0.55-1.83, crosstalk x0.57-1.74, self-heating tau x0.58-1.71).
   - `layout.ring_pitch_um` default 50 um (assumed) and the site offsets (`site_offsets_um`) used for `self_res`:
     monitor_pd (0, 25), temp_sensor (40, 0), driver (0, -30), aggressor (60, 0) um. Any layout value can be used
     through the radial `cross_res` table instead (d >= 20 um).
   - `rom_dc_scale` is 0.58 to 2.3 depending on family and dlambda/dT hypothesis (table above): only the undercut
     family and the no-undercut families under H86 stay inside [0.5, 2].
   - [review] Self-heating tau sigma_ln for the undercut family: about 0.6 (keep-out dominated), not 0.27 ("Not
     done" 6).
   - [review] g26 `ring.thermal.selfheat_tau_s` note says "the in-ring FEM ROM puts 65 % of Z_opt below 1 us (t63
     1.7 us)". The artefact gives 38.5 % of the DC in poles below 1 us
     (`summary.share_of_dc_in_poles_below_1us_opt_ring`) and 54 % of the step response at 1 us; t63 1.76 us is right.
     Please correct the note (pole shares are not unique between fits; quote the step fraction).
   - [review] In ROM mode `realize` loads one `ring.thermal.rom_file` for every world: the family, film and stack
     hypotheses (`uncertainty.thermal.family`, the film hypothesis, `uncertainty.thermal.stack`) do not select
     different artefacts, and the review's ROM run had to drop them to load. Six artefacts exist (3 families x 2
     films; no stacked-EIC artefact, "Not done" 11); a per-world ROM choice is needed before ROM mode can sample these
     hypotheses.
3. **validation (VALIDATION.md)**: VAL-8a, 8b, 8c fail and 8d passes with the numbers above
   (`thermal_val8_imec.json`, `comparison`; the sidecar `thermal_val8_imec.sidecar.json` carries the sha256). Report
   them as they are. rom_dc_scale per hypothesis for the closure table: `rom_dc_scale.table` in every sidecar
   (refreshed from the modes artefacts by `--stage propagate`; re-run it when a modes artefact changes:
   `rom_dc_scale.dlambda_dt_hypotheses.H_solver.sha256` names the file used).
4. **materials**: please add (a) air/vacuum for undercut cavities (this module uses k 0.026 W/(m K), rho c 1.2e3
   J/(m^3 K), handbook, flagged in the sidecar; the cavity k is irrelevant to < 1.4 %: `sweeps.air_k`), (b) a film-k law
   valid below 150 nm (the 60 nm slab and the 88 nm CMOS film use the 150 nm value, 69 W/(m K); Ju & Goodson's
   ~74 W/(m K) at 74 nm supports that), (c) SiN thermal data if the passivation should not be oxide.
5. **modes**: my femwell TE0 of rib160 (400 x 160, 60 nm slab, 2 um slabs) gives n_eff 2.4369 straight, the same as
   your rib160 straight value; bent at 7.5 um, 2.4421.

## Deviations from the task text

- 07 P1-3's "thin-film k 56-66 W/(m K)" is superseded by `mrm.materials` (71.8 / 80.0 W/(m K) for 160 / 200 nm), as
  the materials handoff says.
- Cross terms: the frequency-domain check (0.3 dB, 3 deg) is applied where |H| >= 1e-3 |H(0)|; above that frequency
  the diffusive cross response is below -60 dB and its phase is not defined in a useful sense. The time-domain check
  covers the whole window.
- The production level is the 50 nm mesh (scale 1.0), not the finest level (35 nm), because of the former 3 GB cap
  ("Memory cap"); the 35 nm level is in the convergence study and bounds the production error (<= 0.29 %).
- Superposition: the 07 metric "error on the crosstalk residues" is replaced by the step-response error (max over
  10 ns-10 ms of |y_bank - y_ROM| / y_ROM(inf)); fitted residues of two near-identical smooth curves are not unique, so
  a residue-by-residue comparison measures the fit, not the physics (the old residue-L1 numbers, 0.13-0.99, stay in
  the record for reference).
- The heater families are axisymmetric equivalents (the GF PDK four silicide arcs are two full annuli; a 3-D check
  gives +1.0 % on Z_heater->ring for 140-degree arcs); heater leads are only in the 3-D model (-4.0 % on
  Z_heater->ring at equal mesh).
- The 3-D checks use fewer mesh levels than the accuracy rule asks (two single-ring levels, one bank level).
- [review] The superposition number compares the 3-D bank with the axisymmetric FEM at the exact bank distances in the
  comparison variant (Dirichlet, 200 um substrate), not with the production table: it excludes the Foster-fit error
  (< 1.4 %) and the distance-interpolation error (0 at grid points, -2.5 % at 350 um), and assumes that the 3-D /
  axisymmetric ratio carries over to the production boundary (Robin, 775 um).
- [review] The 07 P1-3 stack hypothesis (stacked EIC) was not extracted ("Not done" 11).
- [review] Integrator: 07 is normative (ARCHITECTURE header), and `docs/CHANGELOG.md` does not yet record the
  superposition-metric change (07 "error on the crosstalk residues" -> the step-response error above). This task does
  not own the CHANGELOG; please add the entry.
- VAL-8 uses our materials (validation-owner rule); imec's Table I set is a variant.
