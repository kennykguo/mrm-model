# Handoff: extract_junction (`mrm/extraction/junction.py`, `junction_<xsec>[_lateral].json`)

Status (2026-10-09, resumed run, no solver started): **one of the four planned artefacts exists**,
`junction_rib160.json` (rib160, L-shaped junction, the default hypothesis). It is complete, it has the ARCHITECTURE 5.4
format, the runtime reads it (LUT mode realizes; see "Runtime"), and its fast tests pass. `junction_rib200.json`,
`junction_rib160_lateral.json` and `junction_rib200_lateral.json` do not exist (see "Not done"). The resumed run
did not re-run any solver. It added post-processing of the stored data only (`--postprocess`, `--mesh-report`).
A review followed (also without a solver; see "Review" at the end): it corrected the convergence claims, the
assumption labels and the analytic-law handoff, and extended the sidecar and the tests.

## Files

- `mrm/extraction/junction.py`: the extraction, plus the no-solver helpers added in the resumed run
  (`postprocess`, `tidy3d_reassessment`, `modes_consistency`, `bend_formulation_bias`, `mesh_counts`,
  `mesh_report`, `estimate_peak_gb`, `devsim_settings`, `not_done_record`) and in the review (`free_order_rows`,
  `jacobian_remesh_noise`, `range_notes`, `doping_ranges`; see "Review").
- `mrm/data/extracted/junction_rib160.json` (sha256 `809979b8…`, unchanged since commit be4e80e) and
  `junction_rib160.sidecar.json` (post-processed; `history` records it).
- `tests/verification/test_extraction_junction.py`: 33 fast tests pass (also with `-W error`), 71 skip (the three
  missing artefacts), 1 slow test is deselected (it needs `junction_rib160_lateral.json`; no slow test re-runs the
  committed L-shape artefact, see "Not done").

## Commands

```
MG=".venv/bin/python tools/memguard.py --heavy --"          # 6 GB cap, machine-wide heavy lock
$MG .venv/bin/python -m mrm.extraction.junction --xsec rib160 --topology lshape --doping-sweep [--quick]
.venv/bin/python -m mrm.extraction.junction --postprocess     # no solver; sidecars only, data files unchanged
.venv/bin/python -m mrm.extraction.junction --mesh-report [--scales 2.0 1.4142 1.0]   # no solver
```

New options: `--scale S` (DEVSIM production scale; 1 = 1 nm at the junction), `--ladder a b c` (convergence scales,
coarse to fine), `--study-scale S` (Jacobian, oxide extent, temperature, doping sweep), `--direct-mode-res-floor R`
(um, coarsest mode mesh of the direct re-solve). The defaults are the settings of the committed artefact (not
re-run to check bit-identity). A new run records them in `mesh.settings` and in `command` (with `--doping-sweep`
when it ran). The committed sidecar now carries `command` (explicit settings) and `command_as_run` (the logged
memguard command, `--limit-gb 2.7`, no `--heavy`).

## Runtime

- Contract keys at the top level (read by `mrm.plant.electrical.load_junction_table` and
  `mrm.params.realize._junction` through `resolve_from`, which checks the sha256): `v_reverse_v` (-0.5 .. 4.0 V,
  0.05 V, forward negative, 91 points), `dneff` (bent TE0 at r_ref, relative to 0 V), `alpha_fca_per_m` (absolute,
  intensity, full circumference), `capacitance_f_per_m` (low-frequency dQ/dV), `r_series_ohm_m` (R_s = r'/(fill L)).
  Also `doping`, `method: devsim`, `depletion_check`, `fill_fraction: 1.0`.
- Smoke test (no solver; now the fast test `test_runtime_realize_in_lut_mode`): g26 with `fidelity.junction_model:
  lut`, `ring.junction.lut_file: extracted/junction_rib160.json` and the analytic-only keys removed loads and
  realizes (model 1, 91-point grid; the derived eff_at_0v is 66.2 pm/V at the preset fill 0.8, 70.5 pm/V at the CAL-5
  fill 0.852). `junction_table_from_plant_params` and `PhaseParams` build from it.
- n_g: g26 realizes n_g 3.9525 (L 42.14 um), the junction 3.9388. The CAL-5 fill scales with n_g, so with g26's n_g
  the endpoint fill is 0.855 (0.852 x 3.9525/3.9388), and the ring C_j, R_s follow from f L at runtime.
- g26 still uses `fidelity.junction_model: analytic` (its comment "the junction LUT artefact is not extracted yet" is
  stale for rib160). That is the runtime fallback for everything this task did not produce (rib200, the lateral
  topology). In LUT mode, `ring.junction.lut_file` is one file: a world drawn with `uncertainty.film_thickness`
  '200' would still read the rib160 LUT, and no rib200 LUT exists, so keep analytic mode (or pin film 160) when the
  film hypothesis is sampled. If the integrator keeps the analytic law, the LUT gives these values for it
  (fill 0.852, L = c/(n_g FSR) = 42.29 um), see "Analytic law fitted to the LUT".
- Loss budget preset: g26 `ring.optics.loss_budget.alpha_fca_db_per_cm` is 80 dB/cm (ring level, eta 0.848). This LUT
  gives a ring-average 51.6 dB/cm at 0 V (fill 0.852); the CAL-5-consistent dopings of the sweep give 48.9 (3e18,
  fill 0.874) to 74.3 dB/cm (5e18, fill 0.710). The g26 alternatives eta_0p5 (41.7) and eta_0p65 (58.2) bracket the
  nominal LUT value.

## Key numbers: `junction_rib160.json`

Assumptions (all `assumed`, Lightmatter publishes none): rib 400 nm x 160 nm, slab 60 nm, r_ref 6.727 um (modes
artefact); L-shaped junction: shallow n layer from y = 90 nm, vertical leg 100 nm inside the n-side rib edge, n on the
outer side; N_A = N_D = 3.2e18 cm^-3 (chosen so that CAL-5 needs fill 0.85; slightly above the 5e17-3e18 range that
docs/research quotes and that the artefact's `doping.provenance` claims to contain, inside the assumed range
5e17-5e18), P+/N+ 1e19 at 250 nm, P++/N++ 1e20 at 600 nm, contacts at 1000 nm, rho_c 1e-7 ohm cm^2. Nedeljkovic 2011
plasma dispersion, eps_Si 11.7. V_bi 1.016 V. Labelling gaps of the stored artefact (sidecar `inputs.range_notes`):
`doping.ranges.junction_offset_nm` [-60, 60] is the lateral range about the rib centre and does not contain the
L-shape nominal 100 nm (read it as 40-160 nm, as the Jacobian table below does); `y_junction_nm` and `outer_side` have
no range (the code now writes 60-120 nm, a reviewer assumption, and {n, p}); N++ level, implant grading and contact
width are fixed without a range; `xsec.ranges.film_nm` 155-205 spans both films (the modes artefact uses 150-170);
the sidewall (90 deg) has no range, as in the modes artefacts.

The numerical errors below are the supported scale: the spread of the mesh levels. The stored Richardson numbers
(DEVSIM 2e-5 to 5e-4 with order 2, mode mesh 3.6e-5 / 5.9e-5 with order 4) forced an order that the levels do not
show (see "Convergence evidence"); the sidecar flags them `order_forced`.

| quantity (full circumference, fill 1) | value | numerical error (rel; spread) | model and assumption spread |
|---|---|---|---|
| dn_eff(1.6 V), (1 V), (4 V), (-0.5 V) | 2.982e-4, 2.032e-4, 5.865e-4, -1.474e-4 | 8e-4 (DEVSIM, 2.83-2.0 nm; up to 1.3e-3 at -0.5 V), 1.1e-4 (mode mesh) | Soref-Bennett +7.1 %; bend map -0.6 to -1.5 % (below) |
| average 0 to 1.6 V, fill 1 | 61.99 pm/V | as above | doping +28 / +31 pm/V per decade (N_A / N_D) |
| alpha_FCA(0 V), (2 V), (4 V) | 1393.9, 1178.7, 1039.5 /m (60.5, 51.2, 45.1 dB/cm) | 2.5e-4 (DEVSIM), 2.0e-4 (mode mesh); increments do not shrink | Soref-Bennett -5.1 %; bend map +0.9 to +2.2 % |
| C'(0 V), (1 V), (2 V), (4 V) | 1.479e-9, 1.046e-9, 8.55e-10, 6.62e-10 F/m | 1.0e-3 (2.3e-3 if first order) | fringe 9.5 % of C(1 V); oxide domain 0.6 -> 1.2 um +0.8 %; AC (20 GHz) 0.3-0.6 % lower |
| r_s'(1 V) | 2.955e-3 ohm m (contact term 9.4e-5) | 2.3e-5 (observed order 5.7), spread 1.4e-4 | slab -4.1e-5 ohm m per nm; N_A -1.7e-3 per decade |
| f_RC(1 V) = 1/(2 pi r' c') | 51.5 GHz (62.8 GHz at 2 V) | | no pads, ideal driver |
| loss ratio q(V) | 0.0635 / 0.0632 / 0.0629 at 0 / 1 / 2 V | | preset 0.06 (0.03-0.15) |
| bent mode | n_eff 2.44323, n_g 3.93876, dn_eff/dn_Si 0.96758 | n_eff 6e-6, n_g 6e-7 | isotropic bend map (below) |

At the CAL-5 fill 0.852 and L = 42.29 um: C_j = 53.3 / 37.7 / 30.8 / 23.9 fF at 0 / 1 / 2 / 4 V, R_s = 82 ohm,
efficiency 70.5 pm/V at 0 V, 48.4 at 1 V, 38.5 at 2 V, 29.3 at 3.75 V; a 1.0 -> 1.5 V step shifts 22.7 pm (07 P1-7
test: about 25 pm).

Temperature (DEVSIM at 340 K against 298 K, same mesh): d ln(dn_eff)/dT 3.6e-5 /K at 1.6 V, d ln(alpha_FCA)/dT
2.1e-3 /K at 0 V (this is the c_T of 07 P0-2; it contains the assumed plasma temperature factors), d ln C/dT 2.7e-4
/K, d ln r_s/dT 1.2e-3 /K, dV_bi/dT -0.92 mV/K. The runtime model "look up the 298 K table at V + dV_bi/dT (T - T_ref)"
reproduces the 340 K dn_eff within 1.7-2.6 % in reverse bias (1.6e-5 absolute, about 10 %, at -0.5 V forward).

Assumption sensitivities (artefact `jacobian`, per nm or per decade; central differences at the study mesh, geometry
steps re-solve the mode). Linearized effect over the assumed range (the junction offset as nominal +-60 nm):

| input (range) | average pm/V at fill 1 (61.99) | alpha_FCA(0) (1394 /m) | C'(1 V) (1.046e-9) | r_s'(1 V) (2.955e-3) |
|---|---|---|---|---|
| slab 40-100 nm (60) | +4.4 / -8.7 | -214 / +428 | -7e-12 / +1.4e-11 | +8.1e-4 / -1.6e-3 |
| junction offset +-60 nm | +-9.2 | -+85 | +-1.5e-10 | +-1.2e-4 |
| grading 2 -> 30 nm (forward step) | -25 | -237 | -3.6e-10 | +4e-6 |
| width 360-450 nm (400) | -0.6 / +0.7 | +7 / -9 | -5.1e-11 / +6.4e-11 | -5e-5 / +6e-5 |
| N_A, N_D (per decade) | +28 / +31 | +1352 / +2639 | +7.0e-10 / +4.4e-10 | -1.7e-3 / -1.2e-3 |
| plus gap 150-500 nm (250) | 0 | +34 / -85 | 0 | -3.1e-4 / +7.8e-4 |
| p side outside instead of n (difference) | +1.6 | -63 | 0 | 0 |

The N_A/N_D row matters less than it looks: the doping was chosen so that CAL-5 needs fill 0.85, and the runtime
calibrates fill to CAL-5. The CAL-5-consistent dopings (3e18-5e18) give C_j(1 V) of 38.9-40.3 fF at their fill
(L 44 um) and eta_FCA 0.42-0.76 (doping sweep below), so C_j is robust and eta is not. The y_junction step (10 nm)
has no assumed range in the artefact (`doping.ranges` lacks it): per nm +0.087 pm/V, -2.6 /m, +2.1e-12 F/m; over the
range the code now writes (60-120 nm, a reviewer assumption) that is +-2.6 pm/V, -+78 /m, +-6.2e-11 F/m.

Limits of this table (review):
- Linearization: r_s is close to 1/slab for its slab part (72 % of the 1-D sum `aux.r_series_analytic_parts`), so the
  linear -1.6e-3 ohm m (-55 %) at 100 nm overstates the drop; scaling the slab part as 60/slab gives about +36 % at
  40 nm and -29 % at 100 nm. The other rows are close to linear over their ranges, except N_A/N_D (see the doping
  sweep).
- Mesh noise: only the N_A, N_D, grading and outer_side steps reuse the nominal DEVSIM mesh. Width, film, slab,
  junction offset, y_junction and the gap steps re-mesh (the +-step meshes have 1-9 % different node counts), so their
  differences carry mesh noise. Its upper estimate (sidecar `uncertainty.jacobian_remesh_noise`, from the ladder
  spread) exceeds 20 % of the entry for d(dn_eff)/d(film) (0.2-0.5), d(average pm/V)/d(width) (0.2), dC/d(film)
  (0.5-0.75) and dC/d(slab) (0.29); all of these move their output by less than 2 % over the range. The implant and
  contact gaps have no resolvable effect on dn_eff or C (zero within noise).

## Reports (artefact `reports`)

CAL-5 fill (52.8 pm/V average over 0 to -1.6 V) for each n_g hypothesis (fill scales with n_g), no self-heating
relief:

| hypothesis | n_g | fill (endpoint) | fill (least squares) |
|---|---|---|---|
| H_solver (bent mode at r_ref) | 3.9388 | **0.852** | 0.876 (rms 4.7 pm) |
| H86 via R1 (85.9 pm/K with the solver dn_eff/dT) | 2.983 | 0.645 | 0.664 |
| FSR 1.8 THz (1.72-1.88) at R = 7.5 um | 3.534 | 0.764 (0.732-0.800) | 0.786 |
| FSR 1.4 THz (1.32-1.48) at R = 7.5 um | 4.544 | 0.983 (0.930-1.042) | 1.011 |
| FSR 1.53 THz (G25 cluster) at R = 7.5 um | 4.158 | 0.899 (0.893-0.905) | 0.925 |
| FSR 1.6 THz (PDK size) at R = 7.5 um | 3.976 | 0.860 (0.844-0.871) | 0.885 |

With the exact bend transform the H_solver fill would be 0.839-0.847 (see "Bend formulation"). If the G25 read-off
was taken at 0 to +4 dBm, the self-heating relief raises the fill to 0.90-1.19 (`cal5.selfheat_relief`; the power is
not published). The G25 ladder point at 0.8 V (56.4 pm) is 8.8 pm above the LUT shape (47.6 pm at fill 0.852); 1.2
and 1.6 V agree (66.98 vs 66.8, 84.48 vs 84.4).

EO bandwidth (RC single pole x exact small-signal TCMT, 03 section 5.6; critical coupling; q from the LUT). Nominal
case (L from the 1.8 THz FSR, R_src 50 ohm, x = 1/sqrt(3)):

| bias | Q_L 3500 | Q_L 4200 | target | range over all 32 cases |
|---|---|---|---|---|
| -1 V | **30.0 GHz** | 28.0 GHz | CAL-10 30.9 +- 2 | 26.0-51.0 |
| -2 V | **33.95 GHz** | 31.2 GHz | VAL-3 35.4 +- 3 (held out) | 29.0-55.5 |

Ratio -2 V / -1 V: 1.13 (Q_L 3500), 1.11 (4200); measured on one device 1.113. Without the 50 ohm source the
junction alone gives 39.1 / 42.5 GHz. No pad capacitance is included (no number is public). The nominal row is the
x = +1/sqrt(3) side; on the other side of the resonance (x = -1/sqrt(3)) the loss modulation (q 0.063) lowers it to
28.6 / 32.3 GHz (Q_L 3500), still inside both bands (re-derived from the LUT by `test_report_identities_against_the_lut`).

Loss budget (07 P0-2), at the CAL-5 fill 0.852:

| bias | alpha_FCA ring average | eta_FCA at Q_i 7000 / 8400 |
|---|---|---|
| -0.5 V | 1264 /m (54.9 dB/cm) | 0.468 / 0.562 |
| 0 V | 1187 /m (51.6 dB/cm) | 0.440 / 0.528 |
| 1 V | 1082 /m | 0.401 / 0.481 |
| 2 V | 1004 /m | 0.372 / 0.446 |
| 4 V | 885 /m | 0.328 / 0.394 |

alpha_i from Q_i 7000-8400 is 2699-2249 /m (117-98 dB/cm). The FCA explains 44-53 % of it at 0 V; with 0.73 dB/cm
passive loss, 45-65 dB/cm remain unexplained. So eta lies between 0.44-0.53 (the remainder does not heat) and 0.99
(it all heats). The P0-2 premise "the loss is mostly FCA" holds only at the high end of the doping range (5e18:
eta_FCA 0.63-0.76). d eta/dV at 1 V: -0.033 / -0.039 per V. Q_i rises with reverse bias by 4.8-5.8 %/V at 0 V and
3.3-3.9 %/V at 1 V (from fill d alpha/dV / alpha_i, if the rest of alpha_i does not depend on V).

Doping sweep (L-shape, rib160, study mesh 2.83 nm, mode mesh 16 nm, `reports.doping_sweep`):

| N_A, N_D (cm^-3) | fill | pm/V at fill 1 | alpha_FCA(0) full (dB/cm) | eta_FCA at fill (Q_i 7000 / 8400) | f_RC(1 V) | consistent |
|---|---|---|---|---|---|---|
| 5e17, 5e17 | 2.106 | 25.1 | 7.3 | 0.062 / 0.075 | 56.2 | no |
| 1e18, 1e18 | 1.425 | 37.1 | 15.1 | 0.129 / 0.154 | 54.8 | no |
| 2e18, 2e18 | 1.037 | 50.9 | 34.1 | 0.291 / 0.349 | 53.0 | no |
| 3e18, 3e18 | 0.874 | 60.4 | 55.9 | 0.417 / 0.500 | 51.8 | yes |
| 5e18, 5e18 | 0.710 | 74.4 | 104.7 | 0.634 / 0.761 | 50.4 | yes |
| 1e18, 3e18 | 1.250 | 42.2 | 42.8 | 0.365 / 0.438 | 56.8 | no |
| 3e18, 1e18 | 1.197 | 44.1 | 30.7 | 0.262 / 0.314 | 52.2 | no |

## Analytic law fitted to the LUT (for the analytic fallback; no solver)

At fill 0.852, -0.5 to 4 V: dlambda = A((V + V_bi)^m - V_bi^m) with V_bi = 1.016 V (fixed, the artefact's value):
A = 163.8 pm/V^m, **m = 0.437**, rms 0.13 pm, max 0.50 pm; eff at 0 V 70.9 pm/V. With V_bi free: V_bi 1.116 V,
m = 0.407, A = 185.4, rms 0.05 pm. C'(V) = C0 (1 + V/V_bi)^-m_C over 0 to 4 V: C0 = 1.476e-9 F/m, **m_C = 0.502**
(V_bi fixed; max error 0.15 %; over -0.5 to 4 V: 1.482e-9, 0.506, max 0.5 %). So m + m_C = 0.94, not 1: dn_eff is not
proportional to the removed charge. The LUT shows it directly: dn_eff per removed electron (`aux.electrons_per_m`)
falls by 5.6 % from -0.5 to 4 V, while the integral of C dV equals the removed charge to 2.5e-4. The depletion edges
move into regions of lower optical weight (and the fringe charge at the oxide surfaces carries little); the NSM
sublinearity in N matters only in the partly depleted tails. Hence `analytic_junction_table`'s default
m_cap = 1 - m (0.563) would overstate the C-V exponent by about 0.06 (C at 1 V 4 % low); pass m_cap = 0.50 (there is
no schema key for it yet: `junction_table_from_plant_params` calls the table without m_cap, so plant_electrical or
params would have to add one). Ring-level values for the g26 analytic keys: eff_at_0v 70.9 pm/V (preset 72.5), v_bi
1.016 V (preset 0.9, range 0.8-1.0: outside), depletion_exponent 0.437 (preset 0.5), capacitance_at_0v 53.3 fF
(preset 15, range 6-30: outside), series_resistance 82 ohm (preset 100), loss_ratio_q 0.064 at 0 V (0.063 at 1 V;
preset 0.06). Give q directly. If `dqi_dv_frac_per_v` is used instead, the runtime converts it with eff_at_0v
(`analytic_junction_table_from_ring`: q = dqi alpha_i lambda^2 / (4 pi n_g eff0)), so it must be the 0 V value:
+0.048 (Q_i 7000) to +0.058 (Q_i 8400) per V; the 1 V value (+0.033 to +0.039) would give q of about 0.043, 32 %
low.

## Convergence evidence (sidecar `convergence`)

- DEVSIM mesh, three levels h_j = 2.83 / 2.38 / 2.0 nm (4353 / 6149 / 7880 Si nodes), production 2.0 nm:
  dn_eff(1.6 V) 2.98480 / 2.98278 / 2.98234e-4; C'(1 V, AC 20 GHz) 1.04292 / 1.04203 / 1.04190e-9; alpha(0) 1393.55 /
  1393.68 / 1393.90; r'(1 V) 2.95416 / 2.95446 / 2.95457e-3. The stored Richardson errors (3.5e-4 dn_eff 1.6 V,
  2.3e-4 1 V, 8e-5 4 V, 3.9e-4 alpha 0 V, 5.2e-4 alpha 2 V, 3.0e-4 C 1 V, 1.8e-4 C 2 V) all FORCED order 2; only r_s
  shows an order (5.7, error 2.3e-5). Review of `free_order_view`: the cause is not only the narrow ladder. No
  sequence is in the asymptotic range: apparent orders 8.9-17 for dn_eff and C (the first increment dominates), the
  alpha increments grow under refinement (1.3e-1 then 2.3e-1 /m), and dn_eff(-0.5 V) changes sign. The narrow ladder
  (2^0.25 per step) makes the increments (1e-5 to 1e-3) comparable with mesh-to-mesh noise, so no order can be
  observed. Supported error scale: the coarse-to-fine spread, 1.4e-4 (r_s) to 1.3e-3 (dn_eff at -0.5 V); 8e-4 for
  dn_eff(1.6 V), 1.0e-3 for C(1 V), 2.5e-4 for alpha(0) (2.4x these at first order). Independent evidence: the
  lateral junction's sqrt(2) ladder 2.0 / 1.41 / 1.0 nm (logged run, "Numbers from the interrupted runs") moved
  dn_eff(1.6 V) by 1.3e-3 (order 2.1), C by 6.3e-4 (order 1.0) and alpha by 5.0e-4 between 2.0 and 1.0 nm. So the
  2.0 nm L-shape values carry about 1e-3 of mesh error (C up to about 0.2 % if first order).
- Mode mesh, five levels 40 / 28.3 / 20 / 14.1 / 10 nm (production 10 nm): n_eff error 1.5e-5 (order 3.6), n_g 2.5e-6
  (order 4.4), dn_eff/dn_Si 1.0e-5 (order 3.0): observed. The overlap integrals are not: dn_eff(1.6 V) is
  non-monotone over the five levels and alpha(0) has its largest increment last (1394.15 -> 1393.90), so their stored
  order 4 was forced and the stored 3.6e-5 / 5.9e-5 are not supported. Spread of the three finest levels: 1.1e-4
  (dn_eff 1.6 V) and 2.0e-4 (alpha 0 V); over all five levels 4.7e-4 and 2.0e-4. The weight interpolation onto the
  DEVSIM nodes is the likely noise source (not checked).
- Oxide domain 0.3 / 0.6 / 1.2 um: C'(1 V) 1.0304 / 1.0429 / 1.0514e-9; production 0.6 um, +0.8 % to 1.2 um (the
  fringe field converges slowly: the increments shrink only by 1.48x per doubling, and a geometric tail fitted to them
  puts C about 2.5 % low against an unbounded oxide; real BEOL metal and the substrate below the 2 um BOX bound it, so
  1-3 % low).
- Bias grid 0.05 V: linear-interpolation error at mid-points 1.3e-4 of dn_eff(1.6 V) (0.65 % relative just above 0 V).
- AC frequency: r' at 20 vs 50 GHz differs by -6.9e-4; C(20 GHz)/C(1 MHz) = 0.996 at 1 V.

## Cross-checks (sidecar `validation`)

- Direct femwell re-solve with the carrier-perturbed complex permittivity (14.1 nm mesh) vs the first-order overlap:
  dn_eff -0.52 / +0.16 / -0.006 / -0.08 % at -0.5 / 1 / 2 / 4 V; alpha -0.03 / -0.002 / +0.02 / +0.09 %.
- Index weight: femwell integral vs finite-difference re-solve -1.3e-5; DEVSIM node quadrature vs femwell -5.4e-5;
  sum n_g,i w_i = n_g to 8.9e-7.
- C: AC (20 GHz) vs dQ/dV within 0.6 % (reverse bias); electron and hole charge balance 3.6e-12.
- Depletion width, DEVSIM vs abrupt formula (same eps 11.7, mid-leg cut): 1.5-4.3 % smaller (mean 2.3 %); vs the
  Debye-corrected formula 1.2 %. 05 C7c reported 5-8 % with DEVSIM's default eps 11.1 (sqrt(11.1/11.7) accounts for
  2.6 %), so the two agree.
- Analytic box-depletion model through the 2-D weight vs DEVSIM: dn_eff -1.4 to -2.4 %, C -4.3 to -4.6 % (no fringe;
  DEVSIM without oxide agrees with the analytic dn_eff to 0.2-0.6 %), alpha(0) +0.01 %, r_s +11 % (1-D paths).
- **Tidy3D (reassessed, no solver).** The stored `delta_n_eff` = -0.0177 is NOT a valid result: the junction's Tidy3D
  stage (uniform 20 / 10 / 6.7 nm grids, no subpixel averaging) fits no power law (d12/d23 = 1.49, below 1.71, the
  minimum for any positive order with refinement ratios 2 and 1.5), and the code forced first order. Its finest grid is
  3.2e-4 from femwell in n_eff and 6.3e-4 in dn_eff/dn_Si. The independent check is the modes artefact's Tidy3D run on
  the same rib160 (interface-aligned 10 / 5 / 2.5 nm, order 1.18): straight n_eff 2.437058 +- 0.0019 against this
  femwell 2.436840 (+2.2e-4). Status: consistent through the modes artefact; the junction's own stage is inconclusive.
  The code no longer forces the order (`tidy3d_check`), and new runs store the same reassessment.
- Modes artefact (independent femwell set-up, same geometry): straight n_eff -6.0e-5, n_g +7.5e-6, Si index weight
  -2.7e-5; r_ref identical (1e-7 um). Bent: n_eff +3.3e-3, n_g -4.4e-4, Si weight -8.6e-4; the bent differences are
  the bend formulation.

## Bend formulation (open issue; sidecar `uncertainty.model.bend_formulation`)

The bent mode comes from femwell's `compute_modes(radius=...)`, the isotropic conformal map. The modes task showed
that this map overstates the bend shift of n_eff 2.1x (0.00640 vs 0.00303; the exact transform agrees with Tidy3D's
0.00302 to 0.4 %). The stored bend effect on this junction is dn_eff -2.8 % (1.6 V; -2.3 to -3.2 % over the bias
range) and alpha_FCA +4.2 % (0 V). With the exact effect = k x the isotropic one, k = 0.47-0.78 (exact/isotropic
ratio of the n_eff shift and of the Si index-weight change of the same rib; n_g gives 0.72). Hence the stored values
are biased by **dn_eff -0.6 to -1.5 %** and **alpha_FCA +0.9 to +2.2 %** (hard bounds: k = 0 and 1, -2.8 % / +4.2 %).
Consequence after CAL-5 calibration: fill 0.839-0.847 instead of 0.852 (C_j and 1/R_s scale with it), ring-average
alpha_FCA(0) 1145-1170 /m instead of 1187 (eta -1.5 to -3.6 %), shape change over 0-4 V 0.2-0.4 %. These are below
the plasma-model (7 %) and doping spreads. Fix (not done, needs solvers): solve the bent TE0 with
`mrm.extraction.common.FemProblem(...).solve(formulation="exact")`, take the index-weight density of the transformed
tensors (as `mrm.extraction.modes.index_weight_block` does) in `solve_te0`/`ModeResult`, and re-run the extraction.

## Deviations from the original task

- Default topology L-shape, not lateral. The lateral junction cannot reach CAL-5 with fill <= 1 at any doping up to
  5e18 (fill 1.46-3.46, unarchived sweep below). The lateral artefacts were planned as the alternative.
- The body doping is selected by CAL-5 (fill 0.85 target). Doping and fill are not separately identifiable from CAL-5;
  the Jacobian and the doping sweep carry the rest.
- C(V) in the LUT is the low-frequency dQ/dV of the DC sweep; AC is the check (0.6 %).
- The DEVSIM model is 2-D (feasible), with the oxide as a Poisson region; no 1-D correction needed.
- The bent mode uses femwell's isotropic map (above); the Tidy3D stage is inconclusive (above).
- The L-shape mesh is coarser than intended (memory, below).

## Not done (sidecar `not_done`; commands use the 6 GB heavy default)

| missing | why | command (committed settings) | runtime fallback |
|---|---|---|---|
| `junction_rib200.json` (L-shape) | DEVSIM convergence failure at the coarse ladder level 4.0 (N 4.2e18; not memory); ladder and DC fallback tolerance fixed afterwards; re-run never started | `$MG .venv/bin/python -m mrm.extraction.junction --xsec rib200 --topology lshape --doping-sweep` | analytic law (g26 default) for film 200 |
| `junction_rib160_lateral.json` | killed at the direct re-solve, 2.71 GB > 2.7 GB; re-run never started | `... --xsec rib160 --topology lateral --doping-sweep` | none needed (alternative topology) |
| `junction_rib200_lateral.json` | killed in the AC stage, 2.73 GB > 2.7 GB; re-run never started | `... --xsec rib200 --topology lateral --doping-sweep` | none needed |

With the intended settings (fit under 6 GB, estimated peak 3.1 GB for rib200 L-shape): add `--scale 1.4142 --ladder
2.8284 2.0 1.4142 --study-scale 2.0 --direct-mode-res-floor 0.010` to the L-shape command, and
`--direct-mode-res-floor 0.010` to the lateral commands. Each run takes the heavy lock; the rib160 L-shape full study
took 48 min; a lateral ("light") run takes an estimated 20-30 min (the interrupted one reached the direct re-solve
after 9 min). Afterwards run `--postprocess`. The slow test
`test_quick_rerun_matches_committed` needs `junction_rib160_lateral.json`. No slow test re-runs the committed
L-shape: `--quick` uses the nominal 2e18 doping and skips the CAL-5 doping selection (a logged quick L-shape run gave
dn_eff(1.6 V) 2.45e-4 against 2.98e-4), so it cannot be compared with the committed artefact; a slow test for it
needs an in-process `extract(..., quick=True, doping=<the artefact's doping>)` with a tolerance set from one run (not
done: it needs a solver run). Its quick ladder includes scale 4.0, which failed to converge for rib200 at 4.2e18.

The "intended" L-shape settings: the production DC sweep at scale 1.4142 is in the run log (killed); the ladder
(2.83, 2.0, 1.41) and study scale 2.0 are inferred from the lateral pattern (ladder ending at production in sqrt(2)
steps, study one step coarser), not logged. The run logs (`prod.log`, `prod3.log`, `dsweep160.txt`) lived in a session
scratch directory, not in the repository; the reviewer checked this handoff's transcription of them (kills, peaks,
ladders, the lateral sweep table) against them on 2026-10-09.

Also not done: the bend-formulation fix (above); a pad-capacitance and driver model for the bandwidth (no public
number); the CAL-5 self-heating relief (the G25 power is not published); incomplete ionization, band-gap narrowing and
implant profiles (box/erf assumed). The `doping.provenance` string of the artefact says the lateral junction never
reaches fill <= 1: the only evidence is the unarchived sweep below.

## Memory cap

The production runs used `tools/memguard.py --limit-gb 2.7` (under the former 3 GB rule; `--heavy` was not used).
Sources: the run logs of the interrupted session, the code history (`git log -p`), the sidecar, and mesh counts made
now without solving (`--mesh-report`). The sidecar records the same in `mesh.memory`.

| what the cap changed | trigger (log) | accuracy consequence (recorded numbers) | fits 6 GB? |
|---|---|---|---|
| L-shape DEVSIM production mesh 1.41 -> 2.0 nm at the junction; ladder (2.83, 2.0, 1.41) -> (2.83, 2.38, 2.0) (inferred); study mesh 2.0 -> 2.83 nm (inferred) | rib160 L-shape DC sweep at 1.4142 (29868 nodes, 14862 Si nodes) killed at 2.86 GB | mesh error of the 2.0 nm values about 1e-3 (ladder spread 1.4e-4 to 1.3e-3; the lateral junction moved up to 1.3e-3 from 2.0 to 1.0 nm); at 1.41 nm roughly half. The narrow ladder (ratio 1.19) is the larger loss: no level sequence shows an order, so the stored Richardson errors rest on an assumed order. Study-mesh values differ from production by 8e-4 (dn_eff), 2.5e-4 (alpha), 1e-3 (C); this offset cancels in the doping-level Jacobian rows (same mesh) but not in the geometry and junction-position rows, which re-mesh (see "Limits of this table") | yes: >= 2.86 GB measured (killed sample), about 3.1 GB with AC. Scale 1.0 (27674 Si nodes) is 5.1-5.7 GB by the model: not safe with UMFPACK |
| direct re-solve mode mesh floored at 14.1 nm (production 10 nm) | rib160 lateral direct re-solve on the 10 nm mesh killed at 2.71 GB | 14.1 -> 10 nm changes dn_eff(1.6 V) by 1.1e-4 and alpha(0) by 1.7e-4, so the stored direct-vs-perturbative differences (up to 5.2e-3) carry about 1e-4 of mesh mismatch | yes (10 nm needed just over 2.7 GB) |
| AC solves split into 4 jobs instead of 2 | rib200 lateral AC (13346 Si nodes, 5 AC solves per job) killed at 2.73 GB | none (dn_eff of the AC jobs equals the DC run to 5.5e-16) | yes |
| three artefacts missing | the two lateral kills above; the queued re-runs never started before the restart (empty logs) | no LUT for rib200 or the lateral topology | yes, all three |

Not caused by the cap: the rib200 L-shape failure (DEVSIM convergence at the coarse level 4.0, peak 2.39 GB), the
Tidy3D grids (a 6.7 nm uniform grid needs little memory), the 16 nm Jacobian mode mesh and the 10 nm production mode
mesh (both original choices).

Peak-memory model (`estimate_peak_gb`): 0.30 GB + 1.72e-4 GB per Si node (calibrated on the 2.86 GB kill, a lower
bound), plus a superlinear variant (exponent 1.2). It reproduces the other records: L-shape 2.0 nm 1.66 GB (whole run
peaked at 2.00 GB), lateral rib160 1.0 nm 2.12 GB (not killed in DEVSIM stages), lateral rib200 1.0 nm 2.60 GB (killed
at 2.73 GB in AC). Mesh sizes (no solve): L-shape rib160 / rib200 at 1.41 nm: 14862 / 16155 Si nodes (2.9-3.1 GB);
at 1.0 nm: 27674 / 30306 (5.1-6.3 GB). Lateral at 1.0 nm: 10606 / 13346 (2.0-2.6 GB). The scipy SuperLU callback
(`DEVSIM_SOLVER = "superlu_scipy"`) needs less memory than UMFPACK (code note) but is untested in production.

Answer for Kenny: the 3 GB cap cost the stored numbers about 1e-3 relative (2.0 nm instead of 1.41 nm at the
junction), far below the 5-10 % model and assumption spreads. Its real costs are the convergence evidence of the
L-shape (a narrow ladder on which no order can be observed, so the stored Richardson errors rest on an assumed
order) and two of the three missing artefacts (the lateral runs were killed). 6 GB allows the intended settings for
all four artefacts.

**Does 6 GB limit accuracy?** Not for this task. The finest DEVSIM level worth running (1.41 nm) needs about 3 GB.
The intended ladder (2.83, 2.0, 1.41) already has the sqrt(2) steps that the lateral run needed to show an order.
One level finer (1.0 nm: 5.1-6.3 GB estimated) would only tighten a numerical error of about 1e-3; the SuperLU
callback could make it fit. The accuracy
limits of this artefact are the assumptions (doping, slab, junction position and grading), the plasma model (7 %)
and the bend formulation (1-2 %), none of which memory affects.

## Numbers from the interrupted runs (logs only, not artefacts; earlier code versions)

rib160 lateral, N_A = N_D = 2e18, scale 1.0 (10606 Si nodes), killed later at the direct re-solve:
dn_eff(1.6 V) 1.23682e-4, 25.71 pm/V at fill 1 (CAL-5 would need fill 2.05), alpha(0) 905.2 /m, C'(1 V) 4.0988e-10
F/m, r'(1 V) 3.1959e-3 ohm m, f_RC 121.5 GHz. DEVSIM ladder 2.0 / 1.41 / 1.0 nm: dn_eff(1.6 V) 1.2352181 / 1.2363009
/ 1.2368229e-4 (order 2.1), C'(1 V) 4.092827 / 4.091315 / 4.090241e-10 (order 1.0), alpha(0) 904.752 / 905.194 /
905.201. Mode ladder 20 / 14.1 / 10 nm: dn_eff(1.6 V) 1.2365324 / 1.2367309 / 1.2368229e-4.

rib200 L-shape: bent n_eff 2.633634, n_g 4.00856, dn_eff/dn_Si 1.01540 (FD 1.01540), power fraction 0.7726; straight
n_eff 2.625743, n_g 4.01458. Doping selection: fill 2.239 / 1.859 / 1.560 / 1.298 / 1.147 / 0.969 / 0.789 at
5e17 / 7e17 / 1e18 / 1.5e18 / 2e18 / 3e18 / 5e18, selected 4.2e18. Production (scale 2.0, 8598 Si nodes):
dn_eff(1.6 V) 3.0559e-4, 62.42 pm/V at fill 1, alpha(0) 2153 /m (93.5 dB/cm), C'(1 V) 1.2518e-9 F/m, r'(1 V) 2.0209e-3
ohm m, f_RC 62.9 GHz. At fill 0.85 that is eta_FCA about 0.67 at Q_i 7000.

rib160 lateral doping sweep (05:05 run, scale 1.414, mode mesh 16 nm, n_g 3.9402):

| N_A, N_D | fill | pm/V at fill 1 | alpha(0) dB/cm | eta_FCA at fill (7000 / 8400) | C'(1 V) F/m | r'(1 V) ohm m | f_RC GHz |
|---|---|---|---|---|---|---|---|
| 5e17 | 3.455 | 15.28 | 8.52 | 0.073 / 0.087 | 2.414e-10 | 4.809e-3 | 137.1 |
| 1e18 | 2.644 | 19.97 | 17.59 | 0.150 / 0.180 | 3.163e-10 | 3.888e-3 | 129.4 |
| 2e18 | 2.044 | 25.83 | 38.76 | 0.331 / 0.397 | 4.179e-10 | 3.103e-3 | 122.7 |
| 3e18 | 1.761 | 29.99 | 62.43 | 0.532 / 0.639 | 4.937e-10 | 2.694e-3 | 119.6 |
| 5e18 | 1.460 | 36.17 | 114.5 | 0.977 / 1.172 | 6.112e-10 | 2.241e-3 | 116.2 |
| 1e18 / 3e18 | 2.227 | 23.71 | 46.93 | 0.400 / 0.480 | 3.707e-10 | 3.499e-3 | 122.7 |
| 3e18 / 1e18 | 2.440 | 21.64 | 34.22 | 0.292 / 0.350 | 3.707e-10 | 3.071e-3 | 139.8 |

(eta at "fill" uses fill 1 here because the fill exceeds 1.) The lateral junction has the published-MRM capacitance
(17.7 fF at 1 V for 2e18, full circumference) but only 30-70 % of the CAL-5 efficiency.

## For other agents

- **integrator / params_content:** to use the LUT, set `fidelity.junction_model: lut`, `ring.junction.lut_file:
  extracted/junction_rib160.json`, drop the analytic-only keys, and use fill 0.852 (CAL-5, H_solver; 0.839-0.847 after
  the bend correction) or calibrate it. The LUT's C_j (53 fF at 0 V, 38 fF at 1 V at that fill) is outside the preset
  range of `capacitance_at_0v_ff` (6-30 fF), and V_bi 1.016 V is outside `v_bi_v` (0.8-1.0). P1-8 data-activity heat
  scales with C_j. The fitted analytic law above gives the fallback values if analytic mode stays; give
  `loss_ratio_q` 0.064 directly, or `dqi_dv_frac_per_v` at its 0 V value (+0.048 to +0.058), not the 1 V value.
  With g26's n_g (3.9525) the CAL-5 fill is 0.855. In LUT mode keep the film hypothesis at 160 (no rib200 LUT). The
  g26 FCA preset (80 dB/cm ring level) is above this LUT's 51.6 dB/cm at 0 V (48.9-74.3 over the CAL-5-consistent
  dopings); the eta_0p5 / eta_0p65 alternatives bracket it. The g26 comment "the junction LUT artefact is not
  extracted yet" is stale.
- **plant_electrical:** solver evidence on the exponent question (your note 1): shift exponent m = 0.44, C-V exponent
  m_C = 0.50, so m + m_C = 0.94. The default m_cap = 1 - m (0.563) overstates m_C by about 0.06 for this junction (C at
  1 V 4 % low); `junction_table_from_plant_params` has no way to pass m_cap from params yet.
- **plant_thermal / P0-2:** eta_FCA 0.44-0.53 at 0 V with this junction; c_T of alpha_FCA 2.1e-3 /K.
- **extract_modes:** if `modes_rib160.json` is re-extracted, its sha256 changes. The junction artefact stays valid (its
  own sha256 is unchanged), but its r_ref/n_g came from the old file: re-run `--postprocess` (the modes comparison
  tests skip until then) and consider re-extracting the junction.
- **validation_targets:** CAL-10 30.0 GHz (nominal) and VAL-3 33.95 GHz both inside their bands, conditional on
  R_src 50 ohm, no pads and the doping chosen from CAL-5.

## Method (short)

DEVSIM 2-D drift-diffusion on a structured triangular mesh of rib + slab + SiO2 cladding (Poisson in the oxide for
the fringe field), Arora mobility, n_i and V_bi from `mrm.materials`, permittivity 11.7 eps0 (05 C7c), ohmic
contacts at the slab ends. TE0 from femwell (order 2, isotropic conformal bend at r_ref). Index-weight density
w = c eps0 n (1 + x/R)^2 |E|^2 / P evaluated at the DEVSIM nodes; dn_eff = sum w dn A, alpha = sum w alpha A with
Nedeljkovic 2011 (Soref-Bennett kept as an alternative). C from dQ/dV of the 0.05 V DC sweep, checked against
DEVSIM small-signal AC at ten biases; R_s from AC Re Z (20 GHz) plus a transfer-length contact term.

## Numerical notes for anyone re-running DEVSIM on this machine

- DEVSIM's relative-error test never passes at nodes where the potential crosses 0 V; both contacts carry a +10 V
  offset after the equilibrium solve. In deeply depleted nodes the carrier equations stall at a 1.1e-10 relative
  update, so DC solves use 1e-8 (1e-6 as a fallback on coarse meshes at high doping; the run does not record whether
  the fallback was used).
- DEVSIM 2.11 keeps one complex matrix allocated per AC `solve()` (about 37 MB per call on a 15k-node mesh); long
  sweeps therefore run DC only and AC in separate small processes (four per production sweep).
- `reset_devsim()` also drops the UMFPACK solver registered at import; `_devsim_reset()` re-installs it.
- Every solver job runs in a fresh spawned process (`_isolated`), so femwell/DEVSIM memory is returned to the OS.
- The L-shape at the coarse scale 4.0 does not converge at high doping (rib200, 4.2e18); keep the ladder at or below
  2.83.

## Review (2026-10-09, no solver run; data file unchanged)

Checked: the ARCHITECTURE 5.4 keys and the 5 sidecar keys; the runtime path (`resolve_from` sha256, `realize` in LUT
mode, `load_junction_table`, `junction_table_from_plant_params`, `PhaseParams`); units and signs (reverse positive,
dn_eff > 0 and alpha falling under reverse bias, R_s = r'/(f L), C_j = f L c'); a hand re-derivation from the LUT of
every number in "Key numbers" and "Reports" (CAL-5 fills of all six hypotheses, C_j, R_s, f_RC, efficiencies, the
1.0 -> 1.5 V step, q(V), the nominal EO bandwidths through `eo_bandwidth_ghz`, the loss-budget rows and eta bounds,
the analytic-law fit, the Jacobian table, the bend-bias arithmetic). All reproduce. The input sha256 values
(materials.json, modes_rib160.json) still match. The run history and the lateral sweep match the logs.

Findings and what was changed:

1. Convergence evidence overstated (fixed in sidecar metadata, code and this file). The mode-mesh entries for
   dn_eff(1.6 V) and alpha(0) forced order 4 on sequences that show no order, without a flag; their stored 3.6e-5 /
   5.9e-5 are not supported (spread 1.1e-4 / 2.0e-4). The DEVSIM ladder's lack of an order was attributed to the
   narrow ladder alone; the levels are not asymptotic (apparent orders 9-17, growing alpha increments, a sign change).
   `postprocess` now writes `free_order_view` (with `apparent_order`, `pattern`, spreads) for both ladders, flags
   every forced stored entry `order_forced`, and adds the spreads to `uncertainty.numerical`.
2. `richardson` accepted any order from an exactly geometric ladder (no upper bound) but bracketed [0.2, 8] otherwise;
   now both branches use [0.2, 8] (`test_richardson_rejects_out_of_range_orders`).
3. Jacobian mesh noise (new `uncertainty.jacobian_remesh_noise`): the statement "Jacobians are differences on one
   mesh" held only for the N_A, N_D, grading and outer_side rows; see "Limits of this table".
4. Assumption labels (new `inputs.range_notes`; the data file keeps its labels): junction-offset range not containing
   the L-shape nominal, unranged y_junction/outer_side/N++/grading/contact width, a film range spanning both films,
   and a selected doping (3.2e18) above the research range that `doping.provenance` says contains it. The code now
   writes topology-relative ranges (`doping_ranges`), per-cross-section geometry ranges (as the modes artefacts),
   `xsec.unranged` and `doping.unranged`, and a corrected provenance string, for future runs.
5. Analytic-law handoff: "m_cap = 1 - m would understate the C-V exponent" was the wrong direction (it overstates);
   the stated cause (NSM sublinearity) is replaced by the evidence in the LUT; `dqi_dv_frac_per_v` was given at 1 V
   although the runtime treats it as the 0 V value (q would be 32 % low).
6. Sidecar `command` omitted `--doping-sweep` and the settings; it now gives the equivalent explicit command, and
   `command_as_run` the logged one. `main()` appends `--doping-sweep` for new runs.
7. The invalid Tidy3D `delta_n_eff` (-0.0177) stays in the sidecar for the record but is now flagged
   `delta_n_eff_valid: false` with a note.
8. Not fixed (needs a solver run or another owner): no slow re-run test for the L-shape; the bend-formulation fix;
   the missing artefacts; the g26 preset items under "For other agents".

New fast tests: `test_runtime_realize_in_lut_mode`, `test_free_order_view_and_forced_flags`,
`test_richardson_rejects_out_of_range_orders`, `test_assumption_range_notes_and_remesh_noise_are_recomputable`,
`test_report_identities_against_the_lut` (C_j, R_s, f_RC, nominal EO bandwidths, loss budget, and C = dQ/dV against
the removed charge, which holds to 2.5e-4).
