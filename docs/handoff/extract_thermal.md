# Handoff: extract_thermal (`mrm/extraction/thermal_fem.py`, `mrm/extraction/mor.py`)

Status: see "Results" below (filled from the sidecars when the runs finished).

## Files

- `mrm/extraction/thermal_fem.py`: axisymmetric P2 heat FEM (scikit-fem on gmsh meshes), coarse 3-D P1 model (single
  ring and 8-ring bank), exact layered analytic solution (independent check), femwell TE0 weights, extraction
  pipeline, stages, artefact I/O, CLI.
- `mrm/extraction/mor.py`: rational Krylov (Galerkin) ROM, modal Foster form, common real-pole vector fitting,
  variable-projection polish, DC-exact residue fit, error metrics.
- `mrm/data/extracted/thermal_rom_<family>_<film>.npz` + `.json` (schema `thermal_rom/1`), families
  `inring_noundercut` (default), `silicide_undercut`, `metal_above`, films 160 and 200 nm.
- `mrm/data/extracted/thermal_val8_imec.json` (schema `thermal_val8/1`, 07 VAL-8).
- `tests/verification/test_thermal_rom.py` (fast: artefact identities and recorded verification; slow: re-run),
  `tests/validation/test_thermal_replication.py` (VAL-8).

Re-run (each job through `tools/memguard.py --limit-gb 3 --heavy`):
`python -m mrm.extraction.thermal_fem [--family F] [--film N] [--stage extract|uncertainty|analytic|3d|val8]
[--quick] [--out DIR]`. `--quick` writes to a temporary directory unless `--out` is given.

## Artefact format (for `mrm/plant/thermal.py`; follows its handoff section 4)

`.npz` (no pickle): `taus_s` (K,) strictly increasing; `sources` = (heater, waveguide_absorption, junction, driver,
aggressor); `observers` = (ring_mode_weighted, heater_resistor, monitor_pd, temp_sensor); `self_res` (5, 4, K) K/W;
`cross_r_um` (D,) increasing, all > 2R; `cross_res` (5, 4, D, K) K/W; `site_offset_names` (n,) and
`site_offsets_um` (n, 2) for monitor_pd, temp_sensor, driver, aggressor; extras `dc_self_k_per_w` (5, 4),
`dc_cross_k_per_w` (5, 4, D) (exact FEM DC; the residues sum to them to 1e-12), `family`, `film_nm`, `schema`.

Sidecar `<stem>.json`: `schema`, `sha256`, `created_utc`, `family`, `film_nm`, `inputs` (every geometry assumption
with its provenance, the material values used, `materials_provenance()`, the femwell mode info), `tool_versions`,
`mesh` (model sizes and the convergence study), `rom` (fit details), `validation` (full-FEM errors; flat keys
`step_max_rel_err`, `freq_max_db_err`, `freq_max_deg_err`, `superposition_rel_err` as requested), `summary` (key
numbers), `rom_dc_scale` (07 P1-2 diagnostic table), `reciprocity`, `site_offsets_um` (dict), `k_temp_scaling` (07
P2-5, see below), `stages.uncertainty` (Jacobians, sweeps, G25 consistency), `stages.crosscheck_analytic`,
`stages.crosscheck_3d`.

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

## Results (all six artefacts)

Axisymmetric P2 FEM, production mesh 50 nm in the core (scale 1.0, ~46k DOF per model); convergence over four levels
(100, 71, 50, 35 nm) with Richardson extrapolation; Foster fit against the Krylov ROM (147 + 103 states, error
against exact solves < 4e-6 of DC); verification against the FULL FEM (exact complex solves at 16 frequencies, TR-BDF2
time stepping 0.25 ns .. 20 ms with a step-halving error estimate ~3e-5 of DC). Numbers are DC gains in K/mW and 63 %
times; ± is the 1-sigma geometry spread from the Jacobian propagation (assumed parameter sigmas, see
`stages.uncertainty.jacobian`).

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
response is under -60 dB and its phase is meaningless); far pairs (1.5-2 mm) reach 2.8 dB / 10.6° at their band
edge, < 1.3 % in time.

Physics read-outs:
- The film (160 vs 200 nm) changes the thermal ROM by < 1 %; the heater family changes it by 3x (DC) and 4x (tau).
- Self-heating path (07 P0-1): the waveguide-core step response reaches 10 % at ~10 ns, 27 % at 100 ns and 54 % at
  1 us (no-undercut families); 12 % / 25 % at 100 ns / 1 us with the undercut. The heater->ring response is 21 % at
  1 us and 81 % at 10 us (no undercut), 9 % / 42 % with the undercut. z_ratio = Z_opt/Z_heater = 1.32-1.76 (>= 1, as
  expected for an external heater). (Pole-by-pole DC shares are not unique between fits; quote step fractions.)
- Crosstalk at 50 um pitch: 0.7-1.0 % of the self DC, first-order-like with a 63 % time of 0.11-0.12 ms (the d^2 law
  of 04 gives 0.1-0.3 ms at 15-30 um; consistent). The long-range floor is the package: P x R_pkg = 0.002 K/mW
  (R_pkg 2 K/W for the assumed TEC-held package), i.e. 0.07 % of Z_heater->ring per ring; imec's 1.3-2.6 % floor was
  measured with natural convection and an EIC on top (much larger R_pkg). `sub_um` 100 um raises the 50 um crosstalk
  by 43 %; Dirichlet instead of the package Robin lowers it by 9 %.
- Boundary and substrate choices move Z_heater->ring and t63 by < 1 %; the BEOL metal fill by up to 6 % (in-ring) and
  38 % (undercut, where the BEOL carries the lateral heat).
- k(T) (07 P2-5, top-level `k_temp_scaling`): from 300 to 400 K Z_heater->ring falls 9 % (oxide k rises with T),
  t63 rises 19 % (rho c), crosstalk rises 38 % (substrate k falls).

## rom_dc_scale (07 P1-2) and the G25 efficiency

rom_dc_scale = (FSR/P_FSR)/(Z_FEM x dlambda/dT), P_FSR = 25.45 mW (G25 median). With FSR 1.8 THz (0.405 nm/mW):

| family | H86 (85.9 pm/K) | materials ceiling (70.6) | H_solver (modes artefact, ~65.8) | bench 62.5 |
|---|---|---|---|---|
| inring_noundercut | 1.65 | 2.0 | 2.1-2.2 | 2.3 |
| silicide_undercut | 0.58 | 0.71 | 0.76 | 0.80 |
| metal_above | 1.32 | 1.6 | 1.7 | 1.8 |

No family reproduces the G25 efficiency on the printed 45SPCLO stack: the no-undercut families are 1.3-2.3x too
weak, the GF-PDK-like undercut family 1.25-1.7x too strong. The `g25_consistency` sweep (heater gap, BEOL fill,
keep-out) reaches at most 4.2 K/mW in-ring and 4.4 K/mW metal-above (G25 needs 4.7 (H86) to 6.5 (bench) K/mW). The
G25 number therefore points to a partial undercut, a heater inside the ring's own doped slab (our `junction` source
location gives Z = 4.2-4.4 K/mW), or a thicker oxide than the printed 2 um BOX (dlnZ/dBOX = +0.36 per um). This is the
calibration lever (CAL-3): `ThermalScales.dc_scale` under "resistance" scaling also scales tau.

## Independent checks

- **Analytic (exact layered solution, `stages.crosscheck_analytic`).** `LayeredAnalytic` solves the laterally
  infinite multilayer exactly (Hankel transform in r, transmission-line admittances in z) for annular sources; it
  reproduces the half-space disk formula 8P/(3 pi^2 k a) to 0.08 %. Against the axisymmetric FEM on a structured,
  interface-aligned mesh (not gmsh), with the full stack (substrate, BOX, continuous 160 nm Si film, cladding,
  anisotropic BEOL, passivation), Dirichlet and Robin bottoms: self and heater-annulus responses agree to <= 0.14 %
  from DC to 1 MHz (0.43 % at 10 MHz); victim annuli at 20-100 um to 0.3-0.6 % at DC; every difference is below
  8.2e-4 of the self DC. Points where both responses are < 1e-5 of the self DC (far victims above 10 kHz) are
  quadrature noise and carry no information.
- **Full FEM vs ROM.** The Krylov ROM matches exact solves to < 4e-6 of DC; the final Foster model matches the
  time-stepped FEM within the contract (table above); the TR-BDF2 discretization error is ~3e-5 of DC.
- **Reciprocity** (07 P1-2): Z(heater -> ring weight) = Z(ring weight -> heater) to 8e-16 (`reciprocity`).

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

## Requests and notes, by consumer

1. **plant_thermal (`mrm/plant/thermal.py`)**
   - Format follows your section 4 (taus increasing, `site_offset_names`, flat validation keys, `.json` sidecar).
     `superposition_rel_err` is the 8-ring 3-D bank result of the default family, copied into every sidecar with a
     `superposition_note`.
   - `k_temp_scaling` is the top-level sidecar key you asked for (format above).
   - Residues: driving-point pairs (heater->heater_resistor, waveguide_absorption->ring_mode_weighted) are fitted
     non-negative; poles are kept >= 1.5x apart, so max |r_k| <= 8 |H(0)| on self pairs and <= 25 |H(0)| on cross
     pairs up to 300 um (`rom.residue_cancellation`). Collocated self pairs are therefore passive pole by pole; mixed
     signs occur only on non-collocated pairs, as the contract allows.
   - K = 20 to 24 poles (the fit ladder adds poles until the contract passes); every set spans ~1 ns .. 68 ms, with
     poles below 1 us and above 1 ms.
   - The package floor (die-level pole, P x R_pkg) is part of the ROM: a TEC/package base filter in the scenario must
     not double count it (it is 0.07 % of Z_heater->ring per ring at the assumed R_pkg = 2 K/W).
2. **params (schema, presets, realize)**
   - `ring.thermal.family` values: `inring_noundercut` (default), `silicide_undercut`, `metal_above`;
     `ring.thermal.rom_file`: `thermal_rom_<family>_<film>.npz`.
   - `uncertainty.thermal` ranges from the geometry Jacobians (1-sigma, ln): Z_heater->ring 0.16-0.29, tau_fast
     (t63 heater->ring) 0.24-0.48, self-heating tau 0.27, crosstalk at pitch 0.28; per-artefact
     `stages.uncertainty.uncertainty_thermal_suggestion` gives +/-2 sigma factors (in-ring: Z x0.63-1.60,
     time_scale_fast x0.55-1.83, crosstalk x0.57-1.74).
   - `layout.ring_pitch_um` default 50 um (assumed) and the site offsets (`site_offsets_um`) used for `self_res`:
     monitor_pd (0, 25), temp_sensor (40, 0), driver (0, -30), aggressor (60, 0) um. Any layout value can be used
     through the radial `cross_res` table instead (d >= 20 um).
   - `rom_dc_scale` is 0.58 to 2.3 depending on family and dlambda/dT hypothesis (table above): only the undercut
     family and the no-undercut families under H86 stay inside [0.5, 2].
3. **validation (VALIDATION.md)**: VAL-8a, 8b, 8c fail and 8d passes with the numbers above
   (`thermal_val8_imec.json`, `comparison`); report them as they are. rom_dc_scale per hypothesis for the closure
   table: `rom_dc_scale.table` in every sidecar (refreshed from the modes artefact by `--stage propagate`).
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
- The production level is the 50 nm mesh (scale 1.0); the 35 nm level is in the convergence study only (its
  Richardson estimate gives the production error, <= 0.29 %).
- Superposition: the 07 metric "error on the crosstalk residues" is replaced by the step-response error (max over
  10 ns-10 ms of |y_bank - y_ROM| / y_ROM(inf)); fitted residues of two near-identical smooth curves are not unique, so
  a residue-by-residue comparison measures the fit, not the physics.
- The heater families are axisymmetric equivalents (the GF PDK four silicide arcs are two full annuli; a 3-D check
  gives +1.0 % on Z_heater->ring for 140-degree arcs); heater leads are only in the 3-D model (-4.3 % on
  Z_heater->ring).
- VAL-8 uses our materials (validation-owner rule); imec's Table I set is a variant.
