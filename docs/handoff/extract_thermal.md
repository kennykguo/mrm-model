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
