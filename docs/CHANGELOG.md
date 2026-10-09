# Changelog

Contract changes (ARCHITECTURE v2 and the normative sections of docs/research/07_design_critique.md) and the
decisions that deviate from them. Newest first. Each entry names the reason and where the details are.

## 2026-10-09: foundation-stage integration

Integrator decisions (details in docs/handoff/integrator.md):

- **R2 heater form: `fidelity.heater_anchor` default is now `fsr_integral`** (schema default and g26), not 07 P0-3's
  `lock_slope`. Z_heater->ring = dT_FSR/P_FSR with the integral of s(T) over the P_FSR sweep equal to one FSR is
  the exact inversion of the anchor measurement (07 D16). `lock_slope` puts Z 4-5 % low at beta = 1.4e-3 /K and
  makes a stack property depend on each ring's sampled resonance offset. `lock_slope` stays selectable. To keep the
  G25 droop fit y = 0.00904 /mW (CAL-4), `ring.thermal.heater_self_ratio` moved 1.21 -> 1.158 (in-ring), and its
  alternatives tin 3.38 -> 3.243, textbook 1.69 -> 1.621 (silicide stays at its 1.0 floor; its y is now 23 % high).
- **New plant module `mrm/plant/phase.py`** (ARCHITECTURE section 2): PlantParams + ring/heater temperatures +
  junction levels -> the optics inputs (phase, K^2 at the line, a, eta, t_rt). No leaf module did this.
- **Loss temperature coefficients** (`dln_alpha_dt_ring_per_k`, `dln_alpha_dt_heater_per_k`, 07 P1-5) multiply the
  whole alpha_i (the key is d ln alpha/dT) and the added loss heats (mrm/plant/phase.py). 07 P0-2 writes
  alpha_FCA (1 + c_T dT); the calibration owner must use the same form for the CAL-2 refit.
- **Thermal transport delay** (07 P1-2, 4.8: 0-2 us) is applied as whole extra ticks of heater DAC latency
  (`HeaterDriverParams.from_plant_params`, rounded to the nearest tick), as the thermal owner proposed.
- **Laser drift has one owner: the scenario** (`laser_drift` scenario on `laser.dfreq_hz` / `laser.power_delta_db`,
  07 P1-4 path 4). The drift parameters of `mrm.plant.laser.LaserHw` stay keyword-only (default 0) and get no
  schema keys, so drift cannot be applied twice.
- **g26 aggressor site renamed `agg0` -> `aggressor`**, the default site name of `mrm.scenarios.library` (a1, a2,
  aggressor_direct), so `create('a1')` runs on g26 without arguments.
- Artefact fallback: a missing `from:` artefact FILE falls back to the leaf's `fallback` alternative with a warning
  (`mrm.params.config.ARTEFACT_FALLBACK_ALT`; params_content request). A bad sha256, a missing sidecar or key stays
  an error.
- Schema keys added (plant_sensing requests): `ring.monitor.{baseline_ua, stuck_flagged, flicker_band_min_hz,
  reference_tap.{present, coupling_frac, responsivity_a_per_w, dark_i0_na}}`, `ring.heater.readback.{noise_rms_codes,
  average}`, `datapath.{rx_sensitivity_oma_dbm, ber_ref, link_loss_db, ber_floor, bitstat_bw_ghz,
  bitstat_noise_pa_per_rt_hz}`. PlantParams carries them, the temperature-sensor `adc_bits`/`code_zero_c`, the
  lines' `source_laser`/`pol`, and the `sim.*` settings the kernel needs (TimingParams). An absent
  `ring.monitor.dark_activation_ev` now realizes to 0.37 eV (mrm.materials), not 0.
- The 07 P1-10 `dac_glitch` fault is applied to the heater code leaving the DAC latency FIFO
  (`heater_bank_step_row(..., code_add)`); the register readback does not show it.
- `mrm.core.channels`: the temperature-sensor table field is `code_zero_c` (schema name), not `code_offset_c`.
- `realize()`: per-world values of `ring.thermal.xt_dist_ref_um`, `xt_far_um`, `site_ref_um` are used (they were read
  from the config and a hypothesis that set them was silently ignored); worlds that disagree raise.
- `pyproject.toml` package data: `params/*.yaml` (sources.yaml), `validation/targets/*.yaml` and its README.

Requested by other owners (recorded here as asked):

- 2026-10-09 re-freeze of validation.yaml and bounds.yaml by the review of the target split, before any comparison;
  no validation value or tolerance changed; BND-1 windows corrected, BND-2 margin text corrected, BND-3 split into
  BND-3a/3b; reasons in the file headers and docs/handoff/validation_targets.md.
- 07 P0-9 "parabolic segments" (slide-10 aggressor) replaced by the rectified sine (01 verification log item 5);
  scenario kernels compute in ticks; time-compression reference rate 2127 K/s (docs/handoff/scenarios.md).
- Material coefficients are read from mrm/data/materials.json, the cleaned copy of docs/research/02_materials.json
  with the verification-log corrections; 07 P1-3's thin-film k (56-66 W/(m K), +-25 %) is superseded by
  `mrm.materials.k_si_film_w_per_m_k` (02 log #14; docs/handoff/materials.md).
- FSR window A is 1.47 THz, not 07's 1.32-1.48 THz interval (its interior puts a second order within 2 FWHM of a
  TX line; BND-1 review; docs/handoff/params_content.md section 5).
