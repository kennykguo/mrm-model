# Changelog

Contract changes (ARCHITECTURE v2 and the normative sections of docs/research/07_design_critique.md) and the
decisions that deviate from them. Newest first. Each entry names the reason and where the details are.

## 2026-10-09: engine performance round 1

Details, measurements and error bounds: docs/handoff/perf.md (owner: engine).

- **The tick runs in one function** (`mrm.sim.kernel._world_block`): `_prepare_interval`, `_thermal_and_sensors`,
  `_account`, `_auto_substep`, `_heater_step` and `_bias_step` are gone (their code moved into the world block in
  the same order); `truth_row(P, S, b)` stays as a wrapper of `_truth_a` (array arguments). Reason: a numba call
  that takes the KParams/KState tuples costs a reference-count update of every one of their ~270 arrays. No result
  changed (bit-identical, kernel/reference parity and the regression scenarios).
- **Self-heating adapter**: the exact ring without backscatter (g26) evaluates from substep caches (cos/sin of the
  reference phase, rotated; the Airy constants) in `mrm.plant.optics` (`sh_fast_precompute`, `sh_fast_pabs`,
  `sh_fast_outputs`, `sh_bounds`), used by both the engine and `optics.selfheat_pabs` / `selfheat_commit` /
  `selfheat_bounds`. Equal to `_airy_line` up to floating-point rounding (P_abs 2e-13 relative); the optics adapter
  now also fills the slope outputs (RL_DABS_DPHI, R_DABS_DLAM) for these rings. `pack_selfheat_args` layout: new
  blocks SH_O_FK, SH_O_FC, SH_O_FF, SH_O_LE, SH_O_FH and the flag SH_FARHOLD (SH_HDR 18 -> 24); new keyword
  `far_hold`.
- **Phase model** (`mrm.plant.phase`): `ring_optics_row` = `ring_optics_levels_row` + `coupling_row`; the engine
  recomputes K^2 at the lines only when the line wavelengths change (exact).
- `mrm.plant.electrical.heater_target_one` / `bias_target_one`, `mrm.plant.datapath.errors_tick_ber_row`
  (per-channel / given-BER forms of the row kernels; the engine caches the DAC transfer per code and computes the
  BER once per tick).
- `mrm.plant.selfheat.implicit_step_b` (and its kernel mirror) updates the lags in two passes (bit-identical).
- **New fidelity switch** `Plant(optics_far_hold=...)` / `run(..., optics_far_hold=...)`, default False: the
  far-line hold, a bounded approximation (perf.md P5); off by default because the closed loops turn its small P_abs
  error into other trajectories (perf.md 5.4). No schema key yet (params owner: `fidelity.optics_far_hold` if
  wanted). `run(..., sh_tol_k=...)` passes the self-heating tolerance through.
- Second session (perf.md P8-P13, all bit-identical to the first session's engine): the kernel's thermal pass from
  per-block tables (`kernel._th_struct`, `_th_gains`, `_heater_split_all`; `_implicit_step_eng` takes the tables as
  extra arguments); bus finalize without the full view set and without a division for a zero reflection; the junction
  table bracket found once per voltage from the uniform spacing (`mrm.plant.electrical.grid_index`, used by
  `_interp_lin`); the phase model without heap allocations (`mrm.plant.phase.ring_optics_levels_ws`,
  `resonance_shift_ws`, scratch passed in); `kernel.cold_coordinate` stops at an exact fixed point; Truth computes
  er_db / il_db only when a record reads them and at the last step of a block (`_truth_a(..., with_db, dark_known)`);
  the PD dark current once per tick (`mrm.plant.sensing.monitor_tick_row(..., i_dark_in=None)`, new optional
  argument; new state row `RF_I_DARK`).
- Engine accuracy against commit 8646691: every regression scenario and the controller scorecard agree to
  floating-point rounding (largest change 2.2e-6 pm of detuning in the scenarios, 2.6e-6 pm in the scorecard; no ADC
  code, heater code, lock state or line changed; perf.md section 5). B = 1 fixed tick 54-55 -> 15.5 us on AC power;
  07 P0-9's 10 us is not met (perf.md 5.6); its 64 x 1 s Monte Carlo in 5 min is (3.5-3.9 min).
- Review of round 1 (perf.md section 6): `mrm.plant.optics._fast_flag` derives a ring's Airy-path flag from its
  backscatter row at every adapter call (it was kept from the first call, so a caller that rewrote rb through
  `selfheat_views` afterwards ran the ring without its backscatter; the plant never does, results unchanged);
  the incremental cos/sin cache's error bound corrected for constant phase steps (systematic, <= 1.8e-12; ~5e-9 K).

## 2026-10-09: engine and controllers stage integration

Integrator changes (details in docs/handoff/integrator.md section 6 and docs/ENGINE.md):

- **Settled warm start (07 P0-1 item 6).** `Plant.settle()`, `Plant.reset(..., settle=True)` and
  `mrm.sim.runner.run(..., settle=True)` start from the thermal steady state of the warm-start codes and the inputs
  at t = 0 (the kernel run with the DC coefficients of every lag until the ring temperatures move < 2 mK), instead of
  cold lags. It needs reset protocol branch_low or branch_high (laser_ramp starts dark: ValueError).
  `run(..., heater_code='quasi_static')` (`runner.warm_start_codes`) takes the oracle lock codes of `quasi_static`.
  Requested by controllers_a (handoff 2.3), controllers_c (4.3b) and the engine (6a).
- **`run(..., factory_table=table)`** calls `controller.load_factory_cal(table)` before `controller.reset(spec)`
  (controllers_b handoff 2; engine handoff 6b). The runner does not run the factory procedure itself.
- **Baseline registry**: `mrm/control/baselines/__init__.py` imports every baseline module, so all 17 baselines
  resolve by name (controllers_a/b/c handoffs).
- **Schema keys** `metrics.lock_reference` (line | oracle | target, default line) and `metrics.lock_target_pm`
  (engine handoff 3 item 21); `MetricsSpec.from_config` reads both. PARAMETERS_FORMAT.md appendix regenerated.
- **g26 loss budget re-solved for the landed modes_rib160 artefact** (n_g 3.9405 instead of the 4.0 fallback,
  alpha_rad 0.119 instead of 1.0 dB/cm). With the old numbers the G25 notch fell to 26.5 dB (CAL-2: 27.9) and every
  eta hypothesis missed its label by 0.8 % (closure tests). Now alpha_i = 110.08 dB/cm keeps Q_i = 7503 (notch 27.9
  dB under-coupled), and eta = 0.848 at alpha_fca 80: `alpha_other_abs` 14.74 -> 13.35 (over-coupled alternative
  5.97 -> 4.71), `alpha_scatter` 16.0 -> 16.61, the eta_* alternatives of alpha_fca / alpha_scatter re-solved
  (fca + scatter = 96.61 in every eta world), the 200 nm `strip400x200_q_anchor` 10.37 -> 11.09 dB/cm. Recompute
  when modes_rib160 changes (the notes in g26.yaml give the recipe). PARAMETERS.md regenerated.
- tests/validation/test_replicas_static.py::test_other_lines_match_the_control_models_static_point: the per-ring
  bound "other lines add > 3 % heat" became "> 2 % per ring and > 8 % on the first rings" (measured 2.9 % on ring 7,
  last on the bus, whose foreign line was partly dropped by the ring that owns it; 9 % on rings 0-2).
- pytest marker `perf` registered (engine handoff).
- **docs/ENGINE.md** replaces docs/handoff/ENGINE_BRIEF.md as the engine reference (the brief is kept for its
  leaf-signature tables).

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
