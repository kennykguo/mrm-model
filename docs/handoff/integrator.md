# Handoff: integrator, foundation stage

Scope: make the eight foundation modules (core/control API, params machinery, params content, validation targets,
plant optics, plant thermal + selfheat, plant electrical, plant sensing/datapath/laser, scenarios) fit together,
resolve the handoff requests that could be resolved now, and brief the engine owner
(docs/handoff/ENGINE_BRIEF.md). The CHANGELOG entry of 2026-10-09 lists every contract change.

## 1. Files written or changed

New:
- `mrm/plant/phase.py`: PlantParams + ring/heater temperatures + junction levels -> optics inputs (07 P1-5, P1-6,
  P1-7, P0-2, P0-3 R1). No leaf module did this; the kernel calls `ring_optics_row` per world.
- `tests/verification/test_phase.py` (6 tests), `tests/verification/test_plant_wiring.py` (9 tests plus one per
  shipped preset; its `ReferencePlant` is the worked example of the whole plant tick in Python).
- `docs/CHANGELOG.md`, `docs/handoff/ENGINE_BRIEF.md`, this file.

Changed in other owners' files (each change is small and keeps the module's design):
- `mrm/params/config.py`: artefact fallback (params_content request, section 1; exact patch as requested) and
  `ARTEFACT_FALLBACK_ALT`.
- `mrm/params/table.py`: `meta.generation_mix` and `meta.inventory` in PARAMETERS.md (params_content section 2).
- `mrm/params/schema.py`: R2 default `fsr_integral`; keys requested by plant_sensing (monitor baseline, stuck flag,
  flicker band floor, reference tap; readback noise and averaging; BER and bit-statistics hardware).
- `mrm/params/realize.py`: carries the new keys, the temperature-sensor `adc_bits`/`code_zero_c`, the lines'
  `source_laser`/`pol` and the `sim.*` kernel settings; absent dark-current activation -> 0.37 eV; per-world
  interim `*_um` values (rejects worlds that disagree).
- `mrm/params/derive.py`: docstring (R2 default).
- `mrm/params/presets/g26.yaml`: `heater_anchor: fsr_integral`, `heater_self_ratio` 1.158 (and the tin/textbook
  alternatives) to keep y = 0.00904 /mW, R2 notes, aggressor site renamed `aggressor`.
- `mrm/plant/electrical.py`: `HeaterDriverParams.from_plant_params`, `BiasDacParams.from_plant_params`,
  `junction_table_from_plant_params`; optional `code_add` (07 P1-10 dac_glitch after the DAC FIFO) on
  `heater_bank_step_row` / `heater_bank_step` / `HeaterBank.step`.
- `mrm/plant/sensing.py`, `mrm/plant/datapath.py`, `mrm/plant/laser.py`: the `from_plant_params` adapters read the
  new PlantParams fields (keyword arguments still override).
- `mrm/core/channels.py`: temperature-sensor field `code_zero_c` (was `code_offset_c`; params_machinery request).
- `pyproject.toml`: package data for `params/*.yaml` and the validation targets.
- `docs/ARCHITECTURE.md` (materials source, `plant/phase.py`), `docs/PARAMETERS_FORMAT.md` (section 10 fields, R2,
  regenerated key appendix), `docs/PARAMETERS.md` (regenerated).

## 2. Decisions (the contract left them to the integrator)

1. **R2 default = `fsr_integral`.** Physics: P_FSR is the heater power that moves the resonance by one FSR starting
   at `p_fsr_ref_temp_c`; the ring rise is Z P, so int_{T_meas}^{T_meas + Z P_FSR} s(T) dT = FSR is the exact
   inversion. `lock_slope` divides the sweep-average efficiency by the slope at T_lock (~132 degC, hotter than the
   sweep mean), which gives Z 4.4 % low at the G25 line (5.33 vs 5.58 K/mW) and makes Z follow each ring's sampled
   offset. Every agent that looked at it (params machinery, its reviewer, params content) recommended this. The
   droop fit y = TCR R_th,H = 0.00904 /mW is kept by lowering `heater_self_ratio` to 1.158. All preset tests
   (fast and slow, including the 512-world Monte Carlo of every corner) pass with it.
2. **Thermal transport delay** = extra whole ticks of heater DAC latency (rounded), in
   `HeaterDriverParams.from_plant_params`. It also delays the heater-resistor observer by <= 2 us, which the thermal
   owner judged negligible for the TCR loop (poles 3.2 and 13 us).
3. **Laser drift has one owner, the scenario.** The `laser_drift` scenario (linear worst case or random walk,
   calibrated to the printed 24 h limits) drives `laser.dfreq_hz` / `laser.power_delta_db`. `LaserHw` keeps its drift
   parameters as keyword arguments (default 0); no schema keys were added, so the two paths cannot both be on.
4. **Loss vs temperature** (`dln_alpha_dt_*`, 07 P1-5): multiplies the whole alpha_i, and the added loss heats
   (mrm/plant/phase.py). 07 P0-2 writes alpha_FCA (1 + c_T dT); the content agent computed the ER-mechanism
   alternatives with an exponential on alpha_i, which is what is implemented. The CAL-2 refit must use this form.
5. **Junction table per world** (analytic mode): `junction_table_from_plant_params` builds the full-circumference
   dneff(V) from eff_at_0v with the bank-mean lambda/n_g; a ring at another line then shifts by its own lambda/n_g
   (+-0.6 % across 2.8 THz). Per-length C and R_s use the world mean (C scales with each ring's length).
6. **NRZ level mapping**: level 1 (the ones density) = the higher reverse voltage of `nrz_levels_row` (assumed; only
   the sign of the OMA depends on it).
7. **Aggressor site name**: g26's aggressor site is `aggressor`, the library default, so `create('a1')` works.

## 3. Requests still open (for the owners named)

- **plant_thermal**: (a) a (B, N) `dc_scale` for the ROM heater paths (per-ring heater-efficiency spread is lost in
  ROM mode; the interim default is unaffected); (b) `ring.thermal.crosstalk_dynamics` as a per-world switch (the
  textbook corner would then be exact); (c) an optional absolute `ring.thermal.heater_self_k_per_mw` so the G25 droop
  does not follow Z across FSR / thermo-optic worlds (y = 0.0075-0.0078 /mW there instead of 0.00904); (d) the
  shared-lag state compression (handoff section 5) before the FEM ROM can meet 07 P0-9.
- **extraction (thermal_fem)**: at the start of the integration
  `tests/verification/test_thermal_rom.py::test_residues_well_conditioned` failed for `silicide_undercut_160` and
  `metal_above_160` (a driving-point residue of -8.9e-16, a rounded zero, against `>= 0`) while the extraction was
  rewriting those artefacts; on the artefacts present at the end it passes. Clamp exact zeros when saving, or test
  `>= -1e-12 * max|r|`, so a rewrite cannot bring it back.
- **core_control_api**: optional channels for the reference tap (`tx.monitor_ref.*`, `monitor_ref_valid`,
  `adc_ref_overrange`) and `datapath.bitstat_ones` (plant_sensing request); the plant can produce them
  (`MonitorHw.reference_tap`, `DatapathBank` `bs_n1`).
- **params_machinery**: tied multi-target draws (Q_L with the loss budget; continuous eta) and an `unset:` field for
  child presets (g26_rom / g26_lut); both are content requests.
- **calibration**: refit y (CAL-4) with the order-2 thermo-optic model under `fsr_integral`; refit the ER-mechanism
  alternatives on CAL-2 with the loss form of decision 4.
- **content**: `ring.junction.depletion_exponent` range [0.33, 0.67] reaches outside the analytic table's
  [1/3, 2/3] check by rounding (a corner at 0.33 or 0.67 raises in `analytic_junction_table`); use 0.334/0.666 or
  fractions. `ring.monitor.adc_bits` 16 vs ~10 and the full scale remain one joint hypothesis to write.

## 3b. Requests resolved here (by owner of the request)

- params_content: artefact fallback in the loader (section 1 of their handoff); inventory and generation mix in
  PARAMETERS.md (section 2); package data for `params/*.yaml`; the R2 decision (section 3 item 4).
- params_machinery: R2 default decided; PROVIDERS registry unchanged (no new CLI provider yet); package data.
- core_control_api: `ControllerSpec.from_config` runs on the real presets (g26, g25_test_vehicle, teaching_basic;
  g26: period 10 ticks, M = 3); temperature-sensor field renamed `code_zero_c`; the plant's released sensors match
  the declared channels (tests/verification/test_plant_wiring.py).
- plant_sensing: schema keys and PlantParams fields for the monitor baseline, stuck flag, flicker floor, reference
  tap, readback noise/averaging/bits, BER and sampler hardware, temperature-sensor bits and code zero, laser
  source/polarization; MonitorParams carries the mux order; absent E_a realizes to 0.37 eV; dac_glitch applied in
  the electrical FIFO path; the 07 P1-9 cross-module tests (drop vs absorbed power through the optics with the
  phase inputs; dark current vs the ROM monitor_pd observer). Not done: laser drift keys (decision 3).
- plant_electrical: the from_plant_params adapters (key mapping of their handoff section 1), DAC bit errors with the
  coarse-DAC width for dac_pwm worlds, transport delay as heater latency, dac_glitch interface.
- plant_thermal: realize() assembles the system (it already did; verified on every preset); transport delay owner.
- plant_optics: per-line K^2 and the phase rows are built (mrm/plant/phase.py); backscatter rows from realize.
- scenarios: the g26 aggressor site matches the library default; aggressor calibration recipe and signal wiring in
  ENGINE_BRIEF sections 1 and 3; the two laser-drift paths reduced to one.
- validation_targets: CHANGELOG line; package data for the targets.
- materials: ARCHITECTURE line and CHANGELOG note.

## 4. Physics notes found while wiring (please read before writing controllers or acceptance tests)

- **Start-up wrong-line capture in g26 nominal.** With the 7.6 nm blue offset, the 400 GHz (2.29 nm) grid and the
  10.3 nm FSR, every ring's cold resonance sits 0.73 nm (about 2 FWHM) BLUE of the line three channels up
  (line n+3 is 6.87 nm blue of line n). At +4 dBm per line and the nominal bistability ratio (3-7), a cold ring
  heats itself onto that foreign line when the lasers come on; in the sampled worlds of the wiring test two rings
  did so with the heaters off and their monitors clipped at the 40 uA full scale. This is the WDM wrong-line hazard
  of 07 P1-12, made sharper by the offset; the reset protocol (`laser_ramp`) and the supervisor must handle it.
- **Data-activity heat is not small.** At 50 Gb/s the junction CV^2 heat (~0.14 mW at Z_opt 9.9 K/mW) plus the
  assumed driver heat (2 mW static + 100 fJ per transition, ~4.5 mW per ring at 0.34 K/mW to its own ring) lift each
  ring ~2 K above a heater-only estimate. The driver numbers are assumed (no source); squelch and traffic on/off
  scenarios move rings by this much.
- The interim crosstalk poles are ms; after 400 us only the self path has settled (the wiring test checks
  0.97 (1 - tail) Z P <= rise <= 1.05 Z P).

## 5. Test status at hand-off

`.venv/bin/python tools/memguard.py --limit-gb 3 -- .venv/bin/python -m pytest -q -m "not slow"`: 1021 passed,
126 skipped (artefacts not extracted yet: junction LUTs, modes_rib160/coupling_rib160, VAL-8; replicas.py not
written), 0 failed, about 2 min. Slow preset/params tests (`-m slow` on tests/closure and test_params_realize): 17
passed. The new tests also pass with `NUMBA_DISABLE_JIT=1`. `mrm params validate` passes for all 15 presets.
