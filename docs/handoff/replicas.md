# Handoff: replicas (`mrm/validation/replicas.py`)

Owner files: `mrm/validation/replicas.py`, `mrm/validation/__init__.py` (re-exports), `tests/validation/test_replicas_static.py`,
the generated `mrm/validation/latest_predictions.json`, and this file. Contract: 07 P0-5 and section 4 (normative), P1-4
(apparent sensitivity), P1-7 (bias ladder at power), P1-9, P1-11; the target files and `mrm/validation/targets/README.md`.
No parameter was tuned to any target. Nothing outside these files was edited.

Built by the replicas builder; reviewed adversarially on 2026-10-09 (section 8 lists what the review changed). Where this
file and the code disagree, the code's docstrings are current.

## 1. What is there

Every name of `targets_io.REPLICA_NAMES` (22) is implemented and registered in `replicas.REPLICAS`. Signature of every
replica:

```python
fn(plant, target: Target | None = None, **kw) -> ReplicaResult
# plant: preset name | Config | PlantParams; target.replica_args first, keyword arguments override (an explicit None is kept)
@dataclass ReplicaResult: replica: str, value (shaped for targets_io.evaluate, in target.unit; None if blocked), unit: str,
    status: "ok" | "blocked" | "error", target_id, conditions: dict, variants: {label: value}, detail: dict, notes: list
```

Optical-power convention (every replica): `p_in_dbm=None` means P_in -> 0 (no light, no self-heating);
`p_in_dbm="preset"` means the preset's line power at the ring; a number is dBm. Defaults: the G25 bench sweeps P_in -> 0
(power not printed); monitor, bistability and apparent-sensitivity protocols "preset"; relOMA maps the target's printed
power (0 dBm), "preset" when a target does not print it.

Other public API:

```python
resolve_plant(plant, *, batch=1, uncertainty="nominal", variation="off", seeds=0, alts=None, overrides=None) -> (cfg|None, pp)
load_config(plant, *, alts=None, overrides=None) -> Config          # re-loads a Config with extra alts/overrides
preset_for(target) -> str                                           # G25 -> g25_test_vehicle, else g26
run_target(target, plant=None, **kw) -> ReplicaResult               # never raises: blocked / error carry the reason in notes
predict_all(roles=("calibration", "validation", "bounds"), *, ids=None, plant_by_id=None, verbose=False) -> list[dict]
write_predictions(path=None, records=None, **kw) -> Path            # default mrm/validation/latest_predictions.json
default_run_fn() -> Callable[[LockRunRequest], Mapping]              # mrm.sim.runner.run_lock_request, else ReplicaBlocked
StaticRing(pp, *, b=0, n=0, topology="add_drop"|"all_pass", driver_law=None|"voltage"|...|"bench_voltage")
    .optics(lam (L,), t_ring (M,), t_heater=None, v_lo=0., v_hi=None) -> dict of (M, L) per-unit-power fractions
    .state_grid(lam, t_grid, *, t_base, p_in_w, v_lo, v_hi, rho, p_j, p_d, agg_r, agg_h, lines="own"|"all",
                p_lines_w=None) -> + p_abs, p_h, t_h, u, t_pd, drop_avg, thru_avg, reloma, drop_other_w
    .other_lines_terms(t_grid, t_heater=None, v_lo, v_hi, rho, p_lines_w=None) -> (abs W (M,), drop W (M,))
    .lock(*, mode="half_max"|"fraction"|"peak"|"oracle"|"resonance", t_base, p_in_w, bias_v, modulate,
          order=0|"first", lines="own"|"all", p_lines_w=None) -> LockPoint
    .t_res(lam, v=0., k=0), .first_order(lam, v, t_floor), .order_offset_m(k), .u_from_power, .power_from_u (TCR and
    driver compliance), .compliance_ok(p_w, t_h), .code_to_u / .u_to_code (realized DAC), .nrz_levels, .data_heat,
    .monitor_current_a(p_drop_w, p_bus_w=0., t_pd_k=None), .junction_shift_m, .thermal_shift_m, .s_at, .linewidth_k
    attributes: lam_line, p_line_w, line, lam_lines (L,), p_lines_w (L,) chip-input, other_lines, v_comp, z (DC gains)
LockPoint: mode, order, t_ring_k, t_res_k, p_abs_w, p_drop_w (into the monitor PD), p_thru_w, reloma, drop_peak_w, p_j_w,
    p_d_w, p_heater_w, t_heater_k, u, stable, detuning_m, t_pd_k, p_drop_other_w, p_bus_w
LockRunRequest(cfg, pp, scenario, controller, setpoint_a (B,N), initial_code (B,N), duration_s, record_every_s,
               record=("monitor_a", "heater_code", "t_ring_k", "t_base_k"), options={...})   # section 6, engine
bank_gains(pp, b) -> {"<s><o>": (N, N) K/W}  s in h o j d (heater, absorption, junction, driver), o in r h p (ring,
               heater resistor, monitor PD); [i, j] = source of ring j -> observer of ring i
bus_line_powers(pp, b, t_ring_k (N,), *, bias_v=None, v_lo=None, v_hi=None, rho=0.5, p_chip_w=None) -> (N, L) W
LM_DLAMBDA_DT_M_PER_K = 85.86e-12
```

`python -m mrm.validation.replicas [--ids CAL-1 ...] [--out PATH] [--list]` regenerates the predictions file (about
2 min, peak RSS 0.6 GB; run it through `tools/memguard.py --limit-gb 3`). `mrm.validation` re-exports the target API
eagerly and the replica API lazily (PEP 562), so `import mrm.validation.targets_io` (calibration) does not import the
plant.

## 2. How the protocols are reproduced

Static core (`StaticRing`): the state variable is the mode-weighted ring temperature T; for each T the optics come from
`phase.ring_optics_row` + `optics.ring_exact`, the heat balance from the DC gains of `pp.thermal.system`
(`thermal.dc_gain`), the heater from the driver law with the TCR and the driver compliance (`electrical`), so the drive
needed is explicit, u(T). A fixed drive u0 has its steady states where u(T) = u0; dT/dt has the sign of u0 - u(T), so
slow sweeps are followed by exact quasi-static branch continuation (`_relax`). Locks choose T directly (ideal settled
controller).

Bank protocols (`_bank_codes`: CAL-9, VAL-6, BND-2, the CAL-8 primary, the dynamic warm start): every TX line is on at
its chip-input power; the rings are locked in `layout.bus_order` and each ring sees the lines as the rings before it
left them (single forward pass through their level-averaged through transmissions, plus the bus segment loss, as the
plant's bus walk); each ring's own line sets its lock, the other lines heat it and leak into its drop monitor (07 P1-9:
"the drop sums every line and order"); the heater powers come from the N x N bank heat balance (heater, absorption,
junction and driver crosstalk), the PD temperatures from the same gains. In g26 the other lines add ~9 % to each
ring's absorbed heat and ~8 % to its drop at the lock, mostly through the ring's other order 2.6 FWHM from line n+4
(BND-1); ignoring them biased the codes by ~300 (1 %) and CAL-8 by ~1 uA.

Independent cross-checks (tests): the static state is a fixed point of `ReferencePlant` (0.05 K, 0.5 % heater power,
300 us); the bank locks equal `mrm.sim.runner.quasi_static` (the engine's own static survey: 0.02 K, 20 uW, 1 % codes,
0.02 dB relOMA; compared with `lines="own", bus_filter=False`, see 6); the line powers, absorbed heat and oracle codes
equal `mrm.control.models.static_point` (1 %, 0.5 %); the EO response equals a direct TCMT ODE (1 %) and
`mrm.waveform.eo_bandwidth` (0.5 % with the loss modulation off).

| replica | targets | protocol (assumptions in `conditions`) | primary / variants |
|---|---|---|---|
| `loaded_q_from_3db_width` | CAL-1 | low-power sweep, heater 0 V, bias 0 V, stage at T_ref; Q = lambda_min / full width at half depth | plant topology / all_pass, width 3 dB below max |
| `notch_er_vs_heater_v` | CAL-2, VAL-2a/b | low-power sweep at each bench heater voltage (TCR solved); ER = max/min over one FSR | plant topology (add-drop) / all_pass = drop coupler removed, rest unchanged |
| `heater_power_per_fsr` | CAL-3a/b | bench voltage sweep until order m reaches the cold position of order m-1, from 25 degC | median over realized worlds/rings (default: the nominal device) |
| `resonance_vs_heater_v`, `tuning_range_at_v` | CAL-4, VAL-1 | through minimum of a low-power sweep vs bench voltage, relative to 0 V | - |
| `resonance_shift_vs_bias` | CAL-5 | through minimum vs reverse bias, heater 0 V; at power on a red sweep (thermal triangle) | +4 dBm (measurement power, relief included, 07 P1-7) / P_in -> 0 (relief removed) |
| `cold_resonance_offset` | CAL-7 | line minus the low-power resonance at 0 V | / dip of a red sweep at the preset power |
| `apparent_shift_fixed_code` | CAL-6 | fixed integer code (cold resonance on the line at 25 degC), ambient or on-die-aggressor sweep 25 -> 105 degC, laser parked at the line at the preset power (self-heating feedback), TCR for each law; reported T = stage, or base + reference_rise_frac x ring rise (H86_sensor) | plant/ambient/own law / every hypothesis x heating x law |
| `relOMA_map` | CAL-14, VAL-4a/b/c, VAL-5, BND-4 | one MRM (its own line; relOMA = \|T1 - T0\| at its through port, "1 mW into the MRM", 07 D9), NRZ at the datapath swing, 0 dBm; oracle = max relOMA over fixed-drive-stable, deliverable heater states (P_h >= 0, u <= 1, compliance); fixed_code = one drive aligned so the swept peak sits at 69.6 degC, up-sweep in ambient, 5 degC samples, linear in dB; the hysteresis of the fixed-code line is reported | oracle / setpoint_lock (half max), preset power, sweep direction, contour relative to the map's own low-bias level |
| `monitor_current_at_lock` | CAL-8 | bank locked at the set-point (all lines on, bus-filtered), NRZ at the scenario bias, preset power, base 25 degC; MonitorHw photocurrent (signal of every line + dark at the PD temperature); set-point levels on the monitor's total drop | half_max / peak, oracle, lines=own; `method="dynamic"` through run_fn |
| `stabilizer_code_vs_T` | CAL-9a/b/c, VAL-6 | 8 sampled dies (variation on, uncertainty nominal); A1 calibrated on each die's ROM (resonance referencing, ring_t0 = lock T, ENGINE_BRIEF), periodic steady-state rises of every ring and heater-resistor observer by exact FOH simulation; bank locked (half max, all lines); law + TCR + compliance + realized DAC; each ring keeps the order a heater-only ramp acquires at t = 0 of the period (the cusp, the hottest aggressor state) | median over dies / each driver law, full-scale range ends, aggressor_model=uniform, variation=off (nominal die); VAL-6: ambient 0 -> 105 degC excursion |
| `bistability_sweep` | (verification) | slow heater sweep up and down at power, laser parked at the line; jump drives, hysteresis in u, codes, mW | - |
| `eo_bandwidth_at_bias` | CAL-10, VAL-3a/b | waveform mode (`mrm.waveform.eo_bandwidth`): TCMT linearization with index and loss modulation, C_j(V), R_s, through the plant's driver output resistance (50 ohm); laser re-centred at each bias | x = 1/sqrt(3) (max slope) / hwhm, r_drv 0, loss modulation off, the earlier linearized index-only R_source 0 model |
| `pd_responsivity` | CAL-11 | [min, max] over 256 sampled worlds x rings | - |
| `laser_line_statistics` | CAL-12a-d | static error = line - grid over 256 sampled worlds; drift from the `laser_drift` scenario rendered every 60 s over 24 h | linear drift (default) |
| `aggressor_trajectory` | CAL-13a/b | the scenario's A1/A2 calibrated on the ROM, then an independent FOH simulation of the victim's Foster response, last of 4 periods, t = 0 at the cusp; y = 25 degC + model resonance shift / 85.86 pm/K | A2: delta_t_per_period + symmetric triangle / other readings |
| `adjacent_order_penalty` | BND-1 | rigid comb from the cold phase curve, resonance on its own line; distance of every other order inside the TX span +- 2 FWHM to the nearest other TX line, in that order's TCMT FWHM | nominal world / min over 32 sampled worlds |
| `order_hop_margin` | BND-2 | bank locked (oracle, all lines) at a 105 degC base, bias 3.75 V, NRZ; order acquired at 25 degC; margin = shift still provided by each ring's own heater | worst lane / lock half_max, peak, 8 sampled dies |
| `mode_solve_published_geometry` | VAL-7a/b | reads `modes_decea2019.json` (`replication.model_nominal`) | n_g at the 10 um outer radius / centreline; energy fraction / Robinson Gamma |
| `self_heating_time_constant` | VAL-7c | reads `thermal_val7_decea.json` | blocked: no de Cea thermal stack exists |
| `thermal_step_published_stack` | VAL-8a-e | reads `thermal_val8_imec.json` (`comparison[id].predicted`; VAL-8e `{tau_us, share}`) | - |
| `aggressor_feasibility` | BND-3a/b | listed controllers through run_fn (rate and latency limits in `options`), bank warm start, last aggressor period; signed % band or uA band of the victim; best = smallest miss of the target band, then narrowest | blocked until the controllers are registered and the engine provides run_lock_request |

## 3. Predictions on the uncalibrated presets (snapshot, 2026-10-09 ~13:30, after the review)

Artefact state: `modes_rib160.json`, `modes_rib200.json`, `modes_decea2019.json`, `thermal_val8_imec.json` present;
`coupling_rib160.json` and the junction LUTs absent (two fallbacks, recorded per record in
`conditions.artefact_fallbacks`). **The numbers change when artefacts land**: regenerate with
`python -m mrm.validation.replicas` after every artefact or preset change. Totals: 41 ok, 3 blocked (VAL-7c, BND-3a/b),
0 error; `evaluate` (information only) passes CAL-1, 3a, 3b, 7, 11, 12a, 12d, 13a, 13b, VAL-3b, VAL-8d, BND-1, BND-2,
BND-4. A NaN prediction is written as null with `predicted_nonfinite: true` (VAL-4a/b).

| id | predicted | target | note |
|---|---|---|---|
| CAL-1 | 3856 | [3500, 4200] | = realized Q_L to TCMT accuracy |
| CAL-2 | 26.5, 26.9, 29.5, 38.3, 43.9, 31.0 dB | 27.9 ... 19.8 | dispersion-only mechanism crosses critical coupling (07 P1-5; params handoff) |
| CAL-3a/b | 25.39 mW / 1.192 V | 25.45 / 1.195 | R2 anchor with the FSR measured at the cold resonance (-0.2 %) |
| CAL-4, VAL-1 | 13.80 nm at 1.4 V, 15.63 at 1.5 V | 11.59, 13.0 | median-device efficiency; the swept device is 0.348 vs 0.405 nm/mW (06 E4) |
| CAL-5 | 41.4, 57.9, 72.6 pm (+4 dBm) | 56.4, 66.8, 84.4 | P_in -> 0 variant 48.9, 68.8, 87.0 (pure junction); dip read on a 0.1 pm sweep |
| CAL-6 | 66.5 pm/K (H_solver, ambient, voltage) | 85.86 (not scored) | current 87.6; H86 74.3; H86_sensor via aggressor 83.1 (voltage) to 109.5 (current) |
| CAL-8 | 18.8 uA (half max, all lines) | 15.5 | own line only 17.8; peak 37.6; oracle 28.9 |
| CAL-9a/b/c | swing [3.6k, 22.8k]; offsets 32.3k / 38.1k; range [2.9k, 49.8k] | [19.7k, 22.3k]; 7.5k / 9.5k; [19.4k, 49.25k] | see 4.2 |
| CAL-10, VAL-3a/b | 45.5, 47.3 GHz, ratio 1.040 | 30.9, 35.4, 1.09 | waveform mode; photon-lifetime limited (RC pole ~100 GHz at R_s 100 + R_drv 50 ohm, 10 fF); see 8 |
| CAL-12b | median 0.27, q25 -4.6, q75 5.1 GHz | -2, -5.5, 1.5 | preset variation is zero-mean symmetric |
| CAL-12c | 0.0203 nm | <= 0.02 | see 6 (scenarios) |
| CAL-13a/b | rms residual 0.015 K; 801 K/s | 1.0 K; 808 | scenario calibrations: small residuals by construction |
| CAL-14 | -6.17 dB | -2.1 | swing x efficiency / FWHM not calibrated (06 M6: G25-like swing gives about -6.6 dB) |
| VAL-2a/b | 25.5, 23.6 dB | 18.0, 16.9 | follows CAL-2 |
| VAL-4a/b | nan (contour not reached: map -6.1 ... -9.3 dB) | 3.15, 3.4 V | variant relative to the map's own low-bias level: 2.2 V at both temperatures |
| VAL-4c | 3.25 dB | 0.9 | |
| VAL-5 | below 2.9/7.1 K, above 10.6/18.7 K | 7.6/12.6, 4.9/9.4 | wrong asymmetry; the line is bistable at 0 dBm (4.1) |
| VAL-7a/b | n_g 3.05 (outer radius); Si energy fraction 0.83 | 2.73 (unverified); 0.65 | centreline n_g 3.34; Robinson Gamma 0.77 (definition conflict, 6) |
| VAL-8a-e | 1.13 us, 3.12 us, 291, 676 pm/mW, poles 3.5/32.7 us (0.87/0.13) | 5, 10, 222, 843, 3.18/13.0 (0.43/0.57) | extraction artefact read as is |
| BND-1 | 2.62 FWHM | >= 2 | sampled worlds min 0.82 (stress FSR hypotheses) |
| BND-2 | 0.28 nm | >= 0 | half_max 0.74, peak -0.17; 8 sampled dies: worst lane -6.8 nm (device variation, 4.3) |
| BND-4 | 0.0 dB | <= 1 | nominal er_mechanism has no heater-T loss term (07 4.5: only that term can drift) |

## 4. Findings the predictions reveal (for calibration, params, scenarios; nothing was tuned here)

1. **Loop-open line asymmetry and bistability (VAL-5).** At a fixed code the self-heating steepens the cold side (laser
   red of the resonance, positive feedback) and widens the hot side. At 0 dBm the g26 fixed-code line is BISTABLE: the
   up-sweep jumps over the cold flank between 66 and 68 degC (the branches differ by up to 7 K of ring temperature;
   `detail.primary.hysteresis_c`). The 5 degC samples (65, 70 degC) straddle the window, so both sweep directions give
   the same sampled offsets. (The builder's note "single-valued, no hysteresis" was wrong; corrected at review.) The
   published map has the cold side wider; neither Q, eta nor the thermo-optic curvature in the current world reverses
   it; a calibration that matches VAL-5 needs another mechanism (e.g. a heater-temperature loss term or a different
   reading of the map) and must not use VAL-5 to choose it. At P_in -> 0 the line is single-valued and wider on the
   cold side than at 0 dBm.
2. **Slide-10 lane uniformity (CAL-9).** The g26 aggressor site sits 30 um from ring 0: under resonance referencing
   it heats ring 0 by ~95 K and ring 7 by ~21 K, so the nominal die (no variation) gives lane offsets of 1.0k codes at
   the code maxima and 32.5k at the minima, and ring 0 needs negative heater power at the cusp (code 0, infeasible).
   The slide shows ~20k swings on all 8 lanes, i.e. an aggressor that heats every lane alike (variant
   `aggressor_model=uniform`). The aggressor site/geometry (scenarios, layout) is the first suspect, then H_solver vs
   H86 (the A1 target at 85.86 pm/K needs 1.2x more ring rise under H_solver; scenarios handoff section 8).
3. **Device variation vs the 7.6 nm offset (BND-2, CAL-9b).** With the preset's resonance variation (07 P1-16 Berkeley
   6 nm die / 2 nm ring, and the rib160 Jacobian), many sampled rings acquire a different order or need an order hop
   between 25 and 105 degC (worst lane -6.8 nm), and lane offsets reach ~30k codes. No barrel-shift remap (PAT-1, 07
   P1-12) is modelled in the replica: each ring keeps the layout's line.
4. **Apparent sensitivity (CAL-6).** At a fixed code the TCR alone moves the reading by -16 % (voltage) to +10 %
   (current) around the ring's own s(T) (~79 pm/K at the ~130 degC lock temperature, not the 66 pm/K at 25 degC);
   H86_sensor reproduces ~86 pm/K only through the on-die aggressor path with voltage drive (83.1), not through an
   ambient sweep (66.5). The self-heating feedback with the laser parked at the line lowers the reading a little more.
5. **EO bandwidth** is photon-lifetime limited with the assumed R_s C (RC pole ~100 GHz); CAL-10 will move R_s or C by
   a factor ~3, which also decides VAL-3b (an RC-limited junction gives ~1.2, the photon-lifetime limit ~1.03-1.04).
6. **CAL-3 definition.** R2 anchors P_FSR with the FSR at the line wavelength; the bench protocol measures the FSR
   between the cold order and the next order red of it. They differ by the lambda^2 scaling of the FSR (0.2-0.4 %
   here): inside the 2 % tolerance, recorded in the result's notes.
7. **relOMA definitions differ between modules by 0.35 dB.** The replica's relOMA is one MRM's through port over the
   power into it (07 D9 "1 mW into the MRM"): -6.14 dB at g26 nominal, 1.05 V. `mrm.validation.oracle` /
   `mrm.control.models` report OMA at the BUS OUTPUT (after the other seven rings) over the power at the ring:
   -6.49 dB at the same point. CAL-14 and VAL-4c have 0.3 dB tolerances: the report must use one definition.

## 5. Decisions and deviations (please review)

- **Primary hypotheses** were fixed before comparisons: CAL-5 at the measurement power (07 P1-7: the protocol measured
  at power; +4 dBm is the G25 printed input), CAL-8 at half maximum, VAL-5 up-sweep, EO at the maximum-slope
  detuning. The VAL-4 "relative to the map's own low-bias level" variant was ADDED after seeing that the absolute -3 dB
  contour is not reached; it is a variant only and is labelled as such.
- **Changed at review after comparisons were visible** (section 8): the EO method (now the README's waveform mode with
  the plant's 50 ohm driver; VAL-3b moved from 1.025 (fail) to 1.040 (pass by 0.01)) and CAL-8 / CAL-9 / BND-2 /
  VAL-6 (all lines on, bus filtering: CAL-8 moved away from its target, 17.8 -> 18.8 uA). Both changes follow the
  protocol text or the plant's own physics, not the targets; the earlier values are kept as variants and the report
  must flag VAL-3b as "method changed after the comparison".
- **Order acquisition** (`StaticRing.first_order`): bank locks hold the order a heater-only ramp from code 0 acquires
  (PAT-3), chosen at the start of the protocol and held; it was added after seeing sampled dies demand negative heater
  power. It is a protocol rule, not a parameter change; the nominal world is unaffected.
- **Single-MRM protocols** (G25 bench, relOMA maps, bistability, apparent sensitivity) use the ring's own line only;
  bank protocols use every line (section 2).
- **Quasi-static closed loop**: CAL-8, CAL-9, BND-2 and the loop-closed maps assume a settled loop (the 100-131 ms
  aggressor periods are slow next to any loop above ~1 kHz); the thermal path of the aggressor itself is simulated
  dynamically (FOH, periodic steady state), the heater response is DC. The dynamic path exists (`method="dynamic"`)
  and is tested through a reference-plant adapter (test-only; it reads the true photocurrent, 07 P1-1).
- **all_pass topology** = drop coupler removed, other parameters unchanged (at kappa_d/kappa 0.007 the two differ by
  < 1 % in Q_L).
- **VAL-7b definition conflict**: the frozen target text asks for "an energy or power fraction in Si, not dn_eff/dn_Si";
  the extraction owner reads de Cea's Gamma as Robinson's dn_eff/dn_Si. The replica's primary follows the frozen text
  (energy fraction 0.83), the Robinson value (0.77) is a variant; both miss 0.65 by more than the 0.05 tolerance.

## 6. Requests to other owners

- **engine (mrm/sim/runner.py)**: (a) provide `run_lock_request(request: LockRunRequest) -> {"t_s": (T,), "monitor_a":
  (T,B,N), "heater_code": (T,B,N), "t_ring_k": (T,B,N)[, "setpoint_a"]}`: run `request.scenario` (calibrate pending
  aggressors, ENGINE_BRIEF 1) plus `tx.bias_v = options["bias_v"]`, controller `request.controller` with the per-ring
  monitor set-points `setpoint_a` (B, N), warm start from `initial_code` (B, N) and `options["warm_state"]`
  ({t_ring_k, p_abs_w, p_heater_w, p_j_w, p_d_w} (B, N) and t_base_k: the bank's quasi-static lock with every line;
  for the PLANT only, never the controller), honour `options["max_update_rate_hz"]` / `["min_total_latency_s"]`
  (BND-3), record at `record_every_s` (= max(dt, duration/2e5)); return "setpoint_a" when the controller sets its own.
  `ReferenceLockRunner` in tests/validation/test_replicas_static.py is a working N = 1 example on ReferencePlant. Until
  then `default_run_fn` raises ReplicaBlocked and BND-3a/b stay blocked. (b) **`quasi_static` bus filtering**: its
  docstring says each ring gets "the chip power times the through transmission of the rings before it", but the loop
  multiplies only `p_ring[l]` of the ring's OWN line after that ring, so no ring ever sees upstream filtering, and each
  ring absorbs its own line only. In g26 that misses ~1 % of heater power on the downstream rings and ~9 % of the
  absorbed heat (other lines). The replica test compares against it with `lines="own", bus_filter=False`.
- **controllers**: BND-3a/b name `drop_setpoint_pi`, `bias_dither_lockin`, `dual_actuator_midrange`; the registered
  names are `pi_lock`, `bias_dither`, ... and `mrm/control/baselines/__init__.py` is empty, so
  `registry.available()` lists only `composite` until a baseline module is imported. Register the target's names (or
  aliases), import the baselines in the package `__init__`, or tell the target owner the real names.
- **oracle / control models**: relOMA definition (4.7); the oracle heater codes are ~45-75 codes (0.2 %) above the
  replica's all-line bank codes, from the level averaging along the bus (product of level means here).
- **extraction**: VAL-7a/b and VAL-8 artefacts are read by key paths `replication.model_nominal.{n_g_outer,
  n_g_centreline, energy_fraction_si, gamma_robinson}` and `comparison[VAL-8x].predicted` (VAL-8e: `{tau_us: [..],
  share: [..]}`, `share_frac` also accepted); keep them. VAL-7c needs a de Cea thermal stack in thermal_fem.py writing
  `thermal_val7_decea.json` with `tau_us`.
- **plant_optics**: `optics._ring_eval` passes read-only `np.broadcast_arrays` views to numba when every input has size
  1: numba emits "FutureWarning: future versions will not create a writeable array from broadcast_array". Copy after
  broadcasting.
- **scenarios**: `laser_drift` converts the 0.02 nm limit to one frequency drift at 1310 nm for every line, so the
  1318.6 nm line drifts 0.0203 nm in 24 h, above the printed limit (CAL-12c fails by 1.3 %). Convert per line.
  The aggressor site of g26 (30 um from ring 0) makes the slide-10 lanes non-uniform (4.2).
- **report**: read `latest_predictions.json` (schema `mrm.validation.predictions/1`: per record id, role, target_status,
  scored, replica, preset, value, unit, sigma, tolerance, replica_status, predicted, predicted_nonfinite, variants,
  conditions, notes, detail, evaluation, runtime_s; top level frozen_sha256, git_commit, counts). Score with
  `targets_io.evaluate` yourself; `evaluation` is informational. Blocked/error records carry the reason in `notes`.
  Flag VAL-3b ("method changed after the comparison", 5), the VAL-7b definition conflict and the relOMA definition
  (4.7). Regenerate before every report.
- **calibration**: the replicas are the forward models for CAL fits; `variants` and `conditions` say which hypothesis
  each number is under. CAL-13a/b residuals are scenario-calibration residuals, not predictions. `heater_power_per_fsr`
  and `cold_resonance_offset` close R2/R3 by construction. CAL-8 identifies the lumped monitor gain with the
  neighbour leakage included (all lines on): fitting it with the own-line value would bias the gain by ~5 %.

## 7. Tests and performance

`tests/validation/test_replicas_static.py`: 48 fast tests + 1 slow, all pass (~50 s fast with a warm numba cache on
the shared machine). With `tests/closure/test_targets_schema.py` and `tests/verification/test_validation_isolation.py`:
146 passed. Checks: Q vs realized Q_L (2 %), P_FSR vs R2 (1.5 %), offset vs R3 (2 pm), low-power bias ladder = junction
table (0.3 pm), driver-law ordering and the H86_sensor 1/0.8 ratio, lock definitions, bistability vs power, EO response
vs a TCMT ODE (1 %) and vs mrm.waveform (0.5 %), static state = reference-plant fixed point (0.05 K), bank locks vs the
engine's quasi_static, line powers / absorption / codes vs mrm.control.models, dynamic lock through the adapter =
quasi-static (1.5 %), per-ring set-points of the dynamic request, ambient code excursion dynamic vs quasi-static (5 %,
slow), VAL-5 hysteresis reporting, the p_in convention and the PD temperature of the dark current, the driver
compliance, the VAL-8e artefact keys, non-finite predictions marked, feasibility plumbing (limits passed, best by miss).

Under `NUMBA_DISABLE_JIT=1` the whole file runs: the pre-review version passed 38 of 39 in 14.5 min (the one failure
was the VAL-8e key, fixed); after the review, the 23 new or changed tests that exercise the new paths (bank with all
lines and bus filtering, dynamic request, hysteresis, power convention, compliance, EO waveform mode, VAL-8e,
feasibility, reference-plant fixed point and adapter) pass in 10.3 min. The at-power 2-D sweeps and the all-line bank
locks take ~1-4 min each in pure Python.

Runtime of `predict_all` on g26/g25 (warm cache): ~2 min, peak RSS 0.6 GB; the CAL-9 replicas (8 dies x 7 hypotheses
with A1 calibration, FOH simulation and all-line bank locks) take ~17-25 s each, VAL-6 ~21 s, CAL-6 ~10 s, everything
else < 4 s.

## 8. Review (2026-10-09): what changed and why

Bugs fixed:
1. `_dynamic_lock` gave EVERY ring and world ring 0's set-point, code and warm state (wrong for g26's 8 lanes, and the
   dynamic VAL-6 averaged the excursion over those mislocked lanes); now per world and ring from the bank lock.
2. VAL-5 note claimed "single-valued (no hysteresis)" at 0 dBm; the line is bistable between 66 and 68 degC. The
   hysteresis is now computed (opposite sweep at the same drive) and reported.
3. Monitor dark current was taken at 25 degC regardless of the PD temperature (LockPoint now has `t_pd_k`; at a
   105 degC base the dark current is ~0.1 uA).
4. Driver compliance (07 P1-10) was ignored by the static heater law; now applied as in `heater_power_row` and
   flagged as infeasible in the bank and map protocols.
5. `p_in_dbm=None` meant P_in -> 0 in some replicas and the preset power in others (relOMA_map, monitor,
   bistability); one convention now (section 1).
6. BND-3 feasibility: hard-coded 298.15 K base (now the scenario's), the target's rate / latency limits were never
   passed to the run function (now in `options`, as numbers: the target file writes 1.0e6 as a string), and "best" was
   the narrowest band rather than the smallest miss of the target band.
7. VAL-8e read `share_frac`/`shares` but the extraction writes `share`: NaN shares (now read; a missing pair blocks).
8. `current_ua_at_base_105c` was computed at t_base + 80 K whatever t_c was.
9. A NaN prediction became `null` in the JSON with status ok; records now carry `predicted_nonfinite`.

Physics completed: wrong-line light and bus filtering in the bank protocols (section 2); the EO replica in the README's
waveform mode (section 5). Both moved predictions (section 3) and are disclosed in section 5.

Not changed, for the record: CAL-6 hypotheses H_solver and H86_sensor give identical ambient readings (H86_sensor's
ring sensitivity is H_solver's; only its thermometer differs), and dac_pwm reads like voltage drive (P = u V_FS^2/R):
both are correct.

No controllers are in this task's files: the only loop here is the test adapter `ReferenceLockRunner` (Truth-fed,
N = 1, documented as such); it was not stress-tested as a controller. The quasi-static protocols assume an ideal
settled loop; how a real controller fails (noise, model mismatch, A1, a laser power step, quantization) is BND-3's
dynamic path, blocked until the engine adapter and the named controllers exist.
