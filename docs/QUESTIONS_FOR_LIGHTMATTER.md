# Questions for Lightmatter

> Kenny: delete every "Goes in" line and the section "For Kenny" at the end before you send this.

I am Kenny Guo, an electrical engineering student building a control-oriented simulation model of a Lightmatter-class
silicon microring-modulator (MRM) transmitter, which I use to design heater wavelength-lock controllers meant for real
hardware. Most of the model's constants now come from your public papers, talks and patents or from my own solver
estimates, so the questions below are ranked by how much each answer would change which controllers work and how
well, and a partial answer or a range is still very useful.

*Terms used below:* "the 2026 link" is the 16-wavelength bidirectional link of OFC 2026 M4B.3 (Anand et al.) and its
talk deck; "the 2025 device" is the microring transmitter of OFC 2025 Tu3J.6 (Baghdadi et al.). With each number,
please give its conditions (temperature, optical power, bias) and an uncertainty where you have one.

---

## 1. Heater step response at the lock point

With one TX ring of the 2026 link at its lock point and the loop then held open, apply a small heater DAC step (one
that moves the resonance by a few percent of the linewidth) and record the drop-port monitor ADC code at its full
rate from 1 µs to 100 ms after the step. If the on-chip ADC samples slower than 1 µs, an external photodiode on the
drop or through port is fine. Repeat with the step on the adjacent ring's heater, and then on the on-die aggressor heater, recording the
same ring each time. Please also state the heater type and position (doped silicon in the ring, a silicide resistor,
or metal above the ring), whether the ring is undercut, and whether the ring derives from the GlobalFoundries 45SPCLO
PDK ring.

*Why it matters:* the heater and crosstalk time constants and the heat-transport delay set the fastest stable lock
loop; mine are mostly borrowed from imec literature and span about a factor of 10 in achievable loop bandwidth.

> **Goes in** (Kenny only): fit the trace to the interim thermal leaves `ring.thermal.tau_fast_s`,
> `ring.thermal.tau_slow_s`, `ring.thermal.share_fast_frac`, `ring.thermal.heater_node_tau_s`; the tail is set in
> every world by `uncertainty.thermal.package_tail_tau_s` and `uncertainty.thermal.package_tail_share_frac` (over
> `ring.thermal.tail_tau_s` / `ring.thermal.tail_share_frac`), so restate those two; the delay goes in
> `uncertainty.thermal.transport_delay_s` (`ring.thermal.transport_delay_s` is a schema key that nothing reads); pin
> `uncertainty.thermal.time_scale_fast` and `uncertainty.thermal.time_scale_slow` to `nominal`. Adjacent-ring trace:
> `ring.thermal.xt_nearest_frac`, `ring.thermal.xt_tau_ref_s`, `ring.thermal.xt_dist_ref_um`,
> `ring.thermal.xt_tau_min_s`; pin `uncertainty.thermal.crosstalk_scale`. Aggressor trace: `ring.thermal.site_ref_um`,
> `ring.thermal.site_floor_frac` (`ring.thermal.site_max_frac` is another schema key that realize does not pass to
> the plant). Heater type: `uncertainty.thermal.family` with `ring.heater.tcr_per_k` and
> `ring.thermal.heater_self_ratio` (family caveat in step 3 below). If a FEM artefact matches the trace:
> `fidelity.thermal_model: rom` with `ring.thermal.rom_file` (mrm/data/extracted/thermal_rom_*.npz). No replica or
> target exists for a step trace.

## 2. One open-loop static heater sweep of the drop-port monitor

With the loop open, sweep one TX ring's heater DAC code slowly across more than one FSR, first up and then down, and
record the drop-port monitor ADC code against the heater code. Do this at two or three laser powers (please give the
power per line at the ring input, in dBm) and at a known stage temperature (°C), then repeat at that temperature
plus 10 °C.

*Why it matters:* this one data set gives, in the controller's own units, the linewidth and the FSR in heater codes,
the monitor gain, the curvature of the heater drive law, the optical self-heating (from the up/down asymmetry and how
it grows with power) and the heater codes per kelvin.

> **Goes in** (Kenny only): linewidth in codes -> `ring.optics.q_loaded` (through the heater law); FSR in
> codes -> the heater law `ring.heater.full_scale_v` (or `ring.heater.full_scale_a`, `ring.heater.full_scale_w`) with
> `uncertainty.heater.driver_mode` and `ring.heater.p_fsr_mw`; peak monitor code -> `ring.monitor.gain_scale` (pin
> `uncertainty.monitor.gain_scale`), off-resonance floor -> `ring.monitor.dark_i0_na` and
> `ring.monitor.background_fraction`; asymmetry and its power dependence -> `uncertainty.self_heating.eta` and
> `uncertainty.self_heating.z_ratio` (`ring.thermal.z_ratio_opt_heater`); codes per K -> `uncertainty.thermo_optic`
> (`ring.thermal.dlambda_dt_pm_per_k`); coupling branch -> `uncertainty.er_mechanism` and
> `uncertainty.coupling_regime`. Needs a new calibration target with a replica (mrm/validation/replicas.py) and a fit
> group (mrm/calibration/groups.py); none exists for a code sweep.

## 3. Layout, package and thermal crosstalk

For the TX ring bank of the 2026 link, please give the ring pitch (µm) and the position (x, y in µm from the ring
centre) of each ring's heater, monitor photodiode and driver, of the on-die aggressor heater used in the aggressor
tests, and of any temperature sensors. Is any electronic die stacked directly above or below the TX rings (as in
Passage M1000), or are the SerDes beside the photonic die? How is the package held thermally in the bench tests (TEC
under the package, heat sink, free air)? If you have measured it: the heater crosstalk to the nearest and the next
ring, in pm of resonance shift per mW of the other ring's heater power, its settling time, and whether a stacked die
changes it.

*Why it matters:* in my model's latest sensitivity study the stacking and package choice was the largest single cause
of spread in simulated lock performance and the crosstalk size among the next largest, and the assumed aggressor
position is a prime suspect for why the model cannot reproduce the per-lane heater codes of your slide 10.

> **Goes in** (Kenny only): `layout.ring_pitch_um` (or ring records in `layout.sites`),
> `ring.thermal.pd_offset_um`, `ring.thermal.driver_offset_um`, `ring.thermal.temp_sensor_offset_um`, the `aggressor`
> record of `layout.sites` (A1/A2 recalibrate the aggressor power to the printed trajectory), `temp_sensors` (site,
> attach); stacking and package -> pin `uncertainty.thermal.stack` (`side_by_side`, `stacked_eic`,
> `side_by_side_tec_held`; its choices set the crosstalk leaves, so with measured numbers restate it as in step 3);
> crosstalk in pm/mW divided by the ring's own heater tuning in pm/mW -> `ring.thermal.xt_nearest_frac`,
> `ring.thermal.xt_next_frac`, `ring.thermal.xt_floor_frac`, `ring.thermal.site_floor_frac`, settling time ->
> `ring.thermal.xt_tau_ref_s`, `ring.thermal.xt_tau_min_s`; pin `uncertainty.thermal.crosstalk_scale`. Bears on the
> strict xfail test_g26_default_crosstalk_matches_the_fem_rom (tests/closure/test_thermal_crosstalk.py; VALIDATION.md
> closure-crosstalk-dynamics).

## 4. The TX ring itself

For the TX ring of the 2026 link: the silicon film thickness (160 or 200 nm), strip or rib waveguide and the slab
thickness, the waveguide width, the ring radius at the waveguide centre, the measured FSR (GHz or nm, and at what
wavelength), the loaded Q, whether the ring is under- or over-coupled at the operating point, the drop-port coupling
(power fraction per pass, or the drop-port Q), and the junction doping levels. For the 2025 device: the measurement
floor (the deepest notch the setup could resolve) of the published notch-depth-against-heater-voltage sweep.

*Why it matters:* the FSR decides how close each ring's other resonances sit to the other channels (lost and
wrong-channel locks in my model), and the coupling side decides how the extinction and the monitor signal drift while
the loop holds lock; the published data fit both coupling sides about equally well.

> **Goes in** (Kenny only): film -> pin `uncertainty.film_thickness` (`"160"` or `"200"`, the rib200
> artefact); strip/rib, slab, width -> `ring.geometry.width_nm`, `ring.geometry.slab_nm`, `ring.geometry.xsec` (a
> label: a new cross-section needs its own modes and coupling artefacts in mrm/data/extracted/ and the leaves that
> read them); FSR -> `ring.optics.fsr_thz` and pin `uncertainty.fsr: default`; radius -> `ring.geometry.radius_um`
> only as a closure check (R3 is then overdetermined with the solver n_g: state
> `relations.R3_geometry.tolerance_rel` and pin `uncertainty.fsr`; tested); loaded Q -> `ring.optics.q_loaded`, pin
> `uncertainty.q_loaded: nominal`; coupling side -> `uncertainty.coupling_regime` and `uncertainty.er_mechanism`; drop
> coupling -> `ring.optics.kappa_d_over_kappa`; doping ->
> `ring.junction.eff_at_0v_pm_per_v`, `ring.junction.capacitance_at_0v_ff`, `ring.junction.v_bi_v` and the
> free-carrier loss `ring.optics.loss_budget.alpha_fca_db_per_cm` (`uncertainty.self_heating.eta`); 2025 notch floor ->
> the instrument-floor nuisance of the CAL-2 fit (`notch_floor`, mrm/calibration/groups.py), not a preset key.

## 5. Controller input and output facts

Please state:

- the loop update rate (Hz);
- the monitor ADC: its type (for example the dual-slope integrating converter of US11409045B2), integration window
  (µs), sample rate, rings per converter (multiplexing), resolution (bits), and the LSB in µA of photodiode current,
  or else the code read at the ~15 µA drop current of OFC 2026 M4B.3 Fig. 3(b); any baseline subtraction or
  programmable gain;
- the heater drive: its law (voltage, current, power, or DAC plus PWM as in that patent), full scale (V, mA or mW),
  resolution, code limits, slew limit and maximum update rate, and the supply limit of the whole heater array (the
  total current in mA and power in mW available to all heaters at once);
- the latencies (µs) from a change at the photodiode to the ADC code, from the ADC code to the new heater code
  (compute), and from the heater code to heater power;
- whether the loop dithers, and on what (heater, bias or laser), with amplitude and frequency;
- the lock set-point: the monitor peak, a fixed fraction of the peak, or a point on the blue side of the resonance;
- the temperature sensors: type, sample rate, resolution and location;
- whether heater voltage or current readback exists, with its rate and resolution;
- whether the loop also commands the modulator bias.

*Why it matters:* these set the quantization, loop delay, actuator range and gain variation that a controller must
handle (under a voltage or current law the heater gain varies about 2.5 times across the 19,500 to 49,300 code range
of your slide 10), and which controller types the hardware allows.

> **Goes in** (Kenny only): `controller_interface.update_rate_hz` (whole ticks); `ring.monitor.afe_type`
> (`rc` | `integrate_dump`), `ring.monitor.integration_window_s`, `ring.monitor.adc_rate_hz`,
> `ring.monitor.adc_mux.channels_per_adc`, `ring.monitor.adc_mux.order`, `ring.monitor.adc_bits`; LSB =
> `ring.monitor.adc_full_scale_ua` / (2^bits - 1), and the 15 µA code also moves `ring.monitor.gain_scale` (CAL-8);
> `ring.monitor.baseline_ua`, `ring.monitor.pga_gains`; drive law -> pin `uncertainty.heater.driver_mode`
> (`ring.heater.driver_mode`), `ring.heater.full_scale_v` / `ring.heater.full_scale_a` / `ring.heater.full_scale_w`,
> `ring.heater.dac_bits`, `ring.heater.pwm_bits`, `ring.heater.pwm_period_s`, `ring.heater.code_min`,
> `ring.heater.code_max`, `ring.heater.slew_codes_per_s`, `ring.heater.dac_max_update_hz` (law and full scale also
> let VAL-6 become active: a planned re-freeze, mrm/validation/targets/README.md); supply ->
> `array.heater_supply_current_max_a`, `array.heater_total_power_max_w`; latencies -> `ring.monitor.adc_latency_s`,
> `ring.heater.latency_s` (fold the compute time in here: one owner per delay, no compute key),
> `ring.bias_dac.latency_s`, then pin `uncertainty.timing.latency_scale`; dither -> the reference controller
> (mrm/control/baselines/: dither_lockin, bias_dither, pi_lock), not a key; set-point -> the CAL-8 hypothesis
> (`CAL8_SETPOINTS` in mrm/calibration/groups.py) and `metrics.lock_reference` / `metrics.lock_target_pm`;
> sensors -> the `temp_sensors` table; readback -> `ring.heater.readback.present`, `ring.heater.readback.rate_hz`,
> `ring.heater.readback.adc_bits`; bias -> `controller_interface.bind.tx.bias`.

## 6. relOMA and the drive

Please confirm the reference plane of the relOMA maps (OFC 2026 deck slide 9; Hot Chips 2025 slide 16 notes "1 mW of
laser into the MRM"): is relOMA the OMA divided by 1 mW at the MRM input, or the OMA in dBm at the actual input power?
Please also give: the optical power per line at the MRM input during those maps and in normal operation (dBm); the
peak-to-peak swing across the junction (V), with the driver supply and topology (single-ended, differential,
stacked); the modulation efficiency of the 2026 ring (pm/V, and at which bias); what the 1.0 to 3.75 V bias axis is
(the DC reverse bias across the junction, and on which terminal); and, for the 2025 device, the optical power into
the ring when its resonance shift against reverse bias was measured.

*Why it matters:* with the 2025 drive and linewidth my model reaches only about -6 to -8 dB relOMA against the printed
-2.1 to -3.0 dB, so the reference plane or the drive must differ; and the optical power at the ring sets the
self-heating, one of the largest causes of lost and wrong-channel locks in my model.

> **Goes in** (Kenny only): reference plane -> the CAL-14 definition (`metrics.reloma_definition` has one
> value, `oma_over_pin`; the OMA-in-dBm reading exists only as the reported, unfitted `_reloma_dbm_reading` in
> mrm/calibration/pipeline.py); power at the MRM -> `optical_path.chip_input_loss_db` (with
> `laser.power_per_line_dbm`), pin `uncertainty.chip_input_loss: nominal`; swing -> `driver.swing_pp_v`; topology ->
> `driver.level_convention` (pin `uncertainty.driver.levels`), `driver.output_resistance_ohm`; efficiency ->
> `ring.junction.eff_at_0v_pm_per_v`, `ring.junction.depletion_exponent`; bias axis -> `ring.bias_dac.v_min_v`,
> `ring.bias_dac.v_max_v`, `driver.level_convention`; 2025 bias-ladder power (generation G25) -> the CAL-5 power
> hypothesis (`CAL5_POWERS` in mrm/calibration/groups.py) and so `ring.junction.eff_at_0v_pm_per_v`.

## 7. How "~15 GHz/°C" and "808 °C/s within the MRM" were measured

For the "~15 GHz/°C" thermal sensitivity (OFC 2026 M4B.3, introduction): was the ring held at a fixed heater code or
locked; what was the thermometer (the stage or TEC, an on-die sensor, or inferred from the stabilizer code or the
resonance shift); and what was the ring temperature? For "808 °C/s within the MRM" (section 3.2 and Fig. 3): the same
thermometer question; does "a period of 100 ms" mean one up ramp or one full up-and-down cycle; and did the 808 °C/s
test and the > 2000 °C/s test of the deck's slide 10 use the same on-die heater?

*Why it matters:* 15 GHz/°C (about 86 pm/K at 1310 nm) is above the roughly 65 to 77 pm/K that my materials data and
mode solves give for this ring between 25 and 105 °C, so the thermometer behind it changes the size (1.2 to 1.3 times)
and the slope (up to 2 times) of the temperature disturbance a controller must reject.

> **Goes in** (Kenny only): thermometer -> `ring.thermal.reference_rise_frac` (its `sensor` alternative, the
> `H86_sensor` choice) and `uncertainty.aggressor.referencing` (`resonance` | `temperature`); the sensitivity itself ->
> pin `uncertainty.thermo_optic` or write `ring.thermal.dlambda_dt_pm_per_k` with
> `ring.thermal.dlambda_dt_ref_ring_temp_c` and `ring.thermal.dlambda_dt_beta_per_k` (CAL-6 is never fitted); the
> 100 ms period -> the `reading` argument of `a2_808_aggressor` in mrm/scenarios/library.py (`symmetric_triangle`,
> `rectified_sine`, `one_sided_sawtooth`, `ramp_per_period`) and CAL-13b; same heater -> the `aggressor` record of
> `layout.sites` (A1 and A2 both use it now).

## 8. Raw data behind the slide-9 relOMA maps

Please share the raw samples (5 °C steps × bias points) behind the loop-closed and loop-open relOMA maps of the OFC
2026 deck slide 9, and the fixed heater code used for the loop-open map.

*Why it matters:* it is the only data set that tests the thermal tuning, the linewidth and the modulation depth
together; I would keep it out of every fit and use it only as an independent check.

> **Goes in** (Kenny only): held-out data for VAL-4a/b/c and VAL-5 in mrm/validation/targets/validation.yaml,
> which is frozen by sha256 (new data are a re-freeze or a private held-out file, never a calibration target); the
> loop-open code replaces the `align_peak_c` alignment in VAL-5's `replica_args` (also a re-freeze).

## 9. Laser and traffic

- Do the lines from one laser module drift together, in wavelength and in power?
- The laser RIN (dB/Hz).
- What the laser and the line power do during squelch.
- The laser turn-on sequence (ramp time, line order, and whether the heaters are preset before light arrives).
- Is the feed to the TX polarization-maintaining?
- Are traffic start and stop, idle and training patterns known disturbances to the loop (is the loop told about
  them)?
- The junction capacitance (fF) and series resistance (Ω), and the driver's dissipation per ring (static mW and
  energy per bit transition).

*Why it matters:* laser power steps and data activity move each ring's temperature by up to about 2 K in my model,
and the turn-on sequence decides whether a cold ring heats itself onto a wrong channel.

> **Goes in** (Kenny only): drift together -> the `common` argument of `laser_drift` in
> mrm/scenarios/library.py and the die/line split of the `variation` entry `laser.line_offset_ghz`; RIN ->
> `laser.rin_db_per_hz`; squelch -> the `squelch` scenario (`residual_swing`); turn-on -> `sim.reset_protocol`,
> `sim.reset_ramp_s`; polarization -> the `pol` field of `channel_plan.lines`; traffic -> the `traffic_start_stop`,
> `idle` and `training_pattern` scenarios and `datapath.transition_density`; junction ->
> `ring.junction.capacitance_at_0v_ff`, `ring.junction.series_resistance_ohm`; driver -> `driver.static_power_mw`,
> `driver.energy_per_transition_fj`.

## 10. Variation across lanes and dies

Please give the distribution (or the range) of the heater lock code at 25 °C across the lanes of a die and across
dies, the designed resonance stagger between the rings of a bank (if any), and the heater-resistance spread within a
die.

*Why it matters:* these set the ring-to-ring variation that a controller and its start-up search must cover, and which
lane runs out of heater range first at the hot end.

> **Goes in** (Kenny only): the `variation` entries `ring.heater.r_cold_ohm` and `ring.heater.p_fsr_mw`
> (`die_sigma`, `ring_sigma`) and `variation.geometry` (width, thickness, gaps); a designed stagger -> a per-ring list
> in `ring.optics.resonance_offset_nm`; the lock-code spread checks CAL-9b (lane offsets) and BND-2 (worst lane) and
> would need its own calibration target.

## 11. Bring-up policy

How does the loop acquire lock at power-up: the heater sweep range and direction (rising or falling power); whether a
stored calibration is used; whether order hops or the barrel-shift channel remap of US20240405908A1 happen between 0
and 105 °C; and how a lock on a wrong channel is detected?

*Why it matters:* with self-heating the ring is bistable, so the sweep direction decides where each ring lands, and a
lock on a wrong channel at power-up is a failure my model shows.

> **Goes in** (Kenny only): sweep direction -> `sim.reset_protocol` (`laser_ramp` | `branch_low` |
> `branch_high`) and the parameters of the acquisition controllers (mrm/control/baselines/ramp_acquire.py,
> mrm/control/baselines/wdm_supervisor.py); stored calibration -> `experiment.start` (`cold` | `cold_settled` |
> `warm`) and the factory table (mrm/control/calibration.py); hops and remap -> `metrics.allowed_assignments`
> (`fixed` | `cyclic` | `any`) and BND-2; wrong-channel detection -> the wdm_supervisor labels.

---

## For Kenny (do not send)

### How an answer enters the model

1. Write `mrm/params/presets/private/<name>.yaml` (git-ignored by mrm/params/.gitignore; checked with
   `git check-ignore`) with `extends: g26` and `meta: {name, kind: private, description, sources: {LM-PRIV-1: "who,
   date, how; NDA"}}`. `mrm.params.load("<name>")` finds it by name.
2. Write each number as `{value, provenance: private, generation: G26 (G25 for the 2025 device), source: LM-PRIV-1,
   note: "conditions"}`. A leaf that changes a parent value must restate provenance and generation. Settings
   (`controller_interface.update_rate_hz`, `ring.monitor.afe_type`, `ring.heater.driver_mode`, `ring.thermal.family`,
   `ring.thermal.stack`) are written bare.
3. Settle every uncertainty entry the answer decides, or the sampled worlds keep moving the value: `alts:
   {uncertainty.<entry>: <choice>}` when the answer is an existing choice; `nominal` for an interval entry whose leaf
   now holds the answer, and `default` for a hypothesis whose choices are leaf alternatives (`uncertainty.fsr:
   default` after a measured `ring.optics.fsr_thz`; tested); or restate the entry (an
   interval with low = high = nominal works; a restated entry's `choices` replace g26's, and its `note` is inherited,
   so restate that too). Caveat (tested): a pinned hypothesis applies its own leaf alternatives over private leaves
   (pinning `uncertainty.thermal.family: silicide_undercut` replaced a private `ring.thermal.tau_fast_s` with its
   `undercut` alternative). When an answer gives both the choice and the numbers, restate the entry with one empty
   choice (`choices: {measured: {}}`, `nominal: measured`, `weights: null`, provenance private) and set the switch
   (`ring.thermal.family`, `ring.thermal.stack`) directly.
4. Check with `mrm params explain <name> <key>` and `mrm params validate <name>`.
5. Traces and tables (questions 1, 2, 8, 10) are data, not leaves: keep the raw files in presets/private/ too. The
   calibration reads only mrm/validation/targets/calibration.yaml and is wired to g26 (`BANK` in
   mrm/calibration/groups.py, `extends: g26` in mrm/calibration/preset.py), and no replica exists for a step response
   or a code sweep, so fitting them on top of a private preset is new code. The slide-9 data (question 8) are held
   out: never fit them.
6. Do not run `python -m mrm.validation.report --preset <name>` as it is: it writes docs/VALIDATION.md unless `--out`
   is given, and it always writes its data and charts to docs/validation/<name>/, which git does not ignore. The
   calibration refit and the report regeneration of docs/KNOWN_ISSUES.md items 1-2 are due when the answers arrive.

### Ranking changes against the design review

The eleven questions follow the question list of the design review (docs/research/07_design_critique.md, section
5, "Questions for Lightmatter"), in a new order. Question n here was review question 1, 2, 8, 6, 3, 4, 5, 7, 9, 10,
11 for n = 1 to 11.

- Question 3 (layout, package, crosstalk) moved up from 8 to 3: in the last sensitivity study (docs/VALIDATION.md section 4,
  pi_lock_c under A1; computed before the 2026-10-17 fixes and marked stale, so recheck it after regeneration) the
  package/stack hypothesis has the largest total Sobol index for five of the six closed-loop metrics and the crosstalk
  scale is in the top 8, and CAL-9a/b/c fail under every hypothesis with the assumed aggressor site as first suspect
  (calibration.md 4.6, 8).
- Question 4 (the ring itself) moved up from 6 to 4: the FSR, eta and extinction-mechanism hypotheses it pins are all in that
  study's top 8 (the FSR first or second for unlocked and wrong-line rings), and CAL-1/CAL-2 fit both coupling
  branches about equally (calibration.md 4.2).
- Not moved: question 1 stays first although the fast-pole scale and the transport delay barely moved pi_lock_c in
  that study, because the review ranks it by the loop bandwidth a controller can reach, which a fixed-gain study
  cannot show. Questions 5 to 11 keep the review's relative order.

### Added to the review's questions, from the model's results

- Question 5: the heater-array supply limit (`array.heater_supply_current_max_a` 0.15 A and
  `array.heater_total_power_max_w` 0.2 W, both assumed; it binds in about a third of the sampled g26 worlds:
  KNOWN_ISSUES 40, integrator.md 8.3).
- Question 4: the 2025 notch-sweep floor (CAL-2 fits only with a ~30 dB instrument floor that nobody has verified:
  calibration.md 4.2, 11.4).
- Question 6: the driver topology and the 2025 bias-ladder power (calibration.md 4.3, 4.5 and 8: CAL-14 closes only
  near 2.96 Vpp at the at-power efficiency or with the OMA-in-dBm reading; CAL-5 gives 93.6 or 72.5 pm/V depending on
  that power; the fit is stale, but the gap is the point).
- Question 9: the driver dissipation (`driver.static_power_mw` 2 mW and `driver.energy_per_transition_fj` 100 fJ, no
  source: integrator.md 4).
- Question 1: an external photodiode is allowed, because the monitor ADC rate is 250 kHz in g26 (inferred from
  US11409045B2: `ring.monitor.adc_rate_hz`), slower than the 1 µs that the review asks for.
