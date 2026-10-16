# Handoff: plant sensing, data path and laser lines (`plant_sensing`)

Status: done, then reviewed (section 5 lists the review findings and fixes). Files: `mrm/plant/sensing.py`,
`mrm/plant/datapath.py`, `mrm/plant/laser.py`, and the tests in `tests/verification/test_sensing.py` (39 tests),
`test_datapath.py` (21) and `test_laser.py` (14). All pass in about 10-30 s per file. Under `NUMBA_DISABLE_JIT=1`
they pass too (minutes, because the Monte Carlo loops then run in Python). Each file has a subprocess test that checks
that the outputs are bit-identical with and without numba.

Contract items covered: 07 P1-9 (PD, Arrhenius dark current, background light, reference tap), P1-10 (AFE, ADC, PGA,
temperature sensors, heater readback, the fault masks adc_drop, adc_stuck, adc_saturate and dac_glitch), P1-13 (exact
noise discretization, counter RNG, batch and thread invariance), the P0-7 monitor sample blocks with sample ticks,
P0-11 (ones count, bit statistics, error counter, BER(OMA)), P1-12 (line table, line enable, failover steps) and the
P0-10 drift time compression. Not covered: P2-8 (optional). A junction-photocurrent monitor needs no new code: feed
the absorbed power and an effective responsivity to the monitor kernel. It has no config switch.

## 1. How to wire it into the plant and the kernel

Every per-tick kernel has a `*_row` form for one world, for use inside `prange` over B in `sim/kernel.py`, and a
batched form. The Python classes (`MonitorBank`, `TempSensorBank`, `ReadbackBank`, `SensingBank`, `DatapathBank`,
`LaserBank`) allocate and hold the state arrays. For the compiled kernel, build the bank and pass its arrays to the
row kernels: `bank.sf, si, fx, fc, fv, fr, fm, key_adc, key_shot, key_dnl` and `hw.pr, pw, fl_coef, ...`. See
`MonitorBank.tick` for the exact argument order.

Order inside plant interval [k, k+1) (tick convention of `mrm/core/timing.py`):

1. `LaserBank.tick(k, enable, f_offset_hz, p_scale, tune_codes)` gives `(freq_hz, power_w)` (B, L) at the
   laser-module output. The plant multiplies by `pp.laser.input_transmission` and propagates along the bus.
   Scenario signals (mrm/scenarios/generators.py): `enable = laser.line_enable` (a power factor in [0, 1]; the bank
   multiplies the power by it), `f_offset_hz = laser.dfreq_hz`, `p_scale = 10**(laser.power_delta_db / 10)`.
2. `DatapathBank.pattern(k, density = data.ones_density, traffic_on = data.driver_enable)` gives the ones density
   rho (B, N) (`traffic_on` may be the float signal: on at >= 0.5). The optics averages the two NRZ levels with
   rho.
3. Optics and self-heating.
4. `MonitorBank.tick(k, p_pd_w = R_DROP_SUM (rho-weighted), p_bg_w = total bus power at ring n,
   t_pd_k = T at the ROM observer monitor_pd (K), t_adc_k = None with t_base_k = thermal.t_base_k +
   thermal.xpu_rise_k at k + 1 (the bank applies ring.monitor.adc_ref_site: the package base by default, t_pd_k for
   monitor_pd; MonitorHw.adc_temperature), pga_select = actuator monitor.pga_select,
   fault = sensing.fault_mask(B, N, adc_drop=fault.adc_drop, adc_stuck=fault.adc_stuck,
   adc_saturate=fault.adc_saturate, dac_glitch_codes=fault.dac_glitch_codes))` (`fault_mask_row` in the kernel).
5. `DatapathBank.errors(k, oma_tx_w = RL_POUT1 - RL_POUT0 of ring n's assigned line at the bus output)`.
6. `TempSensorBank.tick(k, T at the temp_sensor observers (B, S))`.
7. `ReadbackBank.tick(k, HeaterTickOut.v_v, HeaterTickOut.i_a)`.

At a controller edge (event order step 2 of timing.py): `MonitorBank.release(edge)` and `.channels(rel)`,
`TempSensorBank.release(edge)`, `ReadbackBank.release(edge)`, then `DatapathBank.bitstat_from_monitor(edge,
monitor_bank, p1 = R_DROP_SUM1, p0 = R_DROP_SUM0, thr = actuator datapath.bitstat_threshold, density = the
scenario's data.ones_density of the last interval (not the rho that `pattern` returned), rin_per_hz =
pp.laser.rin_per_hz)`, then
`DatapathBank.release(edge)`. `SensingBank.release` merges the sensor dicts. Each bank returns `(sensors, new)` dicts
with the channel names of `mrm/core/channels.py` and their dtypes: `latest` channels hold their value, `sum`
channels report only the time since the last release, and every channel has a new-sample flag.

Seeds: noise banks take `rng.batch_seeds(experiment.seeds.noise, B)`. The DNL thresholds, the INL bow and the laser
static errors and drift rates take `pp.seeds["device"]`. The laser random walk takes the scenario seeds. `reset(seed)`
re-keys the noise and never redraws a device property (07 P1-13).

Cost per 1 us tick, measured with 8 rings or lines and B = 1, on the shared M2 at load average 15 to 22, so roughly
2 times pessimistic:

| Kernel | Cost |
|---|---|
| Monitor | 0.8-1.5 us; 5.3 us with 18 flicker poles |
| Laser | 0.34 us without RIN; 0.8 us with RIN |
| Pattern, binomial | 2.0 us |
| Pattern, off or PRBS | under 0.5 us |

With the optics at 9-14 us (plant_optics handoff), the 07 P0-9 target of at most 10 us per tick is already
exceeded. Flicker and binomial pattern noise should stay off in long soaks unless they are the object of study.

## 2. Decisions and deviations from the contract (please review)

1. **No `mrm/plant/io.py`.** 07 P0-6 puts the SensorBank in `io.py`, but ARCHITECTURE v2 lists no such file and no
   owner for it. The sensor side of that bank (window, ADC latency FIFO, sample blocks, held channels and flags) is
   in `sensing.py`.
2. **Noise is generated at the sample, not per tick.**
   - integrate_dump: the kernel accumulates the sum of S_k/2 over the window ticks and draws one normal per sample.
     The variance is S/(2 T_w) for a constant S, and stays exact when S varies over the window.
   - rc: the kernel keeps the decayed discrete Gramian since the last sample and draws one normal per sample.
   - Flicker: J OU processes, stepped exactly from sample to sample with closed-form 2 x 2 transitions and Gramians.
     These match Van Loan's method in a test.

   The result is dt-invariant (tested within 5 % for dt = 0.25 to 2 us; in practice within 0.5 %) and cheaper than
   per-tick draws. Shot noise uses its own stream, `Stream.SHOT_NOISE`, so toggling `fidelity.shot_noise` leaves the
   ADC draws unchanged (tested).
3. **Laser RIN enters the line power per tick,** as the exact tick average of white RIN (variance RIN/(2 dt)). It
   reaches the monitors through the optics and is therefore correlated across rings, as it is on hardware.
   Consequence: an rc AFE whose tau is comparable to dt sees ZOH-held RIN. The ADC white noise is not affected.
4. **Laser drift is a pure function of time.** It uses Levy-construction Brownian motions keyed by dyadic node ids,
   interpolated on a 1 ms scenario-time grid. Multi-tick or quasi-static stepping (P0-10), batch size and thread
   count do not change it (tested). `scenario.time_compression` rescales only the drift, never RIN.
5. **The 24 h drift figures are bounds, not rates.** The published numbers are +-0.02 nm (about 3.5 GHz) and
   +-2.5 % over 24 h after burn-in (YOLE via textbook REF). `drift_from_24h_bound` offers four splits, all
   **assumed**:
   - `linear`: per-line rates uniform within the bound;
   - `worst_linear`;
   - `random_walk`: 24 h rms equal to bound/2;
   - `none`.

   `LaserHw` defaults to no drift.
6. **G26 line plan reading (inferred).** One 200 GHz comb of 16 lines centred at 1310 nm. Each direction is a 400 GHz
   grid, and the two are offset by 200 GHz (the textbook REF agrees: compositeGridGHz 200, perDirectionGridGHz 400).
   The grid spans 1301.47-1318.64 nm. The comb read-off is 1301.5-1318.9 nm (+-0.1 nm): the blue end agrees, the red
   end is 0.26 nm (about 45 GHz) longer than the grid, more than the read-off error; this is unresolved. The PAT-4
   example grid (8 lines, 1301.47 ... 1309.43 nm, 200 GHz apart) equals the comb's 8 bluest lines (tested to
   0.006 nm). That supports the 200 GHz comb and its anchor; it says nothing about how the two directions split the
   comb. Which direction takes the lowest line is not published: `tx_slot=0` is assumed.
7. **Bit-statistics noise needs a sampler bandwidth.**
   - sigma^2 = (2 q i + S_tia)/(4 tau_s), with tau_s = 1/(2 pi f_3dB). This is the exact variance of an instantaneous
     sample through a single-pole front end of a hardware bandwidth, not PSD x a dt-derived bandwidth.
   - `bitstat_bw_hz` defaults to 16 GHz. That is the G25 RX AFE bandwidth (printed, simulated), **assumed** to carry
     over. No Lightmatter value exists.
   - The comparator is code-domain: a sample counts as above when code > thr, that is x >= thr + 0.5.
8. **Pattern models.**
   - prbs7/9/15/23 are exact from packed tables (ITU-T O.150 taps) while the scenario density is 0.5, per tick and
     in the bit-statistics window (which must be given the scenario density; the engine passed the realized
     n1/nb until the 2026-10-16 review, section 6).
   - prbs31 (256 MB packed) and any other density use Binomial(n, rho). This is a stated approximation: an
     m-sequence window far shorter than the period is close to binomial.
   - Densities 0 and 1 are exact in every mode.
9. **Error counter.**
   - Expected errors are accumulated per tick, and Poisson(lambda) is drawn at each period boundary k + 1, keyed by
     that tick.
   - A delivery holds the last count; `rx.error_count_valid` marks it.
   - BER uses a Gaussian Q model, Q linear in OMA, calibrated at BER_ref = 1e-9 on the receiver sensitivity.
     Defaults: worst G26 lane -12.8 dBm OMA (textbook REF read-off); 3 dB link loss (printed chip-to-chip; on-chip
     TX and RX path losses assumed 0).
10. **Faults.**
    - A stuck converter repeats its last good code. By default it marks `monitor_valid` False, as the ChannelSpec doc
      says; `stuck_flagged=False` makes it silent, which is the harder and more realistic test. The clip state of the
      hidden conversion does not reach `adc_overrange` (review fix).
    - A dropped sample is not delivered, makes `monitor_valid` False and raises no `adc_overrange` (review fix).
    - The scenario fault channels are floats (mrm.scenarios: on at >= 0.5); `fault_mask` / `fault_mask_row` turn
      them into the int8 FAULT_* mask.
    - Saturate forces full scale with `adc_overrange`.
    - `dac_glitch_row` is provided, but the electrical owner must apply it (section 4).
11. **ADC DNL** is modelled as per-code threshold errors that are pure functions of the device seed, so no 2^bits
    table is stored. |DNL| < `adc_dnl_lsb`, clipped to 1 so the thresholds stay monotone. No missing codes below
    1 LSB (tested). INL is a per-device bow, uniform within +-`adc_inl_lsb` (from the device seed).
12. **Temperature sensors** average over the whole conversion period (boxcar; **assumed** for an integrating or
    sigma-delta converter). `noise_rms_k` is per sample. The flicker corner is applied with S_white = 2 sigma^2 T_s.
    The type (diode, rtd, thermistor) is recorded but changes nothing, because no type-specific data exist.
13. **adc_mux.** Ring n is served by converter n // C. `round_robin` staggers ring n by floor((n mod C) P / C)
    ticks: the C rings take turns on one integrating converter, so C W <= P (checked since the 2026-10-16 review).
    `bank` samples every ring at phase 0 (simultaneous sample-and-hold: a front end per ring ahead of the shared
    quantizer; **assumed** meaning). Under both, the C rings share the converter's DNL thresholds and INL bow; the
    per-ring offset and gain error stay per ring. The rate is per ring, as decided in core_control_api. The
    conversion (de-integration) time is not modelled (adc_latency_s carries it).

## 3. Physics and contract doubts

- **PAT-2 integrator full scale against the G26 monitor current.** I = C V / T_int = 16 pF x 0.5 V / 2 us = 4 uA
  (`integrator_full_scale_a()`), if the 500 mV reference is also the integrator swing (inferred; the patent does not
  say so). The G26 monitor reads 15.5 uA (read-off). [review: PAT-2 integrates i_det - i_os,det after the ANALOG
  baseline subtraction, so the integrator only has to hold the current above the baseline. With a baseline of about
  12-15 uA, 4 uA of span is enough, and the doubt mostly disappears. The model matches this: `baseline_a` is
  subtracted before the PGA and the ADC full scale.] Also printed in PAT-2: D = ceil[N (i_det - i_os,det) / (i_ref -
  i_os,ref)] with N = 400, i.e. about 400 counts (9 bits) per conversion and a ceil quantizer. The model rounds; a
  ceil quantizer equals rounding plus `adc_offset_codes = 0.5`. Together with core_control_api's 16-bit vs about
  400-count point, this makes `adc_full_scale_ua`, `adc_bits` and `baseline_ua` one joint hypothesis.
- **07 D20, "x1.6 to 1.9 per 10 K".** E_a = 0.35-0.38 eV gives x1.56-1.62 per 10 K at 25-35 C (tested), and less when
  hotter. x1.9 needs about 0.5 eV, which is the materials owner's stress corner (Balbi 2009). The text of D20 mixes the
  two; the code follows the 0.35-0.38 eV range.
- **RX sensitivity.** REF gives -14.0 / -12.8 dBm; 01 D12 re-reads -14.7 / -12.5 (both are module constants). The
  curves were measured at 56 Gb/s PRBS9, and G26 runs 50 Gb/s, so using them is slightly pessimistic. The OMA
  reference plane (RX PD input) against our TX bus output needs the full link loss. Only 3 dB is printed.
- **Laser drift correlation within a module** is not published (07 Q9). rho is an option, default 0.
- 07 P1-9 `test_monitor_tracks_intracavity_energy` and `test_dark_current_vs_heater` need optics plus thermal plus
  sensing, so they are integration tests (section 4). The sensing side is tested here: dark current against PD
  temperature through the codes, and level codes.

## 4. Requests by owner

### params (schema and realize)

Keys without a schema entry. My builders take them as keyword arguments today.

- `ring.monitor.baseline_ua`: the analog baseline subtraction of PAT-2 (ring level).
- `ring.monitor.reference_tap.{present, coupling_frac, responsivity_a_per_w, dark_i0_na}` (07 P1-9).
- `ring.monitor.stuck_flagged` (bool setting).
- `ring.monitor.flicker_band_min_hz`: setting, default 1e-3.
- `ring.heater.readback.{noise_rms_codes, average}`. Also carry the integer `ring.heater.readback.adc_bits` in
  PlantParams (it is neither in HeaterParams nor in `pp.values`); `ReadbackHw.from_plant_params(pp, adc_bits=...)`
  needs it passed by hand today (default 12, assumed).
- `ring.monitor.adc_mux.order` is a schema setting but `MonitorParams` does not carry it:
  `MonitorHw.from_plant_params(pp, mux_order=...)` needs it passed by hand (default round_robin).
- realize fills an absent `ring.monitor.dark_activation_ev` with 0 (a temperature-independent dark current, against
  07 P1-9). `MonitorHw.from_plant_params` now substitutes 0.37 eV (inferred, 02 section 10) when the key is absent;
  better, realize should default it.
- `datapath.bitstat_noise_pa_per_rt_hz` (the bit-rate sampler's input noise density; assumed, no data).
- `datapath.rx_sensitivity_oma_dbm`: level ring, i.e. per lane.
- `datapath.ber_ref`, `datapath.link_loss_db`, `datapath.ber_floor`, `datapath.bitstat_bw_ghz`.
- `laser.freq_drift_hz_per_s` (line), `laser.freq_random_walk_hz_per_rt_s` (die), `laser.power_drift_frac_per_s`
  (line), `laser.power_random_walk_frac_per_rt_s` (die), `laser.drift_common_frac` (die; rho), and the setting
  `sim.laser_drift_grid_s`.
- `TempSensorParams` lacks `adc_bits` (S,) and `code_zero_k` (S,), although the schema has `temp_sensors[].adc_bits`
  and `code_zero_c`. Please carry them. `TempSensorsHw.from_plant_params` defaults them (12 bits, -40 C, assumed).
- `LaserParams` lacks `source_laser` and `pol` from `channel_plan.lines`. `LineTable.from_plant_params` assumes that
  tx lines are 'local' and rx lines 'remote'.
- If `laser.line_offset_ghz` has variation in a preset, do not also call `draw_static_error_hz`: realize already put
  that error in `line_freq_hz`.

### core_control_api (channels)

- Optional channels for the reference tap: `tx.monitor_ref.samples / sample_ticks / count / latest`,
  `monitor_ref_valid`, `adc_ref_overrange`. `SensingBank` emits them when a tap exists.
- Suggested: `datapath.bitstat_ones` (B, N) int32, the ones count of the bit-statistics window. The transmitter knows
  it, and a Sun-style controller needs it to form the statistic. The bank computes it (`bs_n1`).

### plant_electrical

`dac_glitch`. Please add a per-tick additive code offset applied to the code that LEAVES the DAC latency FIFO, before
the DAC transfer. The register readback must not show it. `sensing.dac_glitch_row(codes, fault, glitch_codes, lo, hi,
out)` computes the glitched code from the scenario mask (FAULT_DAC_GLITCH = 8).

### kernel / plant integrator

- Wire as in section 1.
- Write the cross-module 07 P1-9 tests:
  - `I_mon * kappa_abs / kappa_d = R * P_abs`;
  - dark current against heater power through the ROM monitor_pd observer.
- `pop_period_ticks` must be at least the real release period, or the FIFOs overflow. `MonitorBank.fifo_lost`
  counts the losses. `m_max` now defaults to ceil(pop_period_ticks / ADC period), so the block holds every sample.
- Build the monitor with `MonitorHw.from_plant_params(pp)`: it stores `pp.seeds['device']` in `hw.device_seeds`,
  and `MonitorBank` uses them for the DNL thresholds when `device_seeds` is not passed (before the review, a bank
  built without `device_seeds` gave every world the DNL pattern of device seed 0).
- Heater readback in `dac_pwm` mode: `HeaterTickOut.v_v / i_a` are rms values. An averaging readback ADC would see
  mean values. R = V/I is the same either way; V x I is the true power only for rms. The model uses rms (assumed).

### content (presets)

- Monitor: `adc_full_scale_ua`, `adc_bits` and the baseline should be a joint hypothesis (section 3).
- Data path: set `datapath.error_count_latency_s` explicitly. If it is absent, realize gives 0 ticks while
  `mrm.core.channels.ChannelHardware` assumes 1 ms for the ChannelSpec.
- Data path: with 10 dBm per line, 6 dB chip input loss, the printed relOMA and a 3 dB link, the G26 lane has about
  10 dB of OMA margin and the model counts no errors at all. If a held-out BER number exists (LM26-P Fig. 3(b) shows
  an inset BER line near 7e-13, 01), an effective link loss that includes on-chip TX/RX losses is needed.
- Laser drift: pick a drift split and say it is assumed.
- Static error: G26 spec +-20 GHz; demonstrated +-15 GHz.
- RIN: no published value. Leave it at 0, or add an assumed corner.

## 5. Adversarial review (second agent): findings and fixes

Each item says what was wrong, what changed, and the test that now covers it.

0. **Vacuous statistical tests (test bug, fixed).** `pytest.approx(x, rel=r)` keeps its default `abs=1e-12`, so any
   comparison of values far below 1e-12 always passes. `test_flicker_sample_variance_is_stationary` compared a
   sample variance of 3.5e-15 A^2 against a WRONG analytic value of 8.3e-16 A^2 (it used only the conditional window
   variance 2 var g(u)/u^2 and dropped the m_a^2 var part) and passed. The kernel was right (3.50e-15 measured
   against 3.55e-15 for the correct stationary value 2 var (u - 1 + e^-u)/u^2). The same default made the E_a check
   (5.9e-20 J), the BER checks near 1e-9 and 1e-33, and the 1e-13-relative Arrhenius checks weak or empty. Every
   such `approx` now has `abs=0`, the flicker test uses the right formula and asserts that the flicker part
   dominates, and a matching rc test was added (`test_rc_flicker_sample_variance_is_stationary`). Other agents'
   tests may have the same trap.
1. **Heater readback with 32-bit codes (bug, fixed).** The V and I codes share one int64 FIFO word (v | i << 32). A
   32-bit current code wrapped negative (the schema allows 1..32 bits). `ReadbackHw.build` now refuses more than
   31 bits. Test: `test_readback_bit_limit_and_from_plant_params`.
2. **Laser line enable was cast to bool (bug, fixed).** The scenario signal `laser.line_enable` is a power factor in
   [0, 1] (a ramp is a turn-on); 0.25 gave full power. `LaserBank.tick` now multiplies the power by the clipped
   factor (bool input keeps its meaning). A negative `p_scale` now clamps the power to 0. Test:
   `test_enable_is_a_power_factor_like_the_scenario_signal`.
3. **traffic_on was cast to bool (bug, fixed).** `data.driver_enable` is on at >= 0.5; 0.3 meant traffic on.
   `DatapathBank.pattern` now thresholds float input at 0.5. Test: `test_traffic_on_accepts_the_scenario_driver_enable_float`.
4. **DNL keyed by device seed 0 by default (footgun, fixed).** `MonitorHw.from_plant_params` now carries
   `pp.seeds['device']`, and `MonitorBank` uses it. Test: `test_from_plant_params_device_seeds_drive_dnl_and_dark_activation_default`.
5. **Absent dark_activation_ev gave E_a = 0 (contract, fixed in the adapter).** See section 4 (params). Same test.
6. **Fault flags (fixed).** A stuck sample reported the overrange of the hidden conversion; a dropped sample still
   raised `adc_overrange`. Test: `test_fault_flags_follow_the_reported_code`.
7. **Block size default M = 1 (fixed).** With the default, every release with more than one sample lost samples
   silently into `lost`. Default now ceil(pop / P). Test: `test_default_block_size_holds_every_sample`.
8. **Bit-statistics sampler noise (contract gap, fixed as an option).** The sampler sigma reused the slow monitor's
   TIA noise density at a 16 GHz bandwidth, and laser RIN (modelled only per tick) was missing at the sampler
   bandwidth. New: `DatapathHw.build(bitstat_noise_a_per_rt_hz=...)` and `bitstat_from_monitor(rin_per_hz=...)`,
   sigma^2 = (S_tia + 2 q i + (R P)^2 RIN) / (4 tau_s). Tests: `test_level_codes_sampler_noise_override`,
   `test_bitstat_sampler_noise_override_and_reset`.
9. **Missing adapters (added).** `ReadbackHw.from_plant_params`, `fault_mask` / `fault_mask_row` (scenario float
   faults -> FAULT_* mask), `MonitorHw.codes_per_w` (07 P1-9 identifiable `monitor_codes_per_uw_at_drop` x 1e6).
   Tests: `test_readback_bit_limit_and_from_plant_params`, `test_fault_mask_from_scenario_signals`,
   `test_codes_per_w_is_the_small_signal_gain`, `test_sensing_bank_merges_every_channel`.
10. **Robustness (fixed).** A flicker corner below the band floor raised an exception (now: no flicker poles); too many
    flicker poles now fail loudly instead of overlapping RNG slots; `TempSensorsHw.from_plant_params` names a missing
    sample rate instead of dividing by zero; `reset()` clears every state and flag of the monitor, temperature,
    readback and data-path banks; `MonitorRelease` arrays are copies (a kept Observation is no longer overwritten);
    the reference tap no longer inherits the drop channel's analog baseline (`reference_tap(baseline_a=0)`).
11. **Statements corrected.** The PAT-4 grid supports the 200 GHz comb anchor, not the direction split (section 2,
    item 6); the 0.26 nm red-end gap to the comb read-off is now stated. The PAT-2 4 uA doubt is weakened by the
    analog baseline subtraction (section 3).
12. **Checked and found right** (re-derived): S/(2 T_w) and the sum of S_k/2 for a varying S (checked numerically
    with a ramp); the rc discrete Gramian and stationary start; the OU window and RC transition/Gramian closed forms
    (also against Van Loan); the log-spaced OU sum sigma^2 = S f_c ln r; flicker stationarity in rc mode (new check,
    0.8 % off at B = 1600); the rc sample autocorrelation exp(-P/tau) for any dt (new test
    `test_rc_noise_sample_autocorrelation_is_exact_and_dt_invariant`); Arrhenius x1.56-1.62 per 10 K; Q(1e-9) =
    5.998; the Levy midpoint variance (b - a)/4; RIN tick variance RIN/(2 dt); the G26 grid arithmetic.

Not changed, still open: the laser RIN seen by an rc AFE with tau near dt is ZOH-held (variance low by
1 - (2 tau/dt) tanh(dt/(2 tau)), 7.6 % at dt = tau, 0.5 % at dt = tau/4); the compute budget (section 1); the
cross-module P1-9 tests; the `dac_glitch` application in plant_electrical.

## 6. physics_sensing review (2026-10-16): findings and fixes

Confirmed by an independent verifier; tests in `tests/verification/test_engine_sensing.py` and
`tests/closure/test_sensing_notes.py`; docs/CHANGELOG.md 2026-10-16.

1. **Bit-statistics window density (engine bug, fixed).** The kernel and `mrm.sim.kernel_ref` passed the realized
   per-tick density n1/nb to `bitstat_window_row`, so the exact PRBS window (taken only at density 0.5) never ran:
   every window was Binomial(64, n1/nb) (8 % of edges matched the sequence; prbs7 std 4.1 instead of 2.8). It now
   takes the scenario density (`RF_DENS`, initialized at reset from the scenario at k0, so the first edge too).
2. **ADC reference site (physics, fixed).** `t_adc` was each ring's PD observer, an undocumented choice that made
   the ring's own heater and the aggressor move the monitor gain (-21 to -23 codes at the g26 lock; 0.14 % at PD0
   under A1). New setting `ring.monitor.adc_ref_site`: `package_base` (default; thermal.t_base_k +
   thermal.xpu_rise_k, the converter on its own die beside the PIC) or `monitor_pd` (the former model; needs one
   converter per ring). `MonitorHw.adc_temperature` is the rule; kernel row `RF_T_ADC`; the Python banks take
   `t_base_k`.
3. **Reference tap (silently ignored, now refused).** The engine builds no tap and the optics do not take its
   coupling share; `MonitorHw.from_plant_params` raises NotImplementedError for `reference_tap.present: true`.
4. **background_fraction (doc, fixed).** It multiplies the TOTAL bus power at the ring (all lines), as section 1
   says; the schema doc said "the line power". g26 range [0, 0.01] -> [0, 3e-4] (0.01 put 69-191 uA on the 40 uA
   scale; 3e-4 keeps every nominal drop peak plus the background ~2 uA below full scale).
5. **Per-world baseline (bug, fixed).** `from_plant_params` flattened `baseline_a` (B, 1) to (B,), which build reads
   as per ring when B == N; it now passes (B, 1).
6. **adc_mux (fixed).** Overlapping windows on one converter are rejected (C W <= P) and the DNL thresholds and INL
   bow are keyed by converter (`PR_ADC_UNIT`, `draw_inl_bow_lsb(channels_per_adc=)`).
7. Not changed: the bit-statistics LEVELS at the first edge after a reset are the reset placeholders (no interval
   has run); a controller should ignore the first window.
