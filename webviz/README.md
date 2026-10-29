# webviz: the mrm model, running

A website that shows the g26 plant model running: 8 silicon microring modulators (MRMs) on one bus waveguide, each
with an in-ring heater, a PN junction for the data and a monitor photodiode (PD) on its drop port; 8 laser lines on
a 400 GHz grid near 1310 nm; a heater lock holding each ring on its line while disturbances push it. Every number
and picture on the site is model output (engine runs, or the model's own optics and thermal functions). Nothing is
drawn by hand except the waveguide outlines, which the model does not have (section "Limits").

Everything lives in this folder. The generator reads the model (`mrm/`) and the experiment files (`experiments/`)
and writes only the data files (`site/data/`, or `runs/webviz_quick/` with `--quick`).

- `generate.py`: makes every data file from the model (one command, about 70 s).
- `derived.py`: the part of the generator that makes the statistics, the hardware limits and how near each run comes
  to them, the crosstalk tables, the scenario inputs and the plain scenario description (imported by generate.py).
- `site/`: the website (`index.html`, `style.css`, `app.js`, `js/*.js`). `site/data/`: the data files it reads.
- `package.json`, `scripts/`: `npm run dev` (the dev server), `npm run regen` (the data), `npm run check` (the checks).
- `make_colormaps.py`: writes `site/js/colormaps.js` (the inferno and coolwarm lookup tables, from matplotlib).
- `check_data.py`: re-checks the data against fresh engine runs (section "How the data was checked", item 6).

## Opening the site

```sh
cd ~/mrm-model/webviz
npm run dev
```

Then open the address it prints (http://localhost:8000) in Chrome or Safari (Firefox works too; WebGL must be on).
Nothing else is needed: `npm install` is not necessary, because the dev server (`scripts/dev-server.mjs`) uses only
Node's own modules (Node 18 or newer). It serves `site/` on 127.0.0.1 without caching and reloads the open page when a
file under `site/` changes (also after `npm run regen`). `PORT=8001 npm run dev` uses another port; if the port is
busy it takes the next free one. The page needs the internet once, for two pinned libraries from cdn.jsdelivr.net:
three.js 0.160.0 (the 3-D view) and plotly.js 2.27.0 (the 2-D views). There is no build step.

Without Node: `cd ~/mrm-model/webviz/site && python3 -m http.server 8000` also serves the site (no live reload).

The address keeps the view: `#v=parts&s=a1_aggressor&r=0&a=1&f=262&p=heater&c=err&m=height` opens the "Parts over
the run" view on the heater part with all rings overlaid (`a=1`), the A1 scenario, ring 0 selected, frame 262
(65.75 ms); `c` and `m` are the 3-D ring colour and map mode. Views (`v`): `scenario`, `die`, `parts`, `stats`,
`metrics`, `limits`, `xtalk`; parts (`p`, shared by the parts and statistics views): `ring`, `heater`, `pd`, `laser`,
`ctrl`, `data`. Keys: space plays and pauses, the arrow keys step one frame (shift: 10 frames).

## Regenerating the data

```sh
cd ~/mrm-model/webviz
npm run regen                                   # every data file, dated today
npm run regen -- --only a1_aggressor            # one scenario (or a comma list; also 'bistability')
npm run check                                   # check_data.py all: PASS or FAIL, about 2 minutes
```

`npm run regen` runs, from the repository root,
`.venv/bin/python tools/memguard.py -- .venv/bin/python webviz/generate.py --date <today> [arguments]`, and
`npm run check` runs `webviz/check_data.py all` the same way (`npm run check -- steady` runs one part).

- About 70 s on the M2 (warm numba cache), peak memory about 0.6 GB, 33 MB of JSON in `site/data/`.
- `--only steady,a1_aggressor,laser_step,cold_start,bistability`: regenerate some files (meta.json and checks.json
  are merged, not lost).
- `--quick --out <dir>`: short runs (a smoke test; the site does not read them). Without `--out` they go to
  `runs/webviz_quick/`.
- `--date` is written into meta.json; the data itself contains no clock times.
- The output is deterministic: the same commit gives the same files (fixed seeds, nominal world).
- The colour tables (`site/js/colormaps.js`) change only with matplotlib:
  `.venv/bin/python webviz/make_colormaps.py`.

## The scenarios

All four run on g26, nominal world (no sampled uncertainty, no device variation), 25 degC, bias 1.05 V, 50 Gb/s
data on every ring. Times are from the reset.

| id | controller | start | length | frames | key frames | what to look for |
|---|---|---|---|---|---|---|
| `steady` | pi_lock_c | warm (settled at the lock) | 40 ms | 400 x 0.1 ms | 68 | every ring held within 0.063 pm of its own operating point (about 125 pm red of its line) |
| `a1_aggressor` | pi_lock_c | warm | 135.7 ms | 542 x 0.25 ms | 88 | the A1 aggressor (peak 0.287 W, 30 um from ring 0) heats ring 0 most; near the hot cusp (65.35 ms) ring 0's heater sits at code 0 for 14.7 ms and the ring leaves its point by 455 pm; ring 4 by 14.7 pm, the others by 0.9-6.4 pm |
| `laser_step` | pi_lock_c | warm | 30 ms | 300 x 0.1 ms | 71 | +10 % light on every line at 10 ms: the rings heat themselves more and the lock settles about 20 pm further red of the point it held (it locks to a fraction of a peak it measured at start-up) |
| `cold_start` | wdm_supervisor | cold: heaters 0, lasers ramp on over 3 ms | 35 ms | 350 x 0.1 ms | 70 | each ring sweeps to map the lines, the supervisor assigns lines (a barrel shift: ring 2 on line 0, ring 0 on line 6) and locks the rings one by one in bus order, 11.8-22.9 ms |

Sources: `steady` and `a1_aggressor` are `experiments/examples/steady_pi_lock.yaml` and `a1_pi_lock.yaml` as they
are. `laser_step` is the steady file at 30 ms plus a `laser.power_delta_db` step of +0.414 dB on every line (built
as `tools/perf_scenarios.py` builds its laser step). `cold_start` is `experiments/acceptance/b5_cold_start.yaml` with
one world, the nominal world and a fixed 25 degC ambient instead of 8 sampled worlds over 0-105 degC, cut to 35 ms
(the 80 ms file locks every ring by 23 ms; the first 35 ms are the same run). wdm_supervisor was chosen because it
locks all 8 rings on lines it declares; acquire_then_lock without a supervisor locks rings on foreign lines
(docs/ENGINE.md, "Controllers on the engine").

Lock error. In the three pi_lock_c runs it is measured about each ring's own steady operating point (the experiment
files' `self_reference`: the median detuning over the last 5 ms of a 20 ms static pre-run), threshold 5 pm,
hysteresis 1 pm, from 5 ms on (pi_lock_c maps its peak in the first 3 ms). In the cold start it is measured about
the declared line, threshold 300 pm, any barrel-shifted assignment allowed (the B5 settings).

## The data files (`site/data/`)

Numbers are rounded to 4 significant digits (temperatures in K to 1 mK, detuning to 1 mK of pm, resonance
wavelengths to 9 digits, spectra to 0.01 dB). Missing values are `null`. Arrays named `[frame][ring]` have one row
per frame and one column per ring (ring 0 first on the bus).

### meta.json

Preset, git commit (and whether tracked files were modified), the date, and:

- `geometry` (um; x along the bus, y across, ring 0 at the origin): ring centres (50 um pitch), ring radius (the
  model's r_ref, 6.727 um), rib width, in-ring heater annulus (3.53-4.53 um), bus waveguide (y = -7.31 um, light
  enters from x < 0), drop waveguides to the PDs, monitor PD positions (ring + (0, 25) um), driver positions
  (ring + (0, -30) um), aggressor sites (`aggressor` (0, -30) um, the default; `aggressor_pd_adjacent` (0, 30) um,
  a stress site), the die temperature sensor `die0` (175, 1000) um, bus order.
- `lasers`: line wavelengths (line 0 is the reddest, 1318.64 nm; line 7 1302.60 nm), frequencies, ring_to_line.
- `rings`: FSR (10.19-10.44 nm), FWHM (340-344 pm), the cold resonances (7.6 nm blue of each ring's line).
- `scenarios`: id, title, one-line description, duration, frame count and length, key-frame count, file names.
- `spectra_grid`, `thermal_grid`, `units` (for every field), `limits` (short statements for the site).

### meta.json additions (device panels)

- `geometry.heater_structure`: the heater body of the preset's heater family (`inring_noundercut`: one doped-silicon
  annulus r = 3.53-4.53 um, z = 0-0.16 um, inside the rib) from `mrm.extraction.thermal_fem.ThermalConfig.heater_rects`
  at the model's r_ref. Assumed (the G26 heater is unpublished).
- `geometry.site_sizes_um`: the radii the FEM extraction gives the sites (PD 3 um, driver 5 um, aggressor 10 um);
  assumed; the plant uses point sites.
- `geometry.distances`: ring pitch, ring radius, ring to its PD and driver, ring and PD to each aggressor site, ring to
  the die sensor (centre to centre, from the preset positions); `geometry.provenance`: the preset provenance of each
  layout key (all positions `assumed`).
- `devices`: the realized plant's fixed settings (heater driver law, DAC bits and rails, full-scale voltage,
  compliance, R0, TCR, bank supply limits, DAC latency; monitor responsivity, dark current, noise, ADC bits, LSB,
  full scale, PGA gains, window, sample period, latency; controller period and samples per step; laser line powers,
  chip input transmission, RIN, drift; data rate, link transmission, receiver sensitivity), each with its preset
  provenance where the preset has the key.
- `thermal_coupling`: the DC gains of the thermal system (`mrm.plant.thermal.dc_gain`), K per mW from each of the 34
  sources into each ring, heater resistor and PD.

### scenario_<id>.json

- `t_s` [frame]: the END of each frame window. A frame is the window (t_s - frame_s, t_s].
- `rings`: per-ring traces, each [frame][ring], recorded by the engine (RecordSpec) every tick and reduced per
  window (`reducers` says how):
  - temperatures `t_ring_k` (mode-weighted ring), `t_heater_k` (heater resistor), `t_pd_k` (monitor PD),
    `t_base_k` (die base): window means, absolute K;
  - `detuning_pm` (resonance minus the declared line; positive = red) mean, `detuning_min_pm` / `detuning_max_pm`
    the envelope over the window, `lam_res_nm` mean;
  - `heater_code`, `monitor_code`: value at t_s; `p_heater_mw`, `p_abs_mw` (absorbed light, heats the ring),
    `p_in_mw` (light reaching the ring on its declared line), `p_drop_uw` (light into the monitor PD),
    `i_monitor_ua`: window means;
  - `reloma_db` (OMA / P_in at the ring's own through port) and `oracle_reloma_db` (the best stable relOMA at that
    instant): 10 log10 of the window mean of the linear values;
  - `line_on` (the line the ring sits on, -1 none), `declared_line`, `order`: value at t_s;
  - for the device panels: `heater_code_min` / `heater_code_max`, `monitor_code_min` / `monitor_code_max` (the
    envelope over the window), `ber` (instantaneous BER of the declared line), `oma_uw` (bus-output OMA), `bias_v`,
    `er_db`, `il_db` (window means).
- `lock_reference`: `d_ref_pm` per ring (the operating point the lock error is measured about) and its kind.
- `drives`: `aggressor_w` {site: [frame]} (window mean power into each aggressor site), `laser_power_factor`
  [frame][line] (chip power relative to nominal), `t_base_k` [frame].
- `events`: timeline markers taken from the data (aggressor power peak and the A1 target cusp, the laser step, the
  start of scoring, each ring's lock time in the cold start).
- `summary`: the engine's own metrics (every tick after `score_from_s`), per ring (`per_ring`) and worst case:
  max and rms lock error, OMA penalty, unlocked rings, rings on a wrong line, excursions, time at the heater rails,
  lock settings, aggressor calibration, warnings; also wrong-line entries, FSR hops, order changes, heater energy,
  peak demanded slew, mean and window BER, expected errors, mean absorbed power, and `saturation_counts` (clipped heater
  commands per ring over the whole run, from the engine's clip counter or, on the Python controller path, the
  ActionValidator).
- `key_frames`: the frame indices that have spectra and thermal maps.

### spectra_<id>.json

For each key frame and ring, the transmission of that ring alone on a window of +-1.5 nm, 241 points (12.5 pm
step): `wavelength = center_nm[frame][ring] + offset_nm[i]`.

- Window centre: the ring's declared laser line in `steady`, `a1_aggressor` and `laser_step` (so the offset is the
  wavelength minus the line, and the line sits at 0); the ring's resonance in `cold_start` (there the resonance
  starts several nm from any line).
- `thru0_db`, `thru1_db` [frame][ring][point]: the through port with the junction at NRZ level 0 (`v_lo_v`, 0.40 V
  reverse) and level 1 (`v_hi_v`, 1.70 V). The two curves are the two bit levels; their difference at the line is
  the modulation (OMA).
- `drop_db`: the drop port (to the monitor PD) averaged over the data (ones density x level 1 + the rest x level
  0). This is what the monitor, and so the lock, sees. The data-averaged through port is the same average of thru0
  and thru1 in linear units.
- `lam_res_nm` (the trace's resonance, the mean of the two levels), `resonance_level_nm` and `notch_level_db`
  [frame][ring][level] (the exact resonance and notch depth of each level: the 12.5 pm grid does not resolve the
  bottom of a deep notch, so draw the bottom from these), `line_nm` [frame][line] (every laser line),
  `levels` (bias levels, ones density, ring and heater temperatures used).
- `bus`: `p_in_mw` and `p_thru_mw` [frame][ring][line], the engine's own optics outputs at t_s (Truth `out_rl`,
  rows `RING_LINE_OUTPUT_ROW_INPUT_POWER` and `RING_LINE_OUTPUT_ROW_THROUGH_POWER`): the light of each line arriving at each ring after the bus segment, and leaving its
  through port (data-averaged). The site sums them over the lines for the bus light.

### thermal_<id>.json

For each key frame, `rise_k` [frame][y][x]: the temperature rise above the base temperature `t_base_k` on a
120 x 50 grid, x from -61 to 415 um (4 um step), y from -61.25 to 61.25 um (2.5 um step). Every ring centre, PD,
driver and aggressor site sits 1.6 um from its 4 nearest nodes, so all are drawn alike.

- `observers`: the engine's own observer temperatures at t_s (`ring_rise_k`, `heater_rise_k`, `pd_rise_k`,
  `die0_rise_k`) and the map's value at the exact ring centres and PDs (`map_at_ring_centre_k`, `map_at_pd_k`).
  Draw the rings with `ring_rise_k` (see "Limits").
- `display_hint_k`: the 99th percentile and the maximum of the map, and the largest engine observer value. The
  map peaks at point sources (the node next to the aggressor site reaches about 1670 K at the A1 peak; driver
  nodes about 35 K): clip the colour scale.

### bistability.json

Ring 0's self-heating hysteresis: drop power (and monitor code) against heater code, at 3 light levels.

- `powers[i]`: one sweep. `p_ring_dbm` (light at the ring: 0, +4, +8 dBm, set by the chip input loss 10, 6, 2 dB,
  the model's world range), `lines_on` (only line 0, so the loop is the ring's own; the last entry repeats +4 dBm
  with all 8 lines on), `fwhm_pm`.
- `static`: the S-curve from the model's static laws: `code`, `p_drop_uw`, `detuning_pm`, `t_ring_k`,
  `p_heater_mw`, `p_abs_mw`, `stable` (true where the heater power rises with the ring temperature, i.e. the
  self-heating loop gain is below 1; the middle branch is unstable and cannot be held open loop), `folds` (where the
  stable branches end).
- `sweep`: the engine run. `up` and `down` legs (t_ms, code, p_drop_uw, detuning_pm, monitor_code, t_ring_k, every
  20 us), the code profile (`t_knots_ms`, `code_knots`: 2 ms hold, 40 ms ramp up, 2 ms hold, 40 ms ramp down), and
  `jumps`: the largest step of the detuning on each leg (the up-sweep jump onto the hot branch, the down-sweep fall
  off it).
- At +8 dBm the monitor ADC saturates (code 65535) near the drop peak; the drop power is not clipped.

### devices_<id>.json

What the device panels, the bank table and the metrics section read. Same frames as the scenario file.

- Device rows: a read-only tap copies the kernel state of world 0 every 10 ticks (each controller step, at the
  step's edge, before that edge's command) and reduces it over each frame (mean unless stated):
  - `heater`: `cmd` (controller command after the channel clip), `reg` (latched register), `dac_code` (code at the
    DAC after its 1 us latency) at the frame end; `v_v`, `i_ma`, `r_ohm`, `p_mw` (the driver law's outputs);
    `step_max_codes` (largest register change between two consecutive controller steps); `cmd_minus_dac_max_abs`;
    `sat_count` (clipped commands, cumulative); bank `bank_p_mw_max`, `bank_i_ma_max`, `supply_p_frac_max`,
    `supply_i_frac_max` (sum of the 8 heaters against the supply limits) and `supply_scale_min` (the supply droop
    factor, 1 = no limit); `p_full_code_mw` (the driver law at code full scale and the frame's heater temperature);
  - `pd`: from the frame means of the records through the model's PD law and ADC transfer
    (`mrm.plant.sensing.MonitorHw.photocurrent_a`, `white_sigma_a`, `ideal_code`): `i_signal_ua`, `i_dark_na`,
    `noise_rms_na` and `noise_rms_codes` (white noise of one sample), `codes_per_ua`, `full_scale_ua`,
    `full_scale_drop_uw`, `ideal_code`; sampled: `i_dark_sampled_na`, `pga`, `samples_per_step` (ADC samples in the
    last release), `overrange_frac`;
  - `laser` [frame][line]: `f_dev_mhz` (frequency minus the nominal grid, with min/max), `p_mw` (module output),
    `p_chip_mw` (into the chip), `p_ppm_min` / `p_ppm_max` (sample spread about the frame mean: the RIN draw);
  - `datapath.ones_density`.
- `running`: the engine's own metrics (`mrm.sim.metrics.summarize`) of the accumulators copied at every frame end,
  i.e. the value of each metric from the scoring start to that frame (null before scoring starts), and `locked` /
  `unlocked_rings` (the engine's lock state with hysteresis at the frame end).
- `key`: at each key frame, `selfheat_loop_gain` [frame][ring] (Z_opt,self(0) x dP_abs/dT_ring, the phase model and
  exact ring at every line with the light at the ring held; below 1 the ring's own heating is stable there; negative
  on the red side of the line), `p_abs_phase_model_mw`, and `ring_rise_by_source_k` / `pd_rise_by_source_k`
  [frame][ring][source] (each observer's rise split by heat source: the sum of its engine lag states grouped by the
  lag's source, exact because an observer is the base plus the sum of its lags).

### Additions for the views (statistics, limits, crosstalk, scenario)

- scenario_<id>.json: `inputs` (every scenario input signal evaluated every 10 µs with the scenario's own evaluator,
  frame mean / min / max, or one `constant` value; `active` when the scenario drives it; the model's doc string),
  `controller_timing` (ticks of the controller, ADC and DAC), `plain_description` (one paragraph written from the run's
  numbers), `summary.warm_start`.
- spectra_<id>.json `bus`: `p_drop_uw` and `p_abs_uw` [frame][ring][line], the engine's own drop-port and absorbed
  power of every line in every ring at the key frame (Truth `out_rl` RING_LINE_OUTPUT_ROW_CLOCKWISE_DROP_POWER, RING_LINE_OUTPUT_ROW_ABSORBED_POWER): the optical crosstalk.
- devices_<id>.json `xt`: `ring_k` and `pd_k` [frame][ring][group], the rise above the base split into own heater, own
  light, own junction + driver, other heaters, other light, other junctions + drivers and aggressor: the engine's lag
  states of each ring and PD observer, grouped by source, read at every controller edge (10 µs) and averaged per
  frame; `ring_by_source_mean_k` / `pd_by_source_mean_k` (whole run and scored, every source).
- meta.json: `devices.timing`; `thermal_coupling.full` (every observer × every source: DC gain K/mW, the lags
  [tau_s, residue K/mW] of each pair, the 10 / 63 / 90 % step-response times); `hardware_limits.rows` (group, label,
  value, unit, preset key, provenance, note).

### stats_<id>.json

From the engine's 10 µs series: the run carries extra RecordSpec records at 10 ticks (window means over every 1 µs
tick; min / max reducers for the detuning and the monitor code, so their extremes are tick extremes), and the device
tap's kernel state at each controller edge (heater register, V, I, R, P, command, DAC code, dark current, laser
lines, ones density, the ADC over-range flag, the supply scale).

- `quantities`: per part, `run` and `scored` statistics (min, p5, p50, mean, p95, max, rms, std) per ring (or per
  line, or for the bank), and `*_pooled` (every ring's samples together). relOMA is computed on linear values and
  given in dB (no rms / std); the BER has no rms / std.
- `states`: ms in each state per ring (10 µs samples; with the number of starts), and `engine_scored` where the engine
  has its own per-tick metric (time at a rail, on a foreign line); locked and unlocked come from the engine.
- `counts`: engine counts (excursions, foreign-line entries, FSR hops, order changes, BER windows), clipped heater
  commands, ADC over-range samples.
- `limits`: for each hardware limit the margin (distance from the limit in its unit) at every 10 µs sample (key frames
  for the loop gain): `margin_min` per ring and its time, `time_at_ms`, `reached`, `first_ms`, for `run` and `scored`.
- `junction`: the junction voltage of NRZ level 0 and 1 (Truth) and the resonance shift between them (phase model),
  medians over the key frames.

### checks.json

The numbers of the next section, regenerated with the data.

## The site: what each view shows

The top bar has the navigation between the seven views, the scenario, the emphasized ring (R0-R7), the All 8
toggle on the views that compare rings, play / speed and the time scrubber with the event marks. Everything follows
the scrubber except the statistics, the limits tables, the bistability plot and the static crosstalk tables. Only the
open view is drawn (Plotly needs a visible box), so switching views redraws it once. Colour ranges are fixed per
scenario, so a colour means the same number all through a run.

**Two clocks.** A *frame* is a window of 0.1 ms (0.25 ms in A1); the traces are window means (codes: the value at
the end). A *key frame* is the instantaneous state at the end of a frame: 68 to 88 per run, every 1 to 20 frames
(denser near events). The spectra, the temperature map, the bus light, the self-heating loop gain and the per-line
optical split exist only at key frames, so those views show the key frame nearest the scrubber and say which one.
The statistics and the limits use a third basis: 10 µs samples, one per controller step (see stats_<id>.json).

### 1. Scenario: what does this run test?

- A plain paragraph written by generate.py from the run's own numbers (`plain_description`: for A1 the aggressor's
  distance, period and peak, when the heater of the ring nearest the aggressor (ring 0 in g26) reaches code 0 and for
  how long, the largest excursions), and the facts of the run (controller, warm or cold start, length, frames,
  scoring start, lock band and reference, world, driven inputs).
- *Disturbance inputs over the run*: one panel per input on one time axis: aggressor power per site (with its
  distance to the nearest ring), die base temperature (ambient or soak; processor heat events when driven), laser
  light into the chip per line (the plant's laser state, so it also shows power steps and the cold start's 3 ms
  turn-on ramp), laser frequency offset per line (drift; 0 in the nominal world), data traffic (rings sending data)
  and ones density, faults (rings with each fault active) and the junction bias set point. Event lines are dotted;
  the grey start is not scored. Zoom (drag) below 1.5 ms and the controller steps appear as thin lines.
- *When does the controller act?*: three controller steps drawn from the model's timing settings (meta
  `devices.timing`): ADC integration windows, the time each code is ready, the controller steps and the heater DAC
  output.
- *Events* (click to move the scrubber) and *Every scenario input*: every input signal of the plant
  (`mrm.scenarios.generators.SIGNALS`) with a plain description, whether the scenario drives it, and its value or
  range in this run.
- *What the model does not show* (the model limits) and the provenance line.

### 2. 3-D die: what is happening on the die right now?

The 3-D view, with the bank overview, the resonance spectrum and the temperature map of the scrubber frame below
it.

**3-D view: what is happening on the die right now?** The die as the model places it (x along the bus, y across
it; ring 0 at the origin). Drag to orbit, scroll or pinch to zoom, hover anything for its numbers, click a ring to
select it. *Reset view* and *Top view* put the camera back.

- Rings: radius to scale (6.73 um, the model's r_ref) and 50 um apart. The rib is 0.4 um wide; it is drawn 1.2 um
  wide (3x) so it shows at full view, and the bus and drop waveguides likewise. The colour is selectable:
  - *temperature*: the ring's own lumped temperature (`t_ring_k`), on the shared temperature scale;
  - *lock error*: detuning minus the lock reference (`d_ref_pm`: each ring's own operating point in the pi_lock_c
    runs, the declared line in the cold start). The scale is the lock band (±5 pm, or ±300 pm in the cold start):
    grey is on the reference, full blue or red is at the edge of the band or beyond it (clipped);
  - *detuning*: resonance minus the declared line, symmetric about 0, red = the ring is red of its line.
  The label under each ring gives the line it sits on (`line_on`), or *off-line*.
- Heater: the annulus inside each ring is the heater body of the preset's heater family, to scale
  (`geometry.heater_structure`: in-ring doped silicon, r = 3.53-4.53 um, 0.16 um thick, the FEM extraction's assumed
  placement); its colour is the heater-resistor temperature on the shared scale. The plant treats it as a lumped
  source. The copper column on it is a power gauge, not a structure: 30 um tall = the scale maximum in the legend
  (20 mW in the warm runs, 50 mW in the cold start). Hover the annulus for code, power, V, I and R.
- Monitor PDs (discs 25 um from each ring centre on the drop side, at the end of the drop waveguide): radius 3 um
  (the FEM extraction's assumed PD site), colour = PD temperature on the shared scale.
- Drivers (grey discs 30 um below each ring, radius 5 um, the extraction's site): heat sources of the junction
  drivers. No colour.
- Aggressor: the red disc of radius 10 um (the extraction's site) at (0, -30) um, the A1/A2 default site (the same
  point as ring 0's driver). Its glow grows with its power (largest in the legend, 0.287 W in A1); it reads *off* in
  the runs without one.
- Scale bars of 10 and 50 um. Under the view: four limit notes (waveguides drawn 3x wide; the map is the lumped
  point-site law; assumed layout; assumed site sizes) and a distances table for the selected ring (pitch, radius,
  ring to its PD and driver, ring and PD to the aggressor and stress sites, ring to the die sensor), each with its
  preset provenance.
- Bus light: the bus is cut into 9 segments (input, between each pair of rings, output). The brightness of each is
  the total light of all 8 lines in it (the engine's `p_in` of the next ring, summed over the lines; after the last
  ring, its `p_thru`), from 0 to the scenario's largest value rounded up to 1 mW. In the warm runs each locked ring
  passes only 21-30 % of its own line (it sits on the slope of its notch), so the bus dims from about 20 mW at the
  input to about 5 mW at the output (about 9 mW in the cold start, whose lock sits further from the notch).
- Temperature map: *flat* colours the die surface; *height* draws it as a surface under a glass die, deeper = colder
  (the same clipped colour fraction sets both); *off* hides it. The map is the point-site law (see "Thermal maps"
  and "Limits"): exact at the PDs, low inside the rings, and it spikes next to point sources, so the colour is
  clipped at the top of the scale (the legend says where).

**Shared temperature scale.** inferno, in degC, from the base temperature (25 degC) to the hottest ring temperature
of the scenario rounded up to 10 degC (140 degC steady and laser step, 150 degC A1, 220 degC cold start). The 3-D
map, the 3-D PDs, the 3-D rings in temperature mode and the 2-D heat map all use it. In the cold start the rings
reach 213 degC during the mapping sweeps, so the die (about 30-50 degC) sits in the dark end.

**Bank overview: how is each ring now?** One row per ring at the scrubber frame: declared line and the line it sits
on (with "foreign line" / "off-line"), the engine lock state, detuning, lock error ("out" beyond the band),
ring and PD temperature, heater code ("rail" at 0 or full scale), heater power, monitor code ("clipped" at the ADC
full scale), relOMA, BER, and the self-heating loop gain at the nearest key frame. The last column is a sparkline of
one chosen quantity over the whole run (lock error, detuning, ring temperature, heater code or power, monitor code,
relOMA) with the scrubber as a line. Click a row to select the ring.

**Where is every ring’s resonance relative to its laser line?** All 8 shows one panel per ring at the same key
frame, on a common transmission scale. Toggle it off for the selected ring alone. Each panel shows the through port
with the junction at each data level (bit 0: 0.40 V reverse, bit 1: 1.70 V), and the drop port averaged over the
data (what the monitor sees). The notch bottoms come from the exact resonance and depth
(`resonance_level_nm`, `notch_level_db`), merged into the 12.5 pm grid; a notch below -50 dB is written at the
bottom. The solid vertical line is the ring's own laser line, the dotted one the resonance; the readout gives the
detuning. The strip above is the whole comb: every laser line (L0 the reddest) and every ring's resonance, with the
window of the plot shaded. Two resonances can sit a few pm apart (in the cold start's sweeps); then only one of them
is labelled (the selected ring first), and hover names each. In the cold start the window follows the resonance, so
the ring's own line is often outside it (an arrow at the edge gives the distance).

**How hot is the die?** The temperature map from above (4 x 2.5 um grid, smoothed for display) on the shared scale,
at the nearest key frame. Each ring disc is filled with the ring's own engine temperature (hover: both the ring
and the map value at its centre, which reads 2-9 K low); the squares are the PDs (hover: their temperature, equal to
the map there), the circle the aggressor site with its power. The marks on the map are light in both themes (the
map's colours do not change with the theme). On a narrow screen only the selected ring is labelled and the
aggressor's power is in its hover.

### 3. Parts over the run

One screen per part, chosen with the tabs. By default, All 8 shows a live snapshot table with a column per ring and
time plots with eight ring traces. Ring buttons change the emphasis without hiding the other rings. Secondary signals
such as oracle modulation, signal photocurrent and applied DAC codes have separate comparison rows. Toggle All 8 off
for selected-ring tiles, frame envelope bands and overlaid secondary signals. Click a plot to move the scrubber; use
the legend to isolate rings.

- *Ring*: true detuning (with the lock band), lock error, ring temperature, absorbed light, relOMA against the
  oracle, OMA lost to the oracle, the self-heating branch (+1 red of the line, −1 blue) and the self-heating loop
  gain (key frames). Below it the bistability card:

  **Why is the ring bistable?** Not tied to the scrubber. Ring 0 swept open loop: heater code ramped up for 40 ms
then down for 40 ms (the engine, blue and orange), over the static S-curve of the model's laws (grey: stable,
dotted: the unstable middle branch), with the two folds (diamonds) where a stable branch ends. Rising, the ring
stays on the cold branch past the hot branch's start, then jumps up at the cold fold; falling, it stays on the hot
branch until that fold, then falls off. The buttons change the light at the ring (0, +4, +8 dBm with only line 0
on, and +4 dBm with all 8 lines); the loop widens with the light. The vertical axis can be the drop power (what the
monitor sees) or the detuning.
- *Heater*: register code with the rails, headroom to the nearest rail, voltage (full scale), current, resistance
  (TCR), power (and the power at full code), heater temperature, the largest code step per controller step and the
  bank supply use (power and current against the limits).
- *Photodiode and ADC*: drop power, PD current (total and signal), dark current, PD temperature, ADC code (band:
  lowest to highest code in the frame) with full scale, over-range samples and the white noise of one sample.
- *Laser, WDM and bus*: light into the chip and frequency offset of the 8 lines, own-line light at each ring, the
  declared line, the line the ring sits on and its status (own, foreign, none); below: the light along the bus over
  the run (total of the 8 lines in each bus segment at the key frames), the whole comb and the power of each line
  at each ring at the nearest key frame, and the line assignment table.
- *Controller I/O*: monitor code (the controller's only input) against the Truth detuning and lock error, command /
  register / DAC code, |command − DAC code|, the largest register step, the timing chain, and the scatter of monitor
  code against true detuning.
- *Data path*: ones density, BER (log) with the limit, OMA at the bus output and at the receiver (with the receiver
  sensitivity), extinction ratio, insertion loss, junction bias, expected errors so far.

### 4. Statistics

Numbers of the whole run or of the scored time (buttons), from stats_<id>.json, for the parts Ring, Heater,
Photodiode and ADC, Laser lines, Controller I/O and Data path: min, 5 %, median, mean, 95 %, max, rms and spread
(standard deviation), with unit and a one-line explanation. One ring: one row per quantity and one column per
statistic. All 8 (the default): one column per ring plus "all rings" (every ring's samples together) for the
statistic chosen at the top right; the laser part always has one column per line. Below: *time in each state* in ms per ring and for
the bank (engine lock state, unlocked after the first lock, lock error outside the band, heater at code 0 or full
scale, ADC clipped / over-range / at 0, on a foreign line, on no line, BER above the limit, bank supply limited),
with the number of starts, and the engine's own per-tick value next to the 10 µs count where the engine has one;
and *counts* (excursions, longest excursion, foreign-line entries, FSR hops, order changes, BER windows, clipped
heater commands, ADC over-range samples).

### 5. Metrics

**Metrics: how good is the lock, ring by ring?** The main optimisation metrics, one row each, with the goal
(minimise / maximise / must be 0 / reference), a one-sentence definition, the unit, the value per ring and the worst
ring: OMA penalty vs the oracle; rings on a foreign line at the end, time on a foreign line, foreign-line entries;
lock excursions and the longest one; time unlocked (after the first lock); unlocked rings at the end; max and RMS
lock error; time to lock (from the scoring start); mean relOMA and the oracle relOMA; heater energy and mean heater
power; heater headroom (codes left before a rail); time at code 0 and at full scale; clipped heater commands; peak
demanded slew; mean BER and expected errors; FSR hops; resonance order changes; mean absorbed light. All are the
engine's own metrics (`mrm/sim/metrics.py`, every 1 us tick after scoring starts), except the clipped commands (the
engine's clip counter, whole run) and two simple ratios named as such (mean heater power = energy / scored time;
headroom from the code range used). "must be 0" metrics read "≠ 0" when they fail. Click a metric to plot it over the
run: the same accumulators copied at every frame end (all rings, the selected ring thick). In the laser step every
ring is "out of band" and unlocked at the end: the lock settles about 20 pm red of the operating point it held before
the step, and the band (±5 pm) is measured about that old point. In the cold start scoring starts at 0 ms, so the max
lock error includes the cold rings, about 5 nm from their lines.

*BER: how many bits does each ring get wrong?* The definition (mrm/plant/datapath.py: OMA_rx = OMA_bus × link
transmission 0.501, Q = Q_ref × OMA_rx / OMA_sens with Q_ref = 5.998 and OMA_sens = 52.5 µW, BER = ½ erfc(Q/√2), every
1 µs tick; expected errors = Σ BER × bits; mean BER = expected errors / bits, mrm/sim/metrics.py), why the 2 s window
metrics (windows completed, windows above 1e-9, largest window BER) read 0 here (the runs last 30-136 ms, so no 2 s
window completes: 0 means "not measured"), a table per ring (mean BER, expected errors, bits, median and highest
10 µs BER, time above the limit, the three window metrics) and the frame BER of every ring over the run (log scale).

### 6. Hardware limits

- *Which hardware limits does this scenario reach?*: for each limit (heater code at 0 and at full scale, heater
  power at full code, driver compliance, bank supply power and current, ADC full scale and code 0, lock band, BER
  limit, self-heating loop gain 1, bias DAC low and high end) the closest approach (margin, worst ring, time), which
  rings reached it, the time at the limit and the first time it was reached, for the whole run or the scored time.
  Click a row to plot the margin of every ring over the run (frame values; 0 = at the limit). In A1, ring 0's heater
  reaches code 0 near the aggressor peak (REACHED, 14.7 ms at the rail in the scored time).
- *Hardware limits the model assumes* (meta `hardware_limits`): heater DAC code range and bits, full-scale voltage,
  the maximum power per heater (the driver law at full code at 25 °C), resistance and TCR, compliance, bank supply
  limits, power for one FSR, DAC latency; bias DAC range and bits, bias set point, data swing, the junction levels V0
  and V1 and the resonance shift between them (model output of the steady run), junction efficiency; PD
  responsivity and dark current, ADC bits and full scale, PGA gains, input noise; the ADC window, sample period and
  latency, the controller period and latency, the DAC latency; laser power per line, chip input loss, bus loss,
  RIN; bit rate, receiver sensitivity, link transmission, BER limit and counting window. Each has its value, unit,
  preset key and provenance tag.
- *All fixed device settings*: the table of the realized plant's settings.

### 7. Crosstalk

- *Who heats whom?*: the DC gain of the model's thermal system from every heat source (each ring's heater, absorbed
  light, junction and driver; the two aggressor sites) to every observer (ring, heater resistor, monitor PD, die
  sensor), K/mW, log colour; choose the observer and source groups.
- *How fast does the heat arrive?*: for the selected ring's ring, heater resistor or PD (or the die sensor) the 12
  strongest sources with their DC gain, poles and residues, and the 10 / 63 / 90 % times of the step response, and
  the step responses of the 6 strongest (log-log). All 8 also compares the response of each ring, heater or PD to a
  step in its own ring’s heater.
- *How much of each ring's temperature is crosstalk?*: an eight-ring snapshot table and one stacked-area panel per
  ring (or PD), sharing a temperature scale. Toggle All 8 off for selected-ring tiles and one area panel. Layers split
  the rise into own heater, own light, own junction + driver, other heaters, other rings' light, other junctions +
  drivers and the aggressor. Below: the crosstalk share (other rings + aggressor, % of the rise) of every ring over
  the run, a table of the mean split per ring (whole run or scored), and the key-frame matrix of who heats
  whom.
- *Optical crosstalk between the lines*: what the model has (each ring's drop and loss take light at every line: the
  other lines reach its monitor PD and heat it; each ring takes light of every line from the bus) and what it does
  not have (no data crosstalk between channels: the BER of a line uses its bus-output OMA with every other ring's
  data averaged; no backscatter; no coherent crosstalk); the power of each line in each ring's monitor PD at the
  nearest key frame, and the foreign-line share of each ring's monitor light and absorbed light over the key frames.

**Light and dark.** The page follows the system setting; the *Theme* button cycles auto, light, dark (remembered in
the browser).

## How each picture is computed

**Engine runs.** `generate.py` runs each scenario through `mrm.sim.experiment`'s own case pipeline (`_plan`,
`_execute`: the code path of `mrm run <file>`), with its own record list. Two read-only taps sit in a subclass of
the engine's Plant: every 10 ticks they read the thermal source powers the engine used (`th_u1`), and at each key
frame they copy Truth (temperatures, bias levels, line wavelengths, optics outputs, every thermal observer). On the
compiled path the taps cut each kernel block into 10-tick blocks that end on controller edges; under the fixed
substep policy that changes nothing (checked below).

**Spectra.** The model's phase model (`mrm.plant.phase.ring_optics_row`: thermo-optic shift at the ring temperature,
junction shift at each bias level, loss at the ring and heater temperatures, couplings at each wavelength) and its
exact add-drop ring (`mrm.plant.optics.ring_exact`), evaluated at the ring's state at the key frame. No new
physics and no other constants.

**Thermal maps.** The model has no temperature field: it is lumped. The engine's thermal system has one observer
per ring, heater resistor, PD and the die sensor; every path from a source to a point site (a PD, the die sensor,
any aggressor path) follows the interim site law: residue c(d) Z0 with c(d) = clip(0.03 x 60 um / d, 0.015, 1) and
one pole tau(d) from the imec crosstalk table. `generate.py` captures the InterimRom and ThermalScales that
`mrm.params.realize` builds for the run (through realize's own `thermal_hook`), adds one free temperature-sensor
site per grid node, and lets `mrm.plant.thermal.assemble` build the system: each node gets a lag from each of the
34 sources (8 heaters, 8 absorbed-light sources, 8 junctions, 8 drivers, 2 aggressor sites). The engine's own
observers keep exactly their lags in this larger system (checked). The lags are stepped by the model's own kernel
(`thermal_step_b`) from the reset state with the source powers the engine used, read every 10 us: the aggressor
power linear in between (as the engine's waveform), the other sources held at the value read at the end of each
interval (the engine holds them per tick and changes them at controller edges). The map is dynamic, not
quasi-static: each node lags its sources as the law says. A settled start begins at the steady state of the reset
inputs; the cold start at zero rise.

**Bistability.** Static: for ring temperatures from 75 K below to 20 K above the lock, the heater power that holds
that temperature follows from the DC gains of the thermal system (other sources held at the warm start), the
absorbed light and drop from the phase model and exact ring at every line that is on (all orders), and the code
from the heater driver law with its TCR at the heater-resistor temperature (`runner._code_for_power`). Engine: a
Python open-loop controller ramps ring 0's code (the other rings hold their warm-start codes) from 0.6 loop widths
below the hot-branch fold to 0.6 loop widths above the cold-branch fold and back.

## How the data was checked

Numbers from checks.json for commit ee381af, generated 2026-10-17. Regenerated on 2026-10-21 at commit 0738712 (working
tree modified): items 1-5 give the same numbers as on 2026-10-17. Regenerated again on 2026-10-23 at commit 0738712
(working tree modified, after the readability pass): checks.json gives the same numbers, except the wall times; the
scenario files changed only in the text of the input descriptions.

1. **The taps change nothing.** Each scenario ran twice, without and with the taps: every metric and every
   recorded trace is bit-identical in all four. The metrics also equal those of `mrm.run(<experiment file>,
   write=False)` (steady, a1_aggressor, cold_start with its overrides; laser_step has no file).
2. **Spectra (check a).** At every key frame and ring (297 frames x 8 rings):
   - the exact resonance of the two levels (their mean) equals the trace's `lam_res` to 5e-10 pm;
   - the minimum of the data-averaged through spectrum lies within 0.56 grid steps (7 pm) of `lam_res`;
   - notch depth: the exact value is in `notch_level_db` (-31 to -65 dB at the lock, down to -99 dB in the cold
     start). A notch that deep has a bottom only a few pm wide, which a 12.5 pm grid cannot hit: the grid minimum is
     0.00002 to 31 dB shallower than the exact value (up to 68 dB in the cold start);
   - the same functions at the laser line reproduce the engine's own outputs at that instant: through power to
     2.6e-7 (relative), relOMA to 1.0e-7 (absolute), and the monitor's optical input (drop summed over all 8
     lines and orders) to 6.6e-8 (relative). In the cold start these are 1.7e-4, 2.9e-5 and 3.6e-5: there a ring
     can change temperature by up to about 140 K within one 0.1 ms frame during the mapping sweeps, and the engine
     evaluates loss and coupling at the previous tick's temperature (docs/ENGINE.md, "What happens in one tick")
     while the spectra use the present one.
3. **Thermal maps (check b).**
   - The extended system holds the engine's 970 lags unchanged (same sources, poles and residues).
   - Reconstructed against the engine at every key frame: PD observers within 0.32-0.44 mK (warm runs) and 32 mK
     (cold start, where the PD rises up to 18.7 K); the die sensor within 0.12 mK (0.95 mK cold); the ring observers
     within 2-18 mK (warm) and 1.1 K (cold start: the ring self poles are 0.5-13 us, faster than the 10 us reading).
   - The map at a PD position equals the engine's PD temperature to the same 0.3-0.4 mK: the PDs follow the same
     point law.
   - The map at a ring centre is NOT the ring temperature: it reads 2.1-9.2 K below the engine's ring (rise about
     109-116 K) in the warm runs, and -19 to +54 K off in the cold start (the ring's own poles are 0.5-13 us, the
     point law's 0.1 ms and longer). The ring is its own lumped observer: its light heats it 1.76 times more per mW
     than its heater, and its self path has three poles; the point law has neither. Draw rings with `ring_rise_k`.
4. **The arrays reproduce the summary metrics (check c)**, using only the rounded exported arrays:
   - max lock error from the detuning envelopes: within 0.005 pm of the metric for every ring and scenario
     (a1_aggressor: 454.86 pm for ring 0 from both);
   - OMA penalty from the window means of relOMA and oracle relOMA: within 0.0004 dB;
   - frames cover the scored time completely, except a1_aggressor, whose frames end at 135.5 ms (the last 0.2 ms
     of the 135.7 ms run fall short of a whole 0.25 ms frame);
   - rms lock error from frame means matches where the error is slow (a1: 66.72 against 66.69 pm for ring 0;
     laser_step: 17.62 against 17.62 pm) and is below the metric where the error is tick-to-tick noise (steady: 0.006
     against 0.016 pm): a window mean removes the noise.
5. **Bistability.** The engine's jumps fall at the static folds: up-sweep jump 12, 38, 78 codes after the cold-branch
   fold and down-sweep fall 7, 18, 56 codes before the hot-branch fold at 0, +4, +8 dBm (loop widths 444, 2626,
   9822 codes). The lag grows with the loop width (a faster ramp in degC per ms against the 3 ms package tail).
   The static hot-branch folds sit 32, 12 and 5 pm blue of the line. docs/EXPLAINED.md of commit ee381af (section
   2.4, "Self-heating") gives about 9 pm at +4 dBm, measured from the drop peak.
6. **Independent re-check** (`check_data.py`, a second script that does not reuse generate.py's code paths; review
   of 2026-10-17). Every difference is at the rounding of the exported numbers:
   - fresh `mrm.run` of the steady and A1 files: every per-ring metric equals the exported summary within its
     5-digit rounding (largest relative difference 4e-5; A1 ring 0 454.86 pm, 4 excursions in both);
   - fresh runs cut at a key frame (steady 40 ms, A1 66.0 ms near the hot cusp, cold start 6.0 ms during the
     sweeps): ring and heater temperatures within 0.5 mK, resonance within 0.005 pm, observers within 5 mK (rings)
     and 0.5 mK (PDs);
   - spectra recomputed from the phase model's rows with a closed-form add-drop ring (not `ring_exact`): curves
     within 0.009 dB (above -40 dB), drop within 0.005 dB, each level's resonance within 0.005 pm, notch depth from
     a 1 fm scan within 0.005 dB of `notch_level_db`;
   - the steady map at 40 ms against the site law evaluated directly (sum over the 34 sources of
     clip(0.03 x 60 um / d, 0.015, 1) x Z0 x P, with the engine's powers): within 0.042 K everywhere (median
     2.5 mK); interpolated at the PDs it is within 8 mK of the engine's PDs (with y flipped, 9 K off).

7. **Device data** (`check_data.py`, in `steady` at 40 ms and `cold_key` at 6 ms; review of 2026-10-21): the heater
   codes at the frame end equal the fresh kernel state exactly; R = V / I; the voltage follows code / code_max x
   V_FS and the resistance the TCR law within 0.1 %; in the steady run the frame means of V, I and R equal the end
   state within 0.1 %; dark current, signal current, the noise of one sample and the ADC full scale follow the laws
   written in the checker (I0 exp(-Ea/k (1/T - 1/T0)), responsivity x drop, sqrt((2 q I + S_tia) / (2 T_int)),
   code_max x LSB / gain) within 2e-3; the running metrics at that frame equal the fresh run's engine metrics within
   5e-5; the heat-source split equals the fresh lag states; the self-heating loop gain equals a recomputation with a
   closed-form ring (loss = 1 - through - drop) within 0.01. `npm run check` ends with PASS (107 checks) or FAIL.
   generate.py also writes to checks.json, per scenario: the heat-source split sums to the engine's ring observers
   within 4e-13 K, P_abs from the phase model matches the engine's within 2.5e-8 (warm) and 2e-5 (cold start), and
   the running metrics at the last frame equal the run's summary.

8. **Numbers added for the views** (`check_data.py derived`, file-only for every scenario, plus the fresh steady
   run): the DC gain of every observer/source pair equals the sum of its exported residues and the full
   table equals ring_k_per_mw; t63 recomputed by bisection from the exported lags; the maximum heater power equals
   V_FS² / R0; the detuning extremes of the statistics equal the frame envelopes (both tick extremes); the scored max
   |lock error| of the statistics equals the engine metric; the run mean ring temperature equals the mean of the
   frame means; the time at each heater rail (10 µs samples) and the 'heater at the low rail' limit equal the engine's
   heater_low_rail_s / heater_sat_s; the rail margins equal the engine's code range used (within 2 codes: one step at
   the window edges); the scored mean BER equals the engine's ber_mean; the per-frame crosstalk split plus the base
   equals the ring temperature (warm runs; in the cold start the 10 µs edge samples differ from the tick mean by up to
   about 8 K during the sweeps, so it is only printed); the scenario inputs reproduce drives.aggressor_w and
   laser_power_factor; V1 − V0 = the swing and (V0 + V1) / 2 = the bias; fresh steady run: the DC gains of its thermal
   system, the per-line drop and absorbed power at 40 ms, and that the per-line drop sums to the PD's drop power.
   In the steady run at 40 ms the foreign lines give 3.8-8.3 % of each ring's monitor light.
   generate.py also writes into checks.json (`stats`, `crosstalk`): the 10 µs envelope max |lock error| minus the
   metric (1e-14 pm), the rail time minus the engine's, the 10 µs mean BER against the engine's, and the crosstalk
   split against the ring temperature.

## Limits

- Every number is the g26 preset's nominal world. Many inputs are assumed, inferred or taken from imec and G25
  devices, not measured on a G26 die (docs/EXPLAINED.md, docs/PARAMETERS.md). The heat dynamics come from imec
  measurements and assumptions (a natural-convection package); G26's own time constants and crosstalk are unknown.
- The temperature map uses the model's lumped point-site law, not a 3-D field solve. It is exact at the monitor PDs
  and the die sensor, and wrong near and inside a ring (it under-reads the ring by 2-9 K at its centre and much
  more at its waveguide, about 7 um out, where the 1/d law gives about 40 K against the ring's 110 K). It has a
  sharp clipped peak at each point source.
- The plant has no layout beyond site positions. The bus and drop waveguides, the coupling gaps and the heater
  annulus are drawn from the model's extraction inputs (assumed values). Positions of PDs, drivers and the
  aggressor are assumed; the default aggressor site and ring 0's driver site are the same point (the preset notes
  this).
- Spectra use the rigid-comb phase model: a heated ring keeps its cold FSR, so the orders next to the locked one
  are off by up to about 33 pm at the lock (the locked order is exact). No backscatter (g26).
- The spectra show each ring alone. On the bus, ring k sees the lines that rings 0..k-1 let through (0.59 % bus
  loss across the bank); `p_in_mw` carries that.
- Key frames are the instantaneous state at the frame's end; the traces are window means.
- In A1, ring 0's large excursion is a limit of the plant with the present numbers (heater authority), not of the
  controller (docs/KNOWN_ISSUES.md, "Ring 0 cannot reduce heater power far enough at the hot end of A1 and A2").
- Site: waveguides are drawn 3x their width; the PD, driver and aggressor discs use the FEM extraction's assumed site
  radii (the plant uses points); the heater annulus is the extraction's assumed geometry; the heater column is a
  power gauge, not a structure. The spectra, map, bus light, loop gain and heat-source split are at the nearest key
  frame, not interpolated. The device rows are 10 µs samples (each controller step), not every tick.
- The laser lines have no drift in the nominal world (drift rates 0); the frequency offset reads 0 and the power
  spread is the per-tick RIN draw (RIN -145 dB/Hz, assumed).
- The page loads three.js and plotly.js from cdn.jsdelivr.net (pinned versions); offline, the views do not start.
- Statistics and limits use 10 µs samples (one per controller step): the rms and the percentiles are of 10 µs window
  means, so they hide some tick-to-tick noise; the extremes of the detuning, lock error and monitor code are tick
  extremes. The 'time in a state' counts 10 µs samples (no hysteresis); the engine's per-tick values are given next to
  them where the engine has one.
- The crosstalk split over the run reads the lag states at the controller edges, not every tick (exact at those
  instants; within 0.05 K of the tick-mean ring temperature in the warm runs).
- Optical crosstalk: the model has the other lines' light in each ring's drop and loss, and the bus attenuation by
  every ring; it has no data crosstalk between channels (the bus-output OMA of a line averages every other ring's
  data), no backscatter (g26) and no coherent crosstalk.
- The plain scenario descriptions are written by generate.py from templates per scenario; their numbers come from the
  run.

### Reading the plot headings

The quantity and unit of each y-axis are written horizontally above the plot area. Every time plot states the
question that it answers and describes its traces in a few words: the 44 Parts panels, the scenario inputs, the
selected metric, the BER, the limit margins and the thermal source split. The metric and limit explanations follow
the selected row. The standalone spectra, maps, bus and crosstalk charts have a question and a reading guide next to
them. The bank sparklines explain the selected quantity and say that each has its own vertical scale. "Noise" is the
expected RMS fluctuation of one ADC sample, in codes; it is not the full code spread in a frame. The captions wrap
with the window width; the plots keep their shared time axis, zoom and scrubber cursor.

### How does the website render math?

The page renders LaTeX locally with the bundled KaTeX 0.19.0 renderer and fonts, so the math needs no network.
Equations, symbols, stated physical quantities and plot tick labels are typeset; plain ADC codes and table values keep
aligned digits. The interface uses the locally installed SF Mono font. Math uses the KaTeX typewriter style
(`\mathtt`) for letters, digits and upright units, and the normal math fonts for operators, Greek symbols and
stretchy delimiters. The fonts are chosen through the renderer, not by CSS on the glyphs, so fractions and scripts
keep the renderer's font metrics. Each formula is scaled by 1.23 to match the visible lowercase height of SF Mono
(SF Mono: 1083/2048 em; KaTeX Typewriter: 0.43056 em). Inline math, plot labels and the hidden plot-label measurement
use the same scale; superscripts and fractions keep their normal relative sizes.

`site/js/math-text.js` handles explicit `\( ... \)` expressions and a small dictionary of the physical notation that
the site already uses. `site/js/math.js` refreshes the live HTML and, after every redraw, puts LaTeX annotations over
the SVG text of Plotly. Scientific tick exponents and Unicode minus signs are converted explicitly; the trace values
do not change. Write new formulas explicitly in LaTeX; do not rely on the notation matching.

To check scientific notation, units, all 44 plot explanations, the horizontal axis headings and the authored text
without a browser, run `node webviz/scripts/check-math.mjs` from the repository root.

### Comparing the eight rings

The viewer starts in **All 8**. The ring buttons select the ring to emphasize; they do not hide the others. Toggle
All 8 off for the detail of one ring. The address saves the choice as `a=1` or `a=0`; an address without it uses
All 8.

In All 8, Parts shows eight ring columns in its live snapshot table and eight curves for each ring quantity. The
secondary signals (oracle modulation, full-code heater power, signal photocurrent, register and applied DAC code) get
their own comparison rows. Statistics compares the ring columns for the selected statistic. The spectra and the
thermal-source histories use one panel per ring; the thermal histories share a temperature scale. The controller
scatter colours its points by ring in comparison mode and by time in single-ring mode. Crosstalk also compares the
response of each ring to a step in its own heater, and keeps the detailed source paths of the selected observer. The
scenario bias and ones density show every ring, not only ring 0 or a bank minimum.

The shared inputs and the bank supply limits stay shared quantities. Identical nominal curves can overlap; use the
legend to isolate rings. The bistability data has ring 0 only, and its card says so. The comparison mode changes no
engine data.

### Plot spacing

Each time-plot row has at least 200 px for data, its own caption space and a gap to the next section. The captions
are measured after the LaTeX is rendered. Event and scrubber lines stop at the edge of each plot; event labels that
are near each other go into separate lanes. The legends take a fixed pixel gap below the time axis, not a fraction of
the height of the whole stack. Ring panels have a 32 px grid gap and fall back to one column when two readable panels
do not fit. The thermal response and its time-constant table each take a full-width section; the pole/residue pairs
wrap inside their own cells. Wide source matrices scroll inside their card, so their labels are not squeezed.

### Layout proportions

The layout uses shared spacing values in `style.css`: 32 px between sections and panels, 22–32 px of card padding and
at least 200 px of data height per time-plot row. Controls are at least 36 px high. Plot labels and matrix cells use
larger type; matrices and standalone charts get enough height for their rows and legends. Captions keep a readable
line length; they do not stretch across the whole screen.

The toolbar groups the scenario, ring and playback controls. On a narrow screen it wraps and scrolls with the page, so
it does not cover the plots. Ring panels and equation explanations fall back to one column when needed. Dense tables
and source matrices keep their readable sizes and scroll inside their cards. The BER explanation shows its
calculation in labelled equation rows with full-size fractions.

Each chart watches the width of its own visible container, and its redraws, resizes and cursor updates run one after
the other. Hidden or detached charts skip updates; this prevents axis-scaling errors at zero width when the view
changes. Spectrum panels also recompute the spacing of their locator labels after a resize. To check the queued
updates, the fresh width measurements and the handling of hidden charts, run
`node webviz/scripts/check-plot-updates.mjs`.

The metrics comparison keeps all eight ring values in the grid. A selected metric shows its full definition above the
time plot; the definition is not repeated down a narrow table column. Where the rings differ, the value of the worst
ring is underlined (there is no extra column for it).
