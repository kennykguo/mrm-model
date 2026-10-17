# webviz: the mrm model, running

A website that shows the g26 plant model running: 8 silicon microring modulators (MRMs) on one bus waveguide, each
with an in-ring heater, a PN junction for the data and a monitor photodiode (PD) on its drop port; 8 laser lines on
a 400 GHz grid near 1310 nm; a heater lock holding each ring on its line while disturbances push it. Every number
and picture on the site is model output (engine runs, or the model's own optics and thermal functions). Nothing is
drawn by hand except the waveguide outlines, which the model does not have (section "Limits").

Everything lives in this folder. Nothing in `mrm/`, `tests/`, `docs/`, `experiments/` or `tools/` was changed.

- `generate.py`: makes every data file from the model (one command, about 1 minute).
- `site/`: the website (`index.html`, `style.css`, `app.js`, `js/*.js`). `site/data/`: the data files it reads.
- `make_colormaps.py`: writes `site/js/colormaps.js` (the inferno and coolwarm lookup tables, from matplotlib).
- `check_data.py`: re-checks the data against fresh engine runs (section "How the data was checked", item 6).

## Opening the site

```sh
cd ~/mrm-model/webviz/site
python3 -m http.server 8000
```

Then open http://localhost:8000 in a browser (Chrome, Safari or Firefox; WebGL must be on). The page needs a web
server: browsers block data loading from a `file://` page. It also needs the internet once, for two pinned libraries
from cdn.jsdelivr.net: three.js 0.160.0 (the 3-D view) and plotly.js 2.27.0 (the 2-D views). There is no build step.

The address keeps the view: `#s=a1_aggressor&r=0&f=262&c=err&m=height` opens the A1 scenario, ring 0, frame 262
(65.75 ms), rings coloured by lock error, map as a height field. Keys: space plays and pauses, the arrow keys step one
frame (shift: 10 frames).

## Regenerating the data

```sh
cd ~/mrm-model
.venv/bin/python tools/memguard.py -- .venv/bin/python webviz/generate.py --date 2026-10-17
```

- About 65 s on the M2 (warm numba cache), peak memory about 0.35 GB, 24 MB of JSON in `site/data/`.
- `--only steady,a1_aggressor,laser_step,cold_start,bistability`: regenerate some files (meta.json and checks.json
  are merged, not lost).
- `--quick`: short runs into `runs/webviz_quick/` (a smoke test; the site does not read it).
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
(docs/ENGINE.md section 5).

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
  - `line_on` (the line the ring sits on, -1 none), `declared_line`, `order`: value at t_s.
- `lock_reference`: `d_ref_pm` per ring (the operating point the lock error is measured about) and its kind.
- `drives`: `aggressor_w` {site: [frame]} (window mean power into each aggressor site), `laser_power_factor`
  [frame][line] (chip power relative to nominal), `t_base_k` [frame].
- `events`: timeline markers taken from the data (aggressor power peak and the A1 target cusp, the laser step, the
  start of scoring, each ring's lock time in the cold start).
- `summary`: the engine's own metrics (every tick after `score_from_s`), per ring (`per_ring`) and worst case:
  max and rms lock error, OMA penalty, unlocked rings, rings on a wrong line, excursions, time at the heater rails,
  lock settings, aggressor calibration, warnings.
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
  rows `RL_PIN` and `RL_THRU`): the light of each line arriving at each ring after the bus segment, and leaving its
  through port (data-averaged). The site sums them over the lines for the bus light. (Added by the site agent; the
  rest of the file is unchanged by it.)

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

### checks.json

The numbers of the next section, regenerated with the data.

## The site: what each view shows

Everything on the page follows the time scrubber, except the bistability plot. Each view answers one question.
Colour ranges are fixed per scenario, so a colour means the same number all through a run.

**Two clocks.** A *frame* is a window of 0.1 ms (0.25 ms in A1); the traces are window means (codes: the value at
the end). A *key frame* is the instantaneous state at the end of a frame: 68 to 88 per run, every 1 to 20 frames (denser
near events).
The spectra, the temperature map and the bus light exist only at key frames, so those views show the key frame
nearest the scrubber and say which one. The rings, heaters, PDs and aggressor in the 3-D view follow every frame.

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
- Heater: the copper annulus inside each ring is where the model's extraction puts the heater (the plant treats it
  as a lumped source at the centre). The copper column on it is the heater power: 30 um tall = the scale maximum in
  the legend (20 mW in the warm runs, 50 mW in the cold start).
- Monitor PDs (the squares 25 um above each ring, at the end of the drop waveguide): colour = PD temperature, on the
  shared scale. The model has no PD size; the marker is 7 um.
- Drivers (grey blocks 30 um below each ring): heat sources of the junction drivers. No colour.
- Aggressor: the red ring at (0, -30) um, the A1/A2 default site (the same point as ring 0's driver). Its glow grows
  with its power (largest in the legend, 0.287 W in A1); it reads *off* in the runs without one.
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

**Where is this ring's resonance relative to its laser line?** The selected ring alone, at the nearest key frame:
the through port with the junction at each data level (bit 0: 0.40 V reverse, bit 1: 1.70 V), and the drop port
averaged over the data (what the monitor sees). The notch bottoms come from the exact resonance and depth
(`resonance_level_nm`, `notch_level_db`), merged into the 12.5 pm grid; a notch below -50 dB is written at the
bottom. The solid vertical line is the ring's own laser line, the dotted one the resonance; the readout gives the
detuning. The strip above is the whole comb: every laser line (L0 the reddest) and every ring's resonance, with the
window of the plot shaded. Two resonances can sit a few pm apart (in the cold start's sweeps); then only one of them
is labelled (the selected ring first), and hover names each. In the cold start the window follows the resonance, so the own line is often outside it
(an arrow at the edge gives the distance).

**What is the controller doing over time?** The selected ring's traces on one time axis, one panel per quantity
(no dual y-axes): detuning (line: frame mean; band: min to max within each frame; green band: the lock band about
the lock reference, named with its width in the readout above the plot; dotted: the reference), heater code, monitor code, ring temperature, and the scenario drive in
the bottom panel: aggressor power (A1), the laser power factor of the ring's line (laser step), or else the light
reaching the ring (`p_in_mw`; it shows the 3 ms laser turn-on in the cold start). The grey start is not scored
(pi_lock_c maps its peak in the first 3 ms). The vertical line is the scrubber; click a trace to move it there.

**How hot is the die?** The temperature map from above (4 x 2.5 um grid, smoothed for display) on the shared scale,
at the nearest key frame. Each ring disc is filled with the ring's own engine temperature (hover: both the ring
and the map value at its centre, which reads 2-9 K low); the squares are the PDs (hover: their temperature, equal to
the map there), the circle the aggressor site with its power. The marks on the map are light in both themes (the
map's colours do not change with the theme). On a narrow screen only the selected ring is labelled and the
aggressor's power is in its hover.

**Why is the ring bistable?** Not tied to the scrubber. Ring 0 swept open loop: heater code ramped up for 40 ms
then down for 40 ms (the engine, blue and orange), over the static S-curve of the model's laws (grey: stable,
dotted: the unstable middle branch), with the two folds (diamonds) where a stable branch ends. Rising, the ring
stays on the cold branch past the hot branch's start, then jumps up at the cold fold; falling, it stays on the hot
branch until that fold, then falls off. The buttons change the light at the ring (0, +4, +8 dBm with only line 0
on, and +4 dBm with all 8 lines); the loop widens with the light. The vertical axis can be the drop power (what the
monitor sees) or the detuning.

**How well is it locked overall?** The engine's own metrics (every 1 us tick after scoring starts), per ring: max
and RMS lock error, OMA penalty, the time unlocked after the ring's first lock (`time_unlocked_s`: the engine's lock
state with hysteresis, so the time before the first lock is not in it), the time with the heater at code 0, and
whether the ring is locked at the end. A value outside the band is marked "out of band" in text. Click a row to
select the ring. In the laser step every ring reads "out of band" and "not locked at the end": the lock settles about
20 pm red of the operating point it held before the step, and the band (±5 pm) is measured about that old point. In
the cold start scoring starts at 0 ms, so the max lock error includes the cold rings, about 5 nm from their lines,
and every ring reads "out of band" although all 8 end locked.

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

Numbers from checks.json for commit ee381af, generated 2026-10-17.

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
     evaluates loss and coupling at the previous tick's temperature (docs/ENGINE.md section 4) while the spectra use
     the present one.
3. **Thermal maps (check b).**
   - The extended system holds the engine's 970 lags unchanged (same sources, poles and residues).
   - Reconstructed against the engine at every key frame: PD observers within 0.32-0.44 mK (warm runs) and 32 mK
     (cold start, where the PD rises up to 18.7 K); the die sensor within 0.12 mK (0.95 mK cold); the ring observers within 2-18 mK
     (warm) and 1.1 K (cold start: the ring self poles are 0.5-13 us, faster than the 10 us reading).
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
   The static hot-branch folds sit 32, 12 and 5 pm blue of the line; docs/EXPLAINED.md section 2.4 gives about 9 pm
   at +4 dBm measured from the drop peak.
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
  controller (docs/EXPLAINED.md section 4, item 5).
- Site: waveguides are drawn 3x their width; the PD and driver markers have no modelled size; the heater column is
  a power gauge, not a structure. The spectra, map and bus light are at the nearest key frame, not interpolated.
- The page loads three.js and plotly.js from cdn.jsdelivr.net (pinned versions); offline, the views do not start.
