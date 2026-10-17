# The mrm model, explained

For Kenny. What the model is, how to run your own controller on it, what each mechanism does, where it disagrees with
your textbook, and what is still wrong. Depth lives in docs/ENGINE.md (the engine), docs/PARAMETERS.md (every number)
and docs/ARCHITECTURE.md (the contracts).

## 1. Overview

**What it is.** A plant model: a simulator of the device that your lock controller will drive. The device is the
transmit bank of Lightmatter's G26 link (OFC 2026): 8 microring modulators (MRMs) on one bus waveguide. Each ring has a
heater inside it, a PN junction for the data and a monitor photodiode (PD) on its drop port (the second waveguide that
takes light out of the ring). 8 laser lines on a 400 GHz grid near 1310 nm, 50 Gb/s NRZ (two-level) data. The plant
steps in 1 us ticks. Your controller runs every 10 us (100 kHz): it reads ADC codes and writes heater DAC codes, as
firmware would.

**Where the numbers come from.** G26 papers and slides; the G25 test vehicle on the same GlobalFoundries 45SPCLO
platform; the foundry; other groups' papers; and our own solvers for an assumed cross-section. Many numbers are assumed.
Every number carries a provenance word: *printed* (stated in a paper), *read-off* (digitized from a figure), *inferred*
(arithmetic on printed numbers), *solver* (our mode or heat solver, assumed geometry), *literature* (other groups'
devices), *assumed* (no source). Your real device values will replace them.

**Worlds.** Where a number is unknown, the preset gives a range or a set of hypotheses. A *world* is one draw of all of
them. A run uses the nominal world, the named corners (hand-picked hard worlds, each a small preset such as `g26_h86`),
or many sampled worlds.

**The parts** (folders of `mrm/`):

- `params/`: loads a preset (a YAML file of device numbers), checks units and provenance, draws worlds.
- `plant/`: the physics of one tick: optics, heat, self-heating, heater and bias electronics, monitor and ADC, laser,
  data path.
- `scenarios/`: the disturbances: the on-die aggressor, ambient soak, laser drift, power steps, traffic changes, faults.
- `control/`: the controller interface and the shipped controllers (`baselines/`).
- `sim/`: the engine (a compiled numba kernel), the runner, the metrics, experiment files and the `mrm` command.
- `extraction/` and `data/extracted/`: offline mode, coupler, heat and junction solvers and their saved results.
- `validation/`: the published measurements as targets, and the generator of the validation report.
- `calibration/`: fits the model to the calibration targets and writes `g26_calibrated` (stale now, section 4).
- `waveform/`: short picosecond runs of one modulator (eye shapes), not used in a lock run.

**The four files you need.**

1. `mrm/params/presets/g26.yaml`: the device. Every number has a value, a provenance, a source and often a note. Read it
   as a table in docs/PARAMETERS.md, or ask for one number (command below).
2. `examples/controllers/template_python_controller.py`: copy it to write your controller. It is a plain integral lock
   on the monitor. The same folder has a one-ring template and a compiled (numba) template;
   `examples/controllers/README.md` explains them.
3. `experiments/examples/steady_pi_lock.yaml` (40 ms at 25 °C) and `experiments/examples/a1_pi_lock.yaml` (one 130.7 ms
   period of the A1 aggressor): an experiment file names the preset, the controller, the scenario, the run length and
   how to score it.
4. docs/KNOWN_ISSUES.md: what is still wrong or stale.

**Rules for your controller.** Read only the observation (ADC codes, sample ticks, temperature-sensor codes, the heater
code now applied) and the spec. Return integer heater codes (0 to 65535). The controller gets device numbers only
through `spec.nominal`, and g26 releases only the line grid and the ring-to-line map; everything else it must measure.
At the start (`start: warm` in the files above) each ring sits locked on the stable side of its line, with a settled
thermal state.

**Commands.** Run them from the repo, in this order. `tools/memguard.py` stops a job that uses more than 3 GB, so a run
cannot freeze the 16 GB Mac. After a change to numba code, the first run compiles for about 2 minutes.

```sh
cd ~/mrm-model

# a shipped controller (pi_lock_c) on g26, 40 ms; it prints a report and writes runs/<UTC time>_<hash>/
.venv/bin/python tools/memguard.py -- .venv/bin/mrm run experiments/examples/steady_pi_lock.yaml

# copy the template, then edit step() in my_lock.py
cp examples/controllers/template_python_controller.py examples/controllers/my_lock.py

# your controller: steady, then under the A1 aggressor (with a controller parameter changed)
.venv/bin/python tools/memguard.py -- .venv/bin/python -m mrm.cli run experiments/examples/steady_pi_lock.yaml --set controller.name=examples.controllers.my_lock:MyIntegralLock
.venv/bin/python tools/memguard.py -- .venv/bin/python -m mrm.cli run experiments/examples/a1_pi_lock.yaml --set controller.name=examples.controllers.my_lock:MyIntegralLock --set controller.params.k_i_codes=800

# your controller over the uncertainty: nominal world, named corners, 128 sampled worlds (about 3 minutes)
.venv/bin/python tools/memguard.py -- .venv/bin/python -m mrm.cli run experiments/examples/robust_pi_lock.yaml --set controller.name=examples.controllers.my_lock:MyIntegralLock

# one number of the device, with its source and the textbook's value
.venv/bin/mrm params explain g26 ring.thermal.tau_fast_s
```

Notes on these commands:

- `python -m mrm.cli` is the same `mrm` command; started this way, Python can import `examples/...`. If you rename the
  class in `my_lock.py`, change it after the colon. Also change the name in `@register(...)`, so the copy and the
  template never share one name.
- `--set PATH=VALUE` changes any number for one run, for example `--set ring.thermal.tau_fast_s=1e-5`. `--alt PATH=NAME`
  picks a stored alternative, for example `--alt uncertainty.thermo_optic=H86`.
- The robust run ends with a table of criteria. It prints FAIL and exits with code 1 when any world breaks one. That is
  a result, not a crash.

**Reading the results.** The report is printed and saved as `report.txt` in the run directory. The columns you will use
most: `max_abs_lock_error_pm` (largest distance of a ring from its own steady lock point, pm), `unlocked_rings`,
`wrong_line_rings` (rings that ended on another laser line) and `oma_penalty_db_worst` (modulation amplitude lost
against the best possible point, dB). Per ring, with a plot of the detuning, for the newest run of kind `run` (a robust
run stores its results in another layout):

```sh
.venv/bin/python - <<'EOF'
import glob, json, os
import matplotlib.pyplot as plt
from mrm.sim.manifest import read_run

def kind(d):
    f = os.path.join(d, "metrics.json")
    return json.load(open(f))["kind"] if os.path.exists(f) else None

run = read_run([d for d in sorted(glob.glob("runs/2*")) if kind(d) == "run"][-1])   # newest run of kind "run"
m = run["trace"]["metrics"]["main"]                    # per-ring metrics, shape (worlds, rings)
print(run["dir"])
print("max |lock error| pm:", m["max_abs_lock_error_pm"][0].round(1))
print("rms lock error pm:  ", m["rms_lock_error_pm"][0].round(2))
print("locked at the end:  ", m["locked_final"][0])
print("OMA penalty dB:     ", m["oma_penalty_db"][0].round(2))

d = run["trace"]["trace"]["main"]["detuning_m@10:mean"]
plt.plot(d["t_s"] * 1e3, d["value"][:, 0, :] * 1e12)
plt.xlabel("time (ms)"); plt.ylabel("detuning from the line (pm)")
plt.savefig(run["dir"] / "detuning.png")
EOF
```

The plot goes into the run directory as `detuning.png`. Every run directory also holds the full resolved config
(`experiment.resolved.yaml`) and the exact arrays (`trace.h5`). `.venv/bin/python -m mrm.cli rerun runs/<dir>` runs it
again and checks that every array is bit-for-bit the same.

## 2. The mechanisms

The order follows the light and the heat through the device. It is not a ranking. Each section says what the mechanism
is, why a lock controller cares, the main setting in g26.yaml with its g26 value, and where your textbook
(`~/interactive-textbooks/lightmatter-capstone`) differs. Neither the textbook's numbers nor the model's have been
checked against your device yet; where they differ, the section names the model's source.

Sign convention: in the model, **detuning = resonance wavelength minus laser line**. Positive means the ring is red of
its line. The textbook's δ is the opposite (laser minus resonance), so the model's positive detuning is a negative δ in
the textbook, its "blue flank".

### 2.1 The laser

- **What.** 8 lines, 1302.6 to 1318.6 nm, 400 GHz apart. Each line has a power, a small frequency error, intensity noise
  (RIN) and slow drift. At power-up the lines ramp on over 3 ms.
- **Why it matters.** The power sets how hard each ring heats itself and how much current the monitor sees. A power step
  moves a lock that targets a fraction of the monitor peak. A line's frequency error moves the target. Drift is a slow
  disturbance. The ramp decides which branch, and which line, a cold ring lands on.
- **Main setting.** `laser.power_per_line_dbm` 10 dBm at the laser module (printed, G26), less
  `optical_path.chip_input_loss_db` 6 dB (inferred), so +4 dBm per line at the first ring. Worlds span 0 to +8 dBm at
  the ring. Line errors: 0 nominal, spread 4 GHz per die and 6 GHz per line (inferred); ±20 GHz is the printed
  requirement. Drift (0.02 nm and 2.5 % per 24 h) is a scenario.
- **Textbook.** It quotes 14 dBm per line at the laser (M1000). The model uses the G26 budget's 10 dBm. Both put +4 dBm
  (the G25 value) at the ring.

### 2.2 The ring optics and coupling

- **What.** Each ring is an exact add-drop ring: light couples in from the bus, circulates, and leaves through the bus
  (through port) or the drop port. The model solves it for every laser line and several resonance orders, with the loss
  of the bus between rings. Each ring is built 7.6 nm blue of its own line, so it must warm by about 109 K to reach the
  line (lock at about 134 °C at 25 °C ambient).
- **Why it matters.** The line shape sets the monitor signal against detuning: its slope is your loop gain, its width
  your capture range. The free spectral range (FSR, the spacing of a ring's resonance orders) and the grid decide which
  foreign lines a ring can grab. A cold ring sits about 2 linewidths blue of the line three channels up, so a bank needs
  a supervisor to assign lines at start-up.
- **Main setting.** `ring.optics.q_loaded` 3850 (Q, the sharpness of the resonance: linewidth about 338 pm; inferred,
  G25; worlds 3500 to 4200). `ring.optics.fsr_thz` 1.8 THz, about 10.3 nm (printed G25 design target).
  `ring.optics.resonance_offset_nm` 7.6 nm (inferred, G25). `ring.optics.kappa_d_over_kappa` 0.007 sets the drop
  coupling (inferred from the G26 monitor current). The ring is under-coupled (it loses more per round trip than it
  couples to the bus), with a 27.9 dB notch (the dip at the through port) at 0 V (G25).
- **Textbook.** Q: textbook 3500 (G25 talk); model 3850, the middle of the interval from 3500 (talk) to 4200 (linewidth
  read-off). FSR: the textbook keeps a measured bound above 14.1 nm; the model treats that bound as unsupported (the G25
  traces are about 1 nm wide) and uses the 1.8 THz design value, with other FSRs as worlds. Notch depth: the textbook
  reads it against heater voltage (G25 sweep); the model computes it from the coupler's wavelength dependence at each
  line and the loss against temperature, so under lock the depth does not follow the heater code.

### 2.3 The junction, bias and data

- **What.** A PN junction around most of the ring (80 %, assumed). Reverse bias removes carriers. That shifts the
  resonance red (the electro-optic effect) and lowers the loss (less light absorbed by free carriers). The 50 Gb/s data
  swings the junction. Data also heats the ring: junction charging plus the CMOS driver beside it.
- **Why it matters.** Bias is a fast second actuator. In the model at the g26 lock it tunes weakly: a 0.2 V step moves
  the resonance only 0.1 to 0.5 pm, because the loss change alters the absorbed heat and cancels most of the
  electro-optic shift (about 10 pm by the electro-optic law alone). This is an open physics check (KNOWN_ISSUES,
  controllers). Data heat lifts each ring about 2 K, so traffic start, stop and squelch move rings by that much.
- **Main setting.** `ring.junction.eff_at_0v_pm_per_v` 72.5 pm/V, falling to about 32 pm/V at 3.75 V (inferred: a
  depletion law, the shift growing as the square root of the voltage plus the 0.9 V built-in voltage, fitted to the G25
  bias ladder). Bias DAC 12 bits, 1.0 to 3.75 V (printed axis, G26); g26 holds it at 1.05 V (`scenario.bias_v`) unless
  you bind it to your controller (`controller_interface.bind.tx.bias: controller`). `driver.swing_pp_v` 1.3 V (printed,
  G25 simulation). Driver heat: 2 mW plus 100 fJ per transition (assumed).
- **Textbook.** Efficiency: it interpolates the G25 bias ladder up to 1.6 V (and lists a flat 50 pm/V from the talk);
  the model uses the depletion law over the whole 1.0 to 3.75 V axis. Swing: the textbook lists 0.9 V as driven on the
  bench; the model reads that as the pattern generator's swing into the driver, and uses 1.3 V. The textbook takes the
  bias shift from the G25 ladder alone, with no term for the change of absorbed heat at the operating power.

### 2.4 Self-heating

- **What.** Part of the light the ring loses becomes heat in the ring. Heat moves the resonance red, which changes how
  much light it absorbs. This feedback makes the ring bistable: two stable temperatures for one heater code.
- **Why it matters.** It decides where you can lock. With the ring red of the laser (positive detuning), the feedback is
  negative and stable. Past the fold (the end of the stable branch), about 9 pm blue of the drop peak, the ring jumps
  about 1.5 nm blue. So a lock on the peak falls off, and a cold ring can heat itself onto a foreign line at power-up.
  On the stable side the feedback divides the heater's effect on the resonance by 3 to 7, and every disturbance by the
  same factor. The absorbed-light path is faster than the heater: about 43 % of it arrives with a 0.4 us pole.
- **Main setting.** Heat fraction η = `ring.optics.heat_fraction` 0.85 (derived from the loss budget; its parts are
  inferred or assumed), worlds 0.5 to 0.95 (`uncertainty.self_heating.eta`). Light heats the ring 1.76 times more per mW
  than the heater does (`ring.thermal.z_ratio_opt_heater`, solver), worlds 1 to 2. Bistability ratio about 6 at +4 dBm
  (above 1 means bistable; `mrm params explain g26 bistability_ratio`).
- **Textbook.** It assumes η = 0.3 and heats the ring with one thermal resistance for heater and light (ratio 1, Ch17
  eq. 17.9a). With those it finds that self-heating about halves the heater gain (Ch10). The model uses 0.85 and 1.76,
  so the reduction is 3 to 7 times. The textbook's 0.3 is a stored stress world:
  `--alt uncertainty.self_heating.eta=stress_0p3`.

### 2.5 The thermo-optic coefficient

- **What.** How far the resonance moves per kelvin of ring temperature (dλ/dT). It grows with temperature, because
  silicon's index changes faster when hot.
- **Why it matters.** It converts every temperature disturbance (ambient, aggressor, neighbours) into pm of detuning. It
  sets a feed-forward gain from a temperature sensor, and the lock temperature.
- **Main setting.** `ring.thermal.dlambda_dt_pm_per_k` 65.1 pm/K at 25 °C, about 73.9 pm/K at the 134 °C lock (solver,
  assumed cross-section, `ring.thermal.dlambda_dt_beta_per_k`). Worlds choose between this solver value and 86 pm/K
  (`uncertainty.thermo_optic`; corner preset `g26_h86`). These are worlds, not devices: one world uses one value for
  every ring.
- **Textbook.** It designs at 86 pm/K, the "~15 GHz/°C" in the G26 introduction, held constant. The model's research
  reads that number as a general remark, not a measurement, and above the materials ceiling (about 71 pm/K at 300 K), so
  the model uses the solver value and keeps 86 as a hypothesis. The textbook also uses dn_Si/dT at 1.86e-4 /K (a C-band
  value); the model's solver uses the 1310 nm value, rising with temperature. To run 86:
  `--alt uncertainty.thermo_optic=H86`.

### 2.6 The heater driver

- **What.** A 16-bit DAC drives each heater. In g26 it is a voltage DAC: power is V²/R, so the gain per code grows with
  the code. The heater resistance rises with the heater's temperature (its temperature coefficient of resistance, TCR),
  which makes the tuning droop at high drive. The DAC has a 1 us latency. One heater can only heat. All heaters share
  one supply limit.
- **Why it matters.** The loop gain changes with the code (about 0.18 pm per code at a quarter of full scale, 0.54 at
  three quarters, before the self-heating reduction), so a fixed-gain loop is over- or under-tuned somewhere. At code 0
  the heater cannot cool further: under A1 ring 0's heater sits at code 0 for about 15 ms and the ring leaves its point
  by 454 to 473 pm, with every controller tried. The supply limit can cut every heater of a die at once.
- **Main setting.** `ring.heater.driver_mode` voltage (worlds also draw current, power and DAC+PWM laws).
  `ring.heater.full_scale_v` 1.6 V (assumed). `ring.heater.p_fsr_mw` 25.45 mW per FSR (printed, G25 wafer median), about
  0.405 nm/mW. `ring.heater.r_cold_ohm` 45.4 Ω (inferred), `ring.heater.tcr_per_k` 1.4e-3 /K (literature, doped
  silicon). Supply limit `array.heater_total_power_max_w` 0.2 W and `array.heater_supply_current_max_a` 0.15 A
  (assumed). No slew limit. At the lock ring 0's heater gives about 14 mW, about 81 K of the ring's 109 K rise; light,
  data heat and neighbours give the rest.
- **Textbook.** TCR: textbook 1e-3 /K (assumed); model 1.4e-3 /K. Both fit the same G25 droop (0.00904 /mW). In the
  textbook the resistance follows only the heater's own heat (Ch11 eq. 11.14). In the model it follows the heater's
  whole temperature, so on a die 80 K hotter the same code gives about 10 % less power. The 19,500 to 49,300 codes the
  textbook quotes are what one G26 test used; the model's DAC spans 0 to 65,535.

### 2.7 The thermal paths and crosstalk

- **What.** Heat flows from each heater, each ring's absorbed light, each driver and the aggressor to every ring,
  monitor PD and temperature sensor, each path with its own delays. In g26 each path is a few poles taken from imec's
  published measurements and fits (the "interim" model). A model reduced from our own finite-element heat simulation
  also exists (`fidelity.thermal_model: rom`), but g26 does not use it yet: it misses the imec targets (section 4).
- **Why it matters.** The heater-to-ring poles set your loop's bandwidth and phase margin. Crosstalk couples the 8
  loops: a neighbour's heater moves your ring, slowly. The die sensor sees the heaters and the aggressor through a slow,
  weak path.
- **Main setting.** Heater to own ring: 5.70 K/mW (derived from P_FSR and dλ/dT), poles 3.18 us (41 %) and
  13.0 us (54 %) (literature, imec), plus a 3 ms package tail (5 %, assumed). `ring.thermal.tau_fast_s`, `ring.thermal.tau_slow_s`,
  `ring.thermal.share_fast_frac`. Worlds scale the time constants 0.5 to 2 times. Crosstalk
  `ring.thermal.xt_nearest_frac` 3 % to the nearest ring with a 1.52 ms pole, 1.5 % to the others (literature, imec, a
  package cooled by natural convection). A package held at a set temperature by a thermoelectric cooler (TEC) would give
  1.0 % arriving in 0.12 ms (corner `g26_tec_package`, from our heat simulation); which package G26 has is unknown. Die
  sensor `die0`: 1 mm away, 1 kHz, 0.0625 K steps, a few K offset (assumed).
- **Textbook.** Poles: textbook 10 us (75 %) and 300 us (25 %), assumed; model 3.18 us and 13.0 us with a 3 ms tail,
  from imec. Crosstalk: textbook 10 % nearest and 3 % next, arriving as fast as the ring's own heater; model 3 % and
  1.5 % arriving over milliseconds. Heater thermal resistance: textbook 4.7 K/mW (0.405 nm/mW ÷ 86 pm/K); model 5.70 K/mW
  (the same P_FSR with its own dλ/dT). Both give the same 0.405 nm/mW from the heater. To run the textbook's network:
  `--alt uncertainty.thermal.family=textbook --alt uncertainty.thermal.package_tail_share_frac=textbook` (corner preset
  `g26_textbook_thermal`).

### 2.8 The aggressor

- **What.** An on-die resistive heater beside ring 0 (G26 paper), driven so that ring 0 follows the published
  temperature swing. Scenario A1 (G26 slide 10): T = 113.6 − 88.5 |sin(π t / 130.7 ms)| °C, a sharp hot cusp, 2127 °C/s
  at the steepest (a fit to the read-off trace). A2: the paper's 808 °C/s (its waveform is not published, so four
  readings exist). Other scenarios: ambient soak, processor power steps, laser drift and sweeps, connector re-mates,
  traffic changes (`mrm/scenarios/library.py`).
- **Why it matters.** It is G26's own design-case disturbance for the lock. It enters as heat through the thermal paths:
  ring 0 gets the most, the other rings less and later, and the monitor PDs and the die sensor warm too.
- **Main setting.** Site `aggressor` in `layout.sites`: 30 um from ring 0 and 55 um from its PD (position assumed). The
  scenario's power is solved before the run so that ring 0 follows the published trajectory. By default the published °C
  become pm through Lightmatter's 15 GHz/°C (`referencing: resonance`); the other choice matches it in kelvin. A stress
  site 5 um from ring 0's PD (`a1_pd_adjacent`) tests monitor dark-current runaway; it is not a real layout.
- **Textbook.** It injects the swing as a temperature straight at the ring's thermal node (Ch18; Ch17 eq. 17.2 adds
  ambient with no lag), so nothing else on the die feels it. The model injects power at a site, so each ring, PD and
  sensor sees it through its own path. Its rate: textbook −2300 to +2200 °C/s (read-off); model 2127 °C/s at the cusp (a
  fit of the temperature trace).

### 2.9 The monitor PD, front end and ADC

- **What.** A germanium PD on each drop port. Its current is integrated for 2 us, then converted by a 16-bit ADC (an
  integrating converter, as in patent PAT-2). The PD adds shot noise and a dark current (current with no light) that
  grows with its own temperature. The front end adds noise. The ADC has a full scale, offsets and nonlinearity.
- **Why it matters.** It is how the controller sees each ring's detuning. Its gain is uncertain, so the shipped locks
  set their target as a fraction of a peak they measure, not as a fixed code. Dark current that rises with PD
  temperature moves the lock point. A peak above full scale clips. The timing adds delay to your loop: 2 us window, 2 us
  ADC latency, 1 us heater latency, 10 us control period.
- **Main setting.** `ring.monitor.responsivity_a_per_w` 0.95 A/W (printed, G25). `ring.monitor.dark_i0_na` 5 nA at 25 °C
  (assumed, foundry limit < 30 nA), ×1.6 per 10 K (literature). `ring.monitor.adc_full_scale_ua` 40 uA (assumed): the
  drop peak is about 32 uA and a half-peak lock about 16 uA. `ring.monitor.adc_rate_hz` 250 kHz (inferred, patent).
  `ring.monitor.input_noise_pa_per_rt_hz` 1 pA/√Hz (assumed). Worlds scale the monitor gain 0.5 to 2 times
  (`uncertainty.monitor.gain_scale`).
- **Textbook.** It has no PD dark current; the model adds it. Front-end noise: textbook 6.4 pA/√Hz (the data receiver's
  amplifier); model 1 pA/√Hz for a slower monitor amplifier (assumed). Full scale: the textbook sizes it at 15 to 20 uA;
  the model uses 40 uA so that a peak search does not clip. Timing: textbook 10 us for sample period, window and delay
  (assumed); model as above.

## 3. Where else to look for textbook differences

The section 2 entries are the differences that matter for controller work. The full list: the "Textbook comparison"
table at the end of docs/PARAMETERS.md (one row per key), `mrm params explain g26 <key>` (its `disagrees:` line) and
docs/research/06_textbook_audit.md (the audit, with evidence).

## 4. Known issues that matter most

The full list is [docs/KNOWN_ISSUES.md](KNOWN_ISSUES.md). The ones to know first:

1. **The calibration is stale.** `g26_calibrated`, `g25_test_vehicle_calibrated` and `mrm/calibration/latest_fit.json`
   were fitted before the final review's fixes and were not fitted again. They stay stale until your real device
   parameters arrive and the fit is run again. Some calibration tests are expected to fail until then. Everything in
   this document runs on `g26`, the default.
2. **The validation reports are stale.** docs/VALIDATION.md and docs/VALIDATION_g26_calibrated.md carry a STALE banner.
   Do not quote their numbers. They are generated again after the new fit.
3. **The model misses several published measurements** (as last computed): the heater tuning law, the notch depth
   against heater voltage, the heater code map, the modulation amplitude (relOMA) targets, and the imec targets of the
   finite-element heat model. Nine confirmed validation findings are recorded and not fixed. See KNOWN_ISSUES sections 2
   to 4.
4. **The heat dynamics come from imec devices and assumptions, not from G26.** The time constants, the crosstalk and the
   package (natural convection or TEC-held) are unknown for G26. Every controller result is conditional on them; the
   first line of every report says so.
5. **Ring 0 runs out of heater at the hot end of A1 and A2.** Under A1 its heater sits at code 0 and the ring leaves its
   point by 454 to 473 pm with every controller tried. This is a limit of the plant with the present numbers, not of a
   controller.
6. **Start-up can capture the wrong line.** Cold rings can heat themselves onto a foreign line, and the WDM supervisor
   (the shipped controller that assigns lines) can still declare a ring good on a wrong line in some sampled worlds.
7. **Sampled worlds lose rings** even with `pi_lock_c` from a good warm start, and the assumed heater supply limit binds
   in about a third of the sampled worlds. A claim over the uncertainty needs every world to pass.
8. **A scenario given by name runs at 1.0 V bias, the config at 1.05 V.** A scenario passed to `runner.run` by name (for
   example `"static"`) and the same scenario in an experiment file give results that are not comparable number for
   number.
