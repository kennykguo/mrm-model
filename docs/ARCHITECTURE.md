# Architecture: how the parts fit together

This document says how the model's parts fit together: the package layout, units, configuration, saved solver
results, controller interfaces and accuracy rules. Read [EXPLAINED.md](EXPLAINED.md) for the physical intuition
and [ENGINE.md](ENGINE.md) for how to run or change the simulator.

The main flow is:

```text
preset + uncertainty choices → realized device parameters → plant + scenario + controller
                                                                   ↓
                                                        metrics and recorded traces
```

Offline solvers supply saved results to the parameter-loading stage. They do not run inside the simulator.

## Where to start reading

Documents, in this order:

1. [guide/mrm_guide.pdf](guide/mrm_guide.pdf): the reading guide, from the add-drop ring of textbook chapters 0-5 to
   the whole model.
2. [EXPLAINED.md](EXPLAINED.md): the physical picture, and "Try it" (the first runs).
3. [KNOWN_ISSUES.md](KNOWN_ISSUES.md): what the model does not do yet.
4. [ENGINE.md](ENGINE.md), sections 1-6: how to run the engine and what happens in one tick.
5. This document, section 2: where each part of the code lives.
6. [CODE_STYLE.md](CODE_STYLE.md): how the code is written.

Code, in this order (the same order as the package docstrings and the guide):

1. `examples/controllers/template_python_controller.py` and `mrm/control/api.py`: what a controller gets (codes
   in) and returns (codes out).
2. `mrm/plant/optics.py`, `ring_rows_nb`: the add-drop ring field calculation, including backscatter.
3. `mrm/plant/phase.py`, `ring_optics_row`: where the ring's phase, coupling and loss come from.
4. `mrm/plant/selfheat.py`, `solve_ring_temperature`, and `mrm/plant/thermal.py`: self-heating and the heat paths.
5. `mrm/plant/sensing.py`, `monitor_tick_row`: from drop power to a monitor ADC code.
6. `mrm/sim/kernel_ref.py`, `ReferenceEngine.step`: one tick in plain NumPy. `mrm/sim/kernel.py` is the fast copy
   of the same tick; skip it until you need it.
7. `mrm/sim/runner.py`, `run`: one closed-loop run of B worlds.

### One tick through the modules

At each tick the engine calls the plant modules in a fixed order (the full list is in the docstring of
`mrm/sim/kernel.py` and in [ENGINE.md](ENGINE.md), section 4):

```text
controller edge (mrm/core/timing.py)   sensors release their samples, the controller steps, commands are queued
  → scenario and laser                 disturbance signals and the laser lines for this tick
  → phase (mrm/plant/phase.py)         ring temperature and bias → round-trip phase, coupling, loss
  → heater (mrm/plant/electrical.py)   heater code → heater power
  → heat step (mrm/plant/thermal.py    one implicit step: the heat paths (sums of RC lags) carry the heat, and
     and mrm/plant/selfheat.py)        every ring's temperature is solved with the light it absorbs; the walk of
                                       the light along the bus (mrm/plant/optics.py) runs inside this step
  → sensing (mrm/plant/sensing.py)     drop power → photocurrent → ADC code, queued for the next controller edge
  → Truth and metrics                  the true state and the score (the controller never sees them)
```

## 1. Units, naming, provenance

### Use SI inside the code

Internal values use m, s, W, K, Hz, A, V, rad and J. **Hz always means cycles per second.**
Name angular rates `*_rad_per_s` so that the two cannot be confused.

Configuration keys carry their units in their names, following `<name>[_<qualifier>]_<unit>`.
The allowed tokens come from the single table in `mrm/params/units.py`:

```text
nm um mm m pm s ms us ns ps hz khz mhz ghz thz w mw uw nw a ma ua na v mv ohm f ff k c db dbm
pct frac per rt ev
```

Combine tokens with `per` and `rt` for compound units. `_c` denotes Celsius, an affine unit;
`_db` and `_dbm` are logarithmic units. Unitless keys have no unit suffix. `mrm.params` converts
configuration values to SI **exactly once**.

### Record where every number came from

The allowed provenance labels are a closed set. Their ranking, strongest first, is:

```text
printed > read-off > calibrated > inferred > solver > literature > assumed
```

Three additional labels are allowed: `derived` inherits the weakest provenance of its inputs;
`override` marks an override; and `private` marks private data. Generation tags are also a closed set:
`G25 G26 EVK100 L20 M1000 Guide OCP OCI GF PAT none`.

Every preset leaf must be a mapping with these fields:

```text
{value, provenance, generation, source, range?, alternatives?, note?, textbook?, disagrees?}
```

A question mark denotes an optional field. `source` names an entry in `mrm/params/sources.yaml`.
Bare numbers are allowed only in overrides, where the loader tags them `override` automatically.
The loader enforces these rules.

## 2. Package layout and ownership

The divisions follow responsibility: `params/` builds a device, `plant/` models its physics,
`scenarios/` supplies disturbances, `control/` supplies decisions, and `sim/` coordinates the run.
`extraction/` produces offline solver results; `calibration/` fits parameters; `validation/` checks them.
The file-level responsibilities are:

```
mrm/
  units.py                 physical constants, conversions (dB, dBm, wavelength<->frequency), unit-suffix parser
  materials.py             Si/SiO2 n(lambda,T) (Li 1980 shape on Frey/Franta level), dn/dT, alpha_L, k(T), rho*c,
                           plasma dispersion (Nedeljkovic 2011 @1.31um; Soref-Bennett switch), TCR by heater family,
                           Ge PD dark current law. Coefficients from mrm/data/materials.json (the cleaned copy of
                           docs/research/02_materials.json with its verification-log corrections; CHANGELOG).
  params/
    units.py               suffix grammar and conversion table
    schema.py              every key: type, enum, unit, required/forbidden under each fidelity choice
    sources.yaml           source keys (mirror of the textbook's tags + literature + GF + patents)
    config.py              load(preset|path, overrides, alts) -> Config; extends:, alternatives, provenance checks
    relations.py           the five relation groups (R1..R5) that tie the preset numbers together; closure /
                           OverdeterminedError
    realize.py             realize(cfg, batch, seeds) -> PlantParams: uncertainty worlds (B,), variation
                           (B,N)/(B,L), geometry Jacobian sampling, relation solve, thermal MIMO assembly,
                           latency->ticks, timing assertions, self-heating lag bound
    derive.py              nominal B=1 view for docs; diagnostics (implied index weight, bistability ratio, ...)
    symbols.py             the math symbol (LaTeX) of each preset input and PlantParams field, for the variable map
    table.py               PARAMETERS.md generator, `mrm params show|table|explain|diff|validate|keys`
    presets/*.yaml         g26 (default), g25_test_vehicle, the named corners of the uncertainty set, teaching_*
                           (self-heating off); g26_calibrated and g25_test_vehicle_calibrated (GENERATED by
                           mrm.calibration)
    presets/private/       git-ignored, provenance `private`
  core/
    channels.py            ChannelSpec, channel registry, flat int64 ABI layout generator
    timing.py              integer-tick clock, latency FIFOs, event order at a controller edge
    rng.py                 counter-based RNG (Philox4x32-10) usable in numba, stream-id enum
  extraction/              offline solvers -> mrm/data/extracted/ (never imported by runtime code)
    modes.py coupling.py thermal_fem.py mor.py junction.py common.py (fdtd_check.py, optional, and run_all.py are
                           not built)
  data/extracted/          committed artefacts + JSON sidecars (schema id, sha256, inputs, tool versions, errors)
  plant/
    optics.py              exact add-drop ring with CW/CCW backscatter; the bus
    phase.py               ring state (T_ring, T_heater, junction levels) -> optics inputs per ring and line: phase
                           model, K^2 at the line, alpha_i(V, T) and the heat fraction eta, the thermal shift
    thermal.py             ROM load/assemble/scale; exact ZOH and FOH per Foster pole; interim 3-pole model
    selfheat.py            implicit per-ring self-heating solve with branch continuation
    electrical.py          heater driver laws (voltage|current|power|dac_pwm), DAC INL/DNL, slew, settle, TCR,
                           compliance; bias DAC; junction runtime (LUT); data-activity heat
    sensing.py             PD (responsivity, Arrhenius dark current, shot noise), AFE (rc | integrate_dump, exact
                           noise discretization), ADC (bits, offset, gain, INL/DNL, PGA, saturation), temp sensors,
                           heater readback, faults
    datapath.py            ones count, bit-statistics sampler, error counter
    laser.py               line table, static error, drift (random walk + linear), RIN, power drift, line enable
    io.py                  channel maps, controller pipeline FIFO, observation packing, action clipping (kernel I/O)
    plant.py               Plant: Python API over the kernel; Truth; settle (settled warm start)
  scenarios/               parametric generators evaluable inside numba + named library + registry;
                           calibrate.py (aggressor power from a stated victim trajectory); compression checks and
                           the convergence certificate
  control/
    api.py                 ControllerSpec, Observation, Controller protocol, BaseController, PerRing/PerDie
                           adapters, telemetry, CompositeController
    registry.py            name -> class
    knowledge.py           controller_interface.knowledge tokens and the filtered view ControllerSpec.nominal
    models.py              linearize(nominal_cfg, op, Ts) -> (A,B,C,D,delay) for model-based controllers
    calibration.py         simulated factory calibration (warm/cold start)
    fxp.py lfsr.py         fixed-point toolkit and the reference fixed-point PI FxpPi
    fxp_mlp.py             fixed-point MLP inference helper: quantize, packed int64 form, numba eval
    cli.py                 `mrm golden | bridge | vectors`: golden traces, bridge and vector utilities
    baselines/             one file per controller family (section 6)
  sim/
    runner.py              run(cfg, controller, scenario) -> RunResult; Python-controller slow path;
                           solve_locked_steady_state (the locked steady state, no time stepping);
                           warm_start_codes (settled warm start)
    state_spec.py          the registry of every kernel array and row (name, meaning, unit, shape, writer); the
                           kernel takes its layout from it
    kernel.py              get_run_block -> run_block: a Python driver that runs the compiled per-chunk function
                           over the worlds on a thread pool (plant substeps + sensor sampling + compiled controller
                           + metric accumulators)
    kernel_ref.py          the pure-NumPy reference of the same tick, bit for bit
    metrics.py             streaming accumulators and summarize()
    experiment.py          run | sweep | mc | compare | robust | acceptance | locked_steady_state;
                           mrm.run(); the compression gate and convergence experiment; --dry-run estimates
    manifest.py            runs/<UTC>_<hash8>/: experiment.resolved.yaml, manifest.json, metrics.json, trace.h5,
                           inputs/, report.txt (convergence.json); compare_runs for `mrm rerun`
    cli.py                 `mrm run | sweep | mc | compare | robust | acceptance | rerun`
    gym_env.py             vectorized gymnasium env, registered as mrm/MRM-v0
    bridge.py              binary lockstep bridge for external controllers, C header and C reference PI
    vectors.py             golden vectors (record, .npz/.hex/.layout.vh, replay through the protocol or the ABI)
  waveform/                short time-domain TCMT + driver + junction RC runs; normalized-coordinate LUT
  validation/
    targets/{calibration,validation,bounds,excluded}.yaml  (the frozen targets, each file with a sha256)
    replicas.py oracle.py targets_io.py
    report.py sensitivity.py cli.py   the generated report docs/VALIDATION.md (`mrm report`): cached
                           parts in docs/validation/<preset>/, Jacobian SVD, tornado / Sobol, rank stability
  calibration/             fits that may read ONLY targets/calibration.yaml: core.py (least squares,
                           profile likelihood, hypothesis comparison, half-prior rule), groups.py (one problem builder
                           per fit group), pipeline.py, preset.py (writes the calibrated presets), report.py
                           (before/after), cli.py (`mrm calibrate`); the fit record latest_fit.json
  cli.py                   `mrm ...` entry point
experiments/               experiment YAMLs (acceptance/, examples/)
examples/                  controller templates (controllers/) + tutorial notebook (tutorial.ipynb)
tests/{verification,closure,calibration,validation,golden,perf,hdl}/   (hdl: Icarus replay of the golden vectors)
webviz/                    the interactive web viewer: generate.py writes site/data/ from the model; site/ is the
                           static page (`npm --prefix webviz run regen`, `npm --prefix webviz run check`)
docs/ ARCHITECTURE.md ENGINE.md EXPLAINED.md KNOWN_ISSUES.md CODE_STYLE.md (how the code is written)
      PARAMETERS.md (generated) PARAMETERS_FORMAT.md VALIDATION.md (generated, g26)
      VALIDATION_g26_calibrated.md (generated) validation/<preset>/ (report data and charts)
      VARIABLES.md and variables/ (the variable map, generated by tools/gen_variable_map.py)
      RENAMES.md (old -> new names of the readability pass) CHANGELOG.md QUESTIONS_FOR_LIGHTMATTER.md
      guide/ (the reading guide, LaTeX and PDF) handoff/ research/ (historical design records)
tools/memguard.py          all solver runs go through it (16 GB machine)
tools/gen_variable_map.py  writes docs/VARIABLES.md, docs/variables/data.json and docs/variables/symbols.json
                           from mrm/sim/state_spec.py and the preset
tools/perf_*.py            the performance tools (perf_scenarios.py and perf_compare.py check that the engine is
                           bit-identical); clear_numba_cache.py, scorecard_compare.py
```

## 3. Time scales

The simulator treats each process at the time scale needed for controller work. It solves very fast
optics as a steady state during a control run; the separate `waveform/` path resolves short optical
and electrical transients.

| Physics | Time scale | Treatment |
|---|---|---|
| Photon lifetime | ~3 ps | Adiabatic: exact steady-state ring per substep in control mode; TCMT ODE in `waveform/` |
| NRZ bits (50 Gb/s) | 20 ps | Mark-density average per substep; `data_modulation=lut` correction |
| Junction RC | ~4 ps | Waveform mode only |
| Waveguide-core self-heating | 0.05–1 µs | ROM poles and an implicit solve |
| AFE / integrating ADC | 1–10 µs | Sensing state |
| Heater/ring thermal | 1–100 µs | ROM poles, exact ZOH/FOH |
| Crosstalk, substrate, package | 0.1–10 ms | Slow ROM poles |
| Ambient, TEC, laser drift | Seconds to hours | Scenario signals; `time_compression` |

Here **TCMT** is temporal coupled-mode theory; **ROM** is a reduced-order model; **AFE** is the
analog front end. A zero-order hold (**ZOH**) keeps an input constant over a step; a first-order hold
(**FOH**) interpolates it linearly. “Mark density” is the fraction of data bits that are ones.

The plant tick is `sim.dt_plant_s`: 1 µs by default, with 0.1 µs allowed. Time is stored as an
`int64` tick count, so time never drifts. See KNOWN_ISSUES.md for scoped features that are not yet wired into the kernel,
including the data-modulation LUT correction.

## 4. Config tree

The top-level configuration sections are:

```text
meta fidelity sim channel_plan laser optical_path
ring {geometry optics junction heater thermal monitor bias_dac}
driver temp_sensors layout array uncertainty variation
controller_interface controller scenario metrics experiment datapath
```

`mrm/params/schema.py` is the authoritative list of keys. `docs/PARAMETERS.md` is generated from that
schema and the presets.

Some keys of the first design are gone: `ring.optics.absorbed_fraction` (the heat fraction now comes from a
loss budget), `d2n_eff_dt2_per_k2` and `controller_interface.*_latency_samples` (latencies are now in seconds and
become integer ticks). `array.pitch_um` is replaced by `layout.sites`, and
`temp_sensor` by `temp_sensors`.

## 5. Artefact formats

An **artefact** is a saved offline solver result. All artefacts live in `mrm/data/extracted/`.
Each has a JSON sidecar recording its schema, the data file's SHA-256 hash, geometry/material/film
inputs, tool versions, mesh or resolution, validation numbers and `created_utc`.
`realize()` verifies the hash before using it.

### 5.1 Modes: `modes_<xsec>.json` (schema `modes/1`)

Each cross-section hypothesis (`rib160`, `rib200`, `strip400x200`, …) has one file. It contains both
a straight-waveguide solution and a bent-waveguide solution at `r_ref_um`:

```
xsec: {width_nm, film_nm, slab_nm, sidewall_deg, clad}, r_ref_um, lambda0_nm, t_ref_c
n_eff_poly_lambda: [c0, c1, c2]     n_eff_bent(lambda) = c0 + c1*(lam-lam0) + c2*(lam-lam0)^2, lam in nm
n_g, dn_g_dlambda_per_nm
dneff_dt_per_k: {t_c: [...], value: [...]}        thermo-optic of the bent mode vs absolute temperature
alpha_l_per_k: {t_c: [...], value: [...]}         clamped expansion used
dlambda_dt_pm_per_k: {t_c: [...], value: [...]}   = (lam/n_g)*(dneff/dT + n_eff*alpha_L)  (H_solver)
index_weights: {si: , sio2: , ...}                dn_eff/dn_i;  check: sum n_g,i dn_eff/dn_i = n_g
power_fraction: {si: , ...}, mode_area_um2, alpha_rad_db_per_cm (bend radiation at r_ref)
jacobian: {d: [width_nm, film_nm, slab_nm], of: {lambda_res_nm_per_nm: [...], n_g_per_nm: [...], ...}}
crosscheck: {tidy3d_n_eff, femwell_n_eff, delta, same_r_ref: true}
```

### 5.2 Coupling: `coupling_<xsec>.json` (schema `coupling/1`)

For each coupler (`through` and `drop`), store these values at its design gap:
`kappa2_per_pass` at `lambda0`, `dln_kappa2_dlambda_per_nm`, `dln_kappa2_dgap_per_nm`,
`coupler_phase_rad`, and a table `kappa2(lambda_nm, gap_nm)`.

The calculation uses a curved-gap supermode/coupled-mode integral along the bend.

### 5.3 Thermal ROM: `thermal_rom_<family>_<film>.npz` (schema `thermal_rom/1`)

The thermal ROM describes how heat from each source reaches each temperature observer over time.
Its arrays are:

```
taus_s             (K,)        shared pole set (must span 10 ns .. 10 ms)
sources            [heater, waveguide_absorption, junction, driver, aggressor]
observers          [ring_mode_weighted, heater_resistor, monitor_pd, temp_sensor]
self_res           (S, O, K)   K/W, source of ring i -> observer of ring i (observer sites at their default offsets)
cross_r_um         (D,)        centre-to-centre distance grid (radially symmetric approximation, valid d > 2R)
cross_res          (S, O, D, K) K/W
site_offsets_um    {monitor_pd: [x, y], temp_sensor: [x, y], driver: [x, y]}  defaults used for self_res
```

For any source–observer pair, the step response is `sum_k r_k (1 - exp(-t/tau_k))`.
Residues may have mixed signs when the source and observer are at different locations.

The sidecar records the family, stack, mesh and errors relative to the full finite-element model
(FEM). Required error records are time-domain error below 2% of steady state over 10 ns–10 ms;
frequency-domain error below 0.3 dB and 3 degrees from DC to 1 MHz; and the eight-ring bank's
superposition error.

There are three families: `inring_noundercut` (default), `silicide_undercut` and `metal_above`,
with 160 and 200 nm films. All six artefacts exist in `mrm/data/extracted/`.

A preset chooses its thermal model with `fidelity.thermal_model`:

- **`interim`** is the default, including for `g26`. `thermal.py` builds a three-pole interim model
  from preset values.
- **`rom`** loads the artefact named by `ring.thermal.rom_file`.

An artefact can replace the interim values only through the calibration split: only the targets in
`calibration.yaml` may be fitted, and the held-out targets stay frozen. G26 therefore stays on `interim` until the FEM ROM passes that process.
The eight-ring superposition check currently fails at 13.9% against a 10% limit, and VAL-8a/b/c fail.
ROM mode also loses the per-ring heater-efficiency spread and per-world family, film and stack
hypotheses, because it uses one artefact per run.

In ROM mode, the die-level time scales apply to a mixed-sign source–observer pair as
a whole when its poles straddle the layer split. See `ThermalScales` in `mrm.plant.thermal`.

### 5.4 Junction: `junction_<xsec>.json` (schema `junction/1`)

The `v_reverse_v` grid spans −0.5 to 4.0 V; negative values mean forward bias. The saved junction
occupies the full ring circumference and records `dneff` relative to 0 V, absolute
`alpha_fca_per_m`, `capacitance_f_per_m`, `r_series_ohm_m`, the doping profile, the method
(`devsim` or `analytic`), and the devsim-versus-analytic depletion-width check.

At runtime, scale it by `ring.junction.fill_fraction` (calibrated by CAL-5).

## 6. Controllers shipped with the model (`mrm/control/baselines/`)

Each controller below has a Python implementation. **(c)** means it also has a compiled Numba
implementation using the flat application binary interface (ABI). The registry name is the name an
experiment file or `runner.run` uses.

1. Drop-port set-point PI, normalized to a fraction of the peak **(c)**: `pi_lock.py`, `pi_lock` / `pi_lock_c`.
2. Heater-dither lock-in using 1f/2f for peak locking **(c)**: `dither_lockin.py`, `dither_lockin` /
   `dither_lockin_c`.
3. Bias-dither lock-in through `tx.bias`, using the `fast_kernel` form (a compiled hook that runs on each new
   monitor sample): `bias_dither.py`, `bias_dither`.
4. Ramp-and-maximize acquisition (Lightmatter patent PAT-3), followed by lock hand-off: `ramp_acquire.py`,
   `ramp_acquire` / `acquire_then_lock`.
5. Extremum seeking by perturb and observe: `extremum_seeking.py`, `es_po` / `es_sine`.
6. Bit-statistics lock (Sun et al.) using `datapath.bitstat_above`: `bitstat_lock.py`, `bitstat_lock`.
7. Temperature-sensor feed-forward plus PI: `feedforward.py`, `feedforward_pi`.
8. Kalman filter plus LQG/MPC using `mrm.control.models.linearize`, with only permitted nominal knowledge
   **(c)**: `kalman_lqg.py`, `kalman_lqg` / `kalman_lqg_c`; `mpc.py`, `mpc` / `mpc_c`.
9. Dual-actuator mid-ranging: fast bias and slow heater: `midranging.py`, `midranging`.
10. WDM bank supervisor: sequential acquisition, wrong-line detection, barrel-shift remapping (PAT-1)
    and sequencing around supply limits: `wdm_supervisor.py`, `wdm_supervisor`.
11. The oracle, located in `mrm/validation/oracle.py`. It is not registered as a controller.

The fixed-point PI `FxpPi` (`mrm/control/fxp.py`) is registered as `fxp_pi`.

## 7. Accuracy rules

1. **Every number needs provenance.** A missing provenance label is a bug, enforced by the loader.
2. **Prefer the strongest applicable evidence.** Measured Lightmatter data takes priority over solver
   output, which takes priority over literature, which takes priority over assumptions. If a printed
   value is not a measurement, or violates a physical bound calculated from materials data, use a
   bound-respecting default. Keep the printed value as a required robustness alternative and state
   the gap in `docs/VALIDATION.md`.
3. **Compute dependent quantities after sampling.** Only the relation solvers may compute them.
4. **Put each test in its class.** `tests/verification/` (is the code right), `tests/closure/` (are the published
   numbers consistent) and `tests/golden/` block CI. `tests/calibration/`, `tests/validation/` and `tests/perf/`
   are reported only.
5. **Keep calibration separate from validation.** Calibration may read only `targets/calibration.yaml`.
   Held-out targets are frozen by SHA-256.
6. **Report uncertainty with controller claims.** Include the nominal result, worst named corner
   and quantiles over the uncertainty set. The first line of every controller report must say that
   performance is conditional on the assumed thermal dynamics.
7. **Respect the 16 GB machine's memory budget.** All solver runs go through `tools/memguard.py`;
   use `--heavy` above 1 GB. The heavy-job lock permits only one heavy job at a time. Heavy jobs
   default to a 6 GB cap; other jobs default to 3 GB. Monte Carlo runs record metrics only.

## 8. Scope of this version, and which rules take precedence

The design was reviewed by four independent critics. Their merged review,
[research/07_design_critique.md](research/07_design_critique.md), holds the design history: the requirements
(numbered P0-x, P1-x and P2-x there), the decisions between conflicting reviews and the questions for Lightmatter.
The scope of this version is below. Some items in it are not finished yet; [KNOWN_ISSUES.md](KNOWN_ISSUES.md) lists
them.

- Required, every core requirement: implicit per-ring self-heating; a heat fraction from a loss budget; one thermo-optic
  anchor with closure; realization after sampling; frozen calibration and held-out targets; integer-tick timing
  and event order; named I/O channels; online metrics; the compiled kernel; long-run substeps and time
  compression; data-path observables.
- Required: controller knowledge and Truth isolation with an oracle; the thermal ROM and its interim model; disturbance
  paths; the coupling, phase, junction, data-activity heat and monitor models; converters and sensors as
  hardware, with faults; actuator limits from hardware; WDM acquisition; counter-based noise; provenance and a
  strict schema; variation by path; experiment files and run manifests; the NumPy reference; the controller API
  and telemetry; the RL environment.
- Required from the fixed-point and HDL item: the fixed-point toolkit, golden vectors and the binary bridge. The Verilator
  HDL co-simulation is a documented stretch goal.
- Required smaller items: an exclusion bound for TPA, FCA and Kerr; backscatter per order; optics in solver-independent
  terms; temperature-dependent conductivity; the data-modulation LUT; multi-rate scheduling; hashed artefacts; the
  CLI and generated docs.
- Optional if time allows: parasitic mode families, optional sensing paths, a fixed-point plant export.
- Deferred, with a stub and a reason: the generic S-matrix netlist. Add it with the second photonic element.

The decisions of the review (its section 0) are adopted as written. If code and this document disagree, update one
of them in the same change and record the decision in [CHANGELOG.md](CHANGELOG.md). For implementation gaps,
consult [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

The evidence behind the parameters is in `docs/research/01..06`: 01 covers Lightmatter publications
and the GF 45SPCLO platform; 02 covers materials, with `02_materials.json`; 03 covers compact-model
equations; 04 covers thermal data and control literature; 05 covers tools; and 06 audits the textbook.
The project brief is [research/00_capstone_brief.md](research/00_capstone_brief.md).
