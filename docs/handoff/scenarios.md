# Handoff: scenarios (`mrm/scenarios/*`)

Owner files: `mrm/scenarios/{__init__,generators,library,registry,calibrate}.py`,
`tests/verification/test_scenarios.py` (60 tests after the adversarial review, about 7-12 s with a warm numba cache and
25-30 s cold; all pass with `NUMBA_DISABLE_JIT=1` too, about 15 s). Section 10 lists what the review changed. Contract: 07 P0-9, P0-10, P0-11, P1-4, P1-12, P1-8 (signals), P1-10 (fault
masks), D6. This file is the **engine contract**: what `mrm/sim/kernel.py`, `runner.py`, `plant.py` and the NumPy
reference kernel consume.

## 1. What a scenario is

A `Scenario` is a set of **components** (parametric generators) on a fixed table of named **signals**, plus
**aggressors** (site, victim ring, target, calibrated power), **defaults**, `time_compression`, an optional
`ResetSpec` (cold/warm start) and provenance. `Scenario.compile(...)` turns it into a `ScenarioArrays` (plain numpy
arrays) that the numba kernels evaluate at any plant tick. `Scenario.render(...)` / `ScenarioArrays.render(...)`
give a `DisturbanceFrame` (arrays on a tick grid) for the NumPy reference, plots and tests; a `DisturbanceFrame`
(recorded profile) is replayed through the same kernel with `Scenario.from_frame(frame)` (07 P0-9).

## 2. Signal table (ids are append-only; `mrm.scenarios.SIGNALS`)

| id | name | level (width) | unit | default | consumer, meaning |
|---|---|---|---|---|---|
| 0 | `thermal.t_base_k` | die (1) | K | 298.15 (assumed 25 degC; `scenario.ambient_c` overrides) | thermal: absolute T_base of the ROM step. 07 P1-4 path 1 (package/TEC). Time-compressible |
| 1 | `thermal.xpu_rise_k` | die (1) | K | 0 | thermal: die-wide rise from XPU/ASIC power events, **added to T_base**. 07 P1-4 path 3 |
| 2 | `thermal.aggressor_w` | aggressor (A) | W | 0, clipped >= 0 | thermal: power of each aggressor site into the ROM `aggressor` column. 07 P1-4 path 2, D6. Piecewise linear (exact FOH) |
| 3 | `laser.dfreq_hz` | line (L) | Hz | 0 | optics: added to each line frequency (positive = higher frequency = bluer). 07 P1-4 path 4 (optical only). Compressible |
| 4 | `laser.power_delta_db` | line (L) | dB | 0 | optics: line power x 10**(v/10). Compressible |
| 5 | `laser.line_enable` | line (L) | frac | 1, clipped [0, 1] | optics: line power factor (0 = off; ramps allowed). 07 P1-12 |
| 6 | `optical.input_loss_delta_db` | die (1) | dB | 0 | optics: every line x 10**(-v/10) (fibre loss / re-mate). Compressible |
| 7 | `data.ones_density` | ring (N) | frac | 0.5, [0, 1] | datapath/optics mark density. 07 P0-11 |
| 8 | `data.transition_density` | ring (N) | frac | 0.5 (`datapath.transition_density` overrides), [0, 1] | electrical CV^2 and driver heat. 07 P1-8 |
| 9 | `data.swing_scale` | ring (N) | frac | 1, >= 0 | electrical: junction swing = `driver` nominal swing x v (0 = squelch). 07 P1-8 |
| 10 | `data.driver_enable` | ring (N) | frac | 1, [0, 1] | electrical: driver on when >= 0.5. 07 P1-8 |
| 11 | `tx.bias_offset_v` | ring (N) | V | 0 (`scenario.bias_offset_v` overrides) | electrical: `BiasBank.step(v_offset=...)`. Always allowed (07 P0-7) |
| 12 | `tx.bias_v` | ring (N) | V | NaN = not driven (`scenario.bias_v` sets it) | electrical: absolute bias, only when `Scenario.drives` has `tx.bias` |
| 13-15 | `fault.adc_drop`, `fault.adc_stuck`, `fault.adc_saturate` | ring (N) | flag | 0 | sensing (not built yet): mask active when >= 0.5 (07 P1-10) |
| 16 | `fault.dac_glitch_codes` | ring (N) | code | 0 | **proposal** for electrical: additive heater code after the DAC FIFO for that tick |
| 17 | `fault.controller_reset` | die (1) | flag | 0 | runner: reset the controller at the next edge when >= 0.5 |

Signal value = default (0 on elements an `absolute` component covers) + sum of components, clipped to [lo, hi]. Two
absolute components on one element are a compile error. The XPU rise and the base temperature are separate signals
(07 P1-4 "never merged"); the plant may add them for the ROM step, but records and attribution stay separate.

## 3. Kernel API (numba; all time in TICKS of the compile dt)

```python
from mrm.scenarios.generators import scen_init_b, scen_eval_b, scen_eval_signal_b, scen_next_break_b
arr = scenario.compile(dt_s=dt, n_rings=N, n_lines=L, batch=B, seed=experiment.seeds.scenario,
                       aggressor_sites=[names of the ROM aggressor sources in column order])
cache = arr.new_cache()            # (B, NK) float64, per run (random-walk / random-level caches)
out = arr.new_out()                # (B, V) float64, pre-filled with elem_default  (or scen_init_b per row)
# inside the engine kernel, per batch row b (prange over b), bb = b if arr.comp_f64.shape[0] > 1 else 0:
scen_eval_b(tick, compression, sig_i64, sig_f64, elem_default, comp_i64, comp_f64[bb], tab_f64[bb],
            keys[b], cache[b], out[b])                        # every signal that has components
scen_eval_signal_b(sig_id, tick, compression, sig_i64, sig_f64, elem_default, comp_i64, comp_f64[bb],
                   tab_f64[bb], keys[b], cache[b], out[b])     # one signal (inlined into the caller)
kb = scen_next_break_b(sig_id, tick, compression, sig_i64, comp_i64, comp_f64[bb], tab_f64[bb])   # int64
value = out[b, arr.layout[name][0] : arr.layout[name][0] + arr.layout[name][1]]
```

- `ScenarioArrays.kernel_args(cache)` = `(compression, sig_i64, sig_f64, elem_default, comp_i64, comp_f64, tab_f64,
  keys, cache)`, the argument order of the batched `scen_eval(tick, *args, out)`.
- Arrays: `sig_i64 (S, 5)` = [offset into out, width, first component, n components, SI_CLIP]; `sig_f64 (S, 2)` =
  [clip lo, clip hi]; SI_CLIP = 1 when the components may push the sum past a finite clip limit (decided at compile
  from conservative value ranges; `ScenarioArrays.clipped_signals`); `elem_default (V,)`; `comp_i64 (C, 11)` and `comp_f64 (Bs, C, 15)` (field indices `CI_*`, `CF_*` in
  generators.py); `tab_f64 (Bs, NT)`; `keys (B, C) uint64`; Bs is 1 (shared) or B (per-world parameters, e.g. a
  calibrated aggressor). Constants: `SIG_*` ids, `NO_BREAK = 2**62`.
- **Time semantics.** Every time field is converted to ticks at compile (a time within 1e-6 tick of a whole tick is
  snapped onto it), so edges given in seconds land on whole ticks exactly. Values are right-continuous at tick
  instants. The defined input over [k, k+1) is the **linear interpolation between the values at ticks k and k+1**
  (FOH; a jump between ticks is a one-tick ramp). `scen_next_break_b(sig, k)` returns the largest kb > k such that
  the tick-sampled signal is exactly linear on [k, kb] (an edge at or just before tick m gives kb = m - 1; returns
  k + 1 near an edge, `NO_BREAK` if linear for ever; curved components -- the smooth kinds and ramps under the
  `bernoulli_td` map -- break every `CF_HSMOOTH` ticks so that their summed deviation from linear stays below the
  signal's `foh_tol`; a signal flagged SI_CLIP breaks every tick, because a kink where the sum meets the clip limit
  cannot be located without the values). With 07 P0-10 `substep_policy: auto`, never step a FOH input past
  kb; fixed-step and multi-tick runs then see the same input.
- **The engine must run at the compile dt** (time fields are in ticks of `arr.dt_s`). Recompile for another dt.
- **Cost** (measured on this machine at load average 11-14, so inflated maybe 2-4x): all 8 components of a mixed
  scenario (A1 power, soak, XPU, per-line laser random walks, random ones density, bias ripple) 1.4 us per tick for
  N = L = 8; one signal with one component 80-110 ns; per-line random walk 340 ns for 8 lines. Recommendation:
  evaluate a signal only at its `scen_next_break_b` ticks and interpolate linearly in between (exact by the
  contract), and evaluate data-path signals at controller edges if the datapath only needs them there. Signals
  without components cost nothing after `scen_init_b`. The helpers are `inline='always'`: a non-inlined numba call
  with many array arguments costs tens of ns.
- Random draws: counter RNG (mrm/core/rng.py), key `stream_key(seed + b, stream_id("scenario_<name>"))`, counter
  (tick = knot/interval index, ring = element, slot = component slot). Element b of a batch equals a B = 1 run with
  seed + b (tested); results do not depend on evaluation order (tested with random-order evaluation).

## 4. Aggressor calibration (07 P1-4, D6) -- what the engine / realize must do

An aggressor with a target (`a1`, `a2`, or `scenario.target`) has no power until calibrated; `compile()` raises.

```python
from mrm.scenarios import calibrate_scenario, foster_from_system
resp = foster_from_system(thermal_system, victim_ring=a.victim_ring, site=a.site)   # list per world (duck-typed)
sc, cal = calibrate_scenario(sc, {a.site: resp}, model={
    "s0_m_per_k": pp.thermal.dlambda_dt_s0_m_per_k[:, a.victim_ring],   # (B,)
    "beta_per_k": pp.thermal.dlambda_dt_beta_per_k[:, 0],               # (B,)
    "t_ref_k":    pp.thermal.dlambda_dt_t0_k[:, 0],                     # (B,)
    # optional: "ring_t0_k": pp.thermal.t_ring_lock_k[:, a.victim_ring]  (see the doubt in section 8)
})
cal[a.site].residual_rms_k, .residual_max_k, .residual_rms_m, .record()   # tagged "calibrated"
```

With several aggressors whose victims differ, pass `model={site: {...}}` (one model per site; s0 is per victim ring).

- `foster_from_system` reads `observer_index`, `source_index('aggressor', name=)`, `lag_obs`, `lag_src`, `taus_s`,
  `res_k_per_w` of `mrm.plant.thermal.ThermalSystem`; any callable step response g(t) (K/W) also works.
- The power is periodic piecewise linear (256 knots per period by default), >= 0 (a heater only heats), solved by
  BVLS with a Tikhonov difference penalty; the periodic steady state is exact (closed-form geometric sum) for Foster
  responses. Verified against an exact FOH simulation (2e-11 K) and end to end through the kernel.
- Against the interim model (aggressor 20 um from ring 0: one 0.46 ms pole, 549 K/W), A1 under resonance referencing
  with s0 = 70 pm/K needs about 185 mW peak, 69 mW mean, and reaches 0.013 K rms (0.28 K max at the cusp).
- Start the run at least `settle` = a few slowest aggressor-path poles before scoring (CAL-13a is scored in periodic
  steady state with t = 0 at the cusp; the A1 trajectory has its cusp at t = 0).

## 5. Time compression (07 P0-10)

- Only `slow` components on `thermal.t_base_k`, `laser.dfreq_hz`, `laser.power_delta_db`,
  `optical.input_loss_delta_db` are compressed (they evaluate at t_real = c x t_sim). Plant time constants, the
  controller rate, the aggressor, XPU and data paths are never compressed (a `slow` flag on them is an error).
- Load-time rule (in `compile`): compressed ramp <= 1 % of the fast-transient rate (default the slide-10 peak 2127
  K/s; laser frequency via 15 GHz/degC). The soak allows c <= 173. dB signals cannot be checked (warning).
- Mandatory convergence experiment: `plan = convergence_plan(sc)` -> {1, 10, 100: (scenario, duration of one soak
  cycle)}; run each on the same plant and seeds; `rep = assess_convergence(sc, {c: metrics}, plant_fingerprint=
  cfg.fingerprint(), code_fingerprint=...)` with metrics `max_abs_detuning_pm, wrong_line_events, fsr_hop_events,
  order_changes, heater_range_used_codes, integrated_errors` (a value may be a per-ring vector, compared element by
  element: mrm.sim.experiment passes max_abs_detuning_pm per ring, `CONVERGENCE_PER_RING`); store
  `rep.certificate.to_dict()` in the run manifest; **reporting code calls `require_compression_validated(sc,
  certificate, plant_fingerprint=..., code_fingerprint=...)`** before publishing a compressed result (with a code
  fingerprint, a certificate of other code, or one that records none, is refused). Tolerances
  (`DEFAULT_CONVERGENCE_TOLERANCES`, (abs, rel[, cap])) are design choices: 0.2 pm or 5 % on max detuning, at most
  0.5 pm (since 2026-10-16: the |d| about the line is dominated by a lock's static offset, so 5 % of a 125 pm
  side-of-fringe offset allowed +-6.3 pm), exact event counts, 16 codes or 1 % on heater range, 1 error or 10 % on
  integrated errors.

## 6. Bindings, reset, config

- `Scenario.drives` -> `resolve_bindings(..., scenario_drives=)` (`("tx.bias",)` when `tx.bias_v` is set);
  `Scenario.offsets` -> `scenario_offsets` (`("tx.bias",)` for `tx.bias_offset_v`). The scenario never writes
  `laser.tune` or `tx.assignment`; laser disturbances use `laser.dfreq_hz`.
- `Scenario.reset` (`ResetSpec`: start cold|warm, heater_init zero|factory, factory_cal): cold start withholds the
  factory calibration (07 P1-1); the runner/`control.calibration` act on it.
- `registry.from_config(cfg)` reads `scenario.name | stack`, `scenario.params` (per name for a stack),
  `scenario.aggressor` + `scenario.target` (07 P1-4: `{units, trajectory: a1|a2, reading, ...}`),
  `scenario.time_compression`, `scenario.ambient_c`, `scenario.bias_v`, `scenario.bias_offset_v`,
  `datapath.transition_density`; works on a `Config` or a plain mapping.

## 7. Named library (`mrm.scenarios.available()`; aliases are the textbook Ch17 ids)

| name (alias) | what | numbers and provenance |
|---|---|---|
| `a1_slide10_aggressor` (a1) | slide-10 on-die aggressor, target at the victim ring, power calibrated | T = 113.6 - 88.5 \|sin(pi t/0.1307 s)\| degC (read-off fit, 01 verif. log 5; CAL-13a), peak 2127, mean 1354 degC/s; 85.86 pm/K (15 GHz/degC, printed aside LM26-P); base 25 degC (printed lower end, assumed as aggressor-off) as the signal DEFAULT, so a stacked soak / cold start replaces it; alt. `waveform=symmetric_triangle`, `referencing=temperature`, `thermometer_fraction` |
| `a2_808_aggressor` (a2) | paper's 808 degC/s run | printed 808 degC/s, 25 -> >105 degC, 100 ms (LM26-P 3.2); readings `symmetric_triangle` (80.8 K, 1616 degC/s edges, default), `rectified_sine` (peak 2538), `one_sided_sawtooth` (808 degC/s over 99 ms, 1 ms return), `ramp_per_period` (808 degC/s both ways, 100 ms per ramp, 200 ms cycle; 04 D1 verification note, inferred) |
| `a1_slide10_aggressor_pd_adjacent` (a1_pd_adjacent), `a2_808_aggressor_pd_adjacent` (a2_pd_adjacent) | STRESS test of monitor dark-current runaway, not a realistic layout (added 2026-10-10 by the integrator, docs/handoff/integrator.md section 7) | A1 / A2 with the same arguments (except `site`) at the layout site `aggressor_pd_adjacent` (`library.AGGRESSOR_PD_ADJACENT_SITE`; g26: (0, 30) um, 5 um from ring 0's monitor PD, the former default). The default site is `library.AGGRESSOR_SITE` = `aggressor` (g26: (0, -30) um) |
| `laser_sweep_emulation` (a3) | optical-only sweep | 50 nm/s over 4 nm (printed SIB), period 0.44 s (read-off), 0.08 s ramps, 0.14 s plateaus (inferred); 699 GHz at 1310 nm |
| `soak` (a4) | TEC soak, compressible | 33-108 degC, 90 x 20.3 min, 0.123 degC/s, 30.45 h (read-off LM26-S slide 10; 01 D5) |
| `connector_remate` (a6) | 10 re-mates, loss redrawn | N(0.43, 0.35) dB per facet, 1.48 dB at mean + 3 sigma ([ECTC] via textbook Ch18, literature); interval assumed |
| `reloma_plane` (a7) | bias 1.0-3.75 V x 0-105 degC with the scenario owning tx.bias | printed axes (LM26-S slide 9); sweep time assumed; the base temperature is a triangle (up for even, down for odd bias index), never a 105 K jump |
| `squelch` (b1) | swing -> 0, average kept | OCI <= -15 dBm OMA (printed normative via OCP); residual 0 assumed; ADDITIVE (residual - 1), so it stacks with `traffic_start_stop` |
| `laser_failover` (b2) | step in line frequency and power | YOLE (qualitative); 10 GHz, -1 dB, gap 0 assumed |
| `xpu_power_step` (b3) | die-wide rise, raised-cosine edge | 10-120 degC/s within ~30 ms ([MRVL] via textbook REF, secondary); default 120 degC/s -> 4.58 K; shape assumed |
| `cold_start` (b5) | ambient uniform 0-105 degC per world, factory cal withheld | printed closed-loop range (LM26-S slide 9; textbook B5) |
| `warm_start` | factory cal passed, ambient 25 degC (signal default) | assumed |
| `training_pattern` (training) | a repeating word (default '10') in a window: ones and transition density of the word | 07 P0-11 training patterns, 04 3.3; densities derived exactly from the word; word assumed |
| `laser_drift` | 24 h drift: linear worst case (default) or random walk | 0.02 nm and 2.5 % per 24 h (printed limits, YOLE via textbook; CAL-12c/d); random walk sigma = limit/3.48 (P(exceed) ~1e-3, derived) |
| `fibre_loss_step` | deterministic loss step | 1.05 dB = 3-sigma re-mate change (derived) |
| `traffic_start_stop`, `idle` | data on/off, constant level | 07 P0-11, 04 3.3; states assumed |
| `ones_density_steps` | 0 -> 0.5 -> 1 every 200 ms (cycle or random) | Sun et al. VLSI 2015 Fig. 5 (printed); transition density 2p(1-p) (derived) |
| `bias_ripple`, `bias_step` | additive bias offset | assumed (no public number): 5 mV at 100 kHz; -50 mV |
| `line_enable` | per-line on/off schedule | 07 P1-12 |
| `aggressor_direct` | uncalibrated power waveform (square, sawtooth, triangle, rectified_sine, burst, const) | user's numbers (`assumed`) |

## 8. Deviations from the contract and doubts (please review)

1. **07 P0-9 "parabolic segments (the slide-10 aggressor)" and D6/P1-4 A1 "dT/dt a +-2200 degC/s sawtooth" are
   wrong** (01 verification log item 5; validation_targets handoff): the trajectory is a rectified sine. The
   generator set has `rect_sine` instead of a parabolic kind, and `aggressor_direct(waveform='parabolic')` raises
   with the reason. A1 is a temperature/resonance TARGET; the power waveform comes from the calibration.
2. **What the slide-10 degC means is ambiguous**, so `resonance` referencing needs one more choice the contract does
   not make: with beta != 0, the ring temperature from which the aggressor rise starts. Default: the base temperature
   (heater off). For a ring locked at T_lock pass `ring_t0_k` (PlantParams `thermal.t_ring_lock_k`); the rise then
   differs by about beta x/2 (about 7 % at the 100 K peak with beta = 1.4e-3/K). Under lock the ring temperature
   barely moves (the heater backs off), so the open-loop shift from T_lock is the disturbance the controller sees.
   I recommend `ring_t0_k = t_ring_lock` for controller studies.
3. **A1 under resonance referencing heats the ring by about 101 K** (s0 = 70 pm/K with beta), not 88.5 K: that is
   07 P1-4's point (1.2-1.3x under H_solver). With the base at 25 degC the ring reaches about 126 degC, above the
   0-105 degC junction range; BND-2 (order hop) and the self-heating branch logic should be checked at that point.
4. **The A1 base temperature is assumed** to be the 25 degC lower end. If the TEC was colder and the aggressor's
   mean heat lifted the minimum to 25 degC, the calibration would need `fit_offset=True` (supported, reported in
   `offset_k`), which moves the base, not the aggressor.
5. **A2 one-sided sawtooth is not reachable by a heater**: the 1 ms return of 80 K needs cooling; the calibration
   keeps p >= 0 and reports the residual (tested). That is the honest result of that reading.
6. **Time-compression reference rate**: 07 says "1 % of the fast-transient rate" without saying which. I use the
   G26 design fast transient (2127 K/s), the strictest value when stacking, never a stacked scenario's own
   (the one-sided sawtooth's 80,800 K/s return edge would loosen the rule 38x). dB signals (laser power drift,
   fibre loss) have no K/s equivalent and only warn; the convergence experiment covers them.
7. **XPU events as a temperature, not a power**: the only data ([MRVL], Ayar) are degC/s at the PIC, so the XPU
   path is a die-wide rise in K with an assumed raised-cosine shape. A package-level power input would need a
   package thermal model that does not exist.
8. **Laser drift default is the linear worst case** (the printed 24 h limit reached linearly, all lines together),
   not a stochastic model; the random-walk mode is calibrated to exceed the limit with probability ~1e-3.
9. `transition_density` is an independent signal; the library keeps it consistent with the ones density
   (2p(1-p) for i.i.d. bits, 0 at all-0 / all-1), but a user-built scenario can set an inconsistent pair.

## 9. Requests to other owners

- **engine (`mrm/sim/kernel.py`, `runner.py`, `plant.py`)**: consume the arrays as in section 3; add
  `thermal.xpu_rise_k` to `thermal.t_base_k` for the ROM base temperature (keep both in records); map
  `aggressor_sites` to `ThermalSystem.aggressor_src` by name; FOH the aggressor power (piecewise linear, exact); call
  `calibrate_scenario` after `realize()` when `scenario.pending_aggressors` is not empty (section 4); apply
  `laser.dfreq_hz`, `laser.power_delta_db`, `laser.line_enable`, `optical.input_loss_delta_db` to the line powers and
  frequencies; pass `drives`/`offsets` to `ControllerSpec.from_config(..., scenario_drives=, scenario_offsets=)`;
  honour `ResetSpec`; refuse to report compressed results without `require_compression_validated` (section 5).
- **plant electrical**: `data.swing_scale` is a factor on the nominal swing (not `swing_vpp` in V as 07 P1-8
  writes): the scenario does not know the preset swing. Squelch = 0 with the junction at the mean level of
  `driver.level_convention`. `fault.dac_glitch_codes` is my proposal for the `dac_glitch` interface you asked for.
- **plant sensing (not built)**: fault masks `fault.adc_drop|adc_stuck|adc_saturate` (>= 0.5 = active), (B, N).
- **params (schema)**: no new keys required. Optional: a `scenario.fast_rate_ref_k_per_s` setting if users should
  change the compression reference from a config (today: `Scenario.replace(fast_rate_ref_k_per_s=...)`).
- **validation (replicas)**: `aggressor_trajectory` can build its target with `library.a1_slide10_aggressor()`
  (`aggressors[0].target.printed_k / resonance_m / ring_rise_k`), which encodes CAL-13a exactly (cusp at t = 0).
  `laser_line_statistics` can render `laser_drift` (linear or random walk) for CAL-12c/d.
- **docs/CHANGELOG.md owner**: "07 P0-9 'parabolic segments' replaced by the rectified sine (01 verification log
  item 5); scenario kernels compute in ticks; time-compression reference = 2127 K/s".

## 10. Adversarial review (second agent) -- what changed and why

Fixed (each has a test in `tests/verification/test_scenarios.py`, "Adversarial review additions"):

1. **A1 / A2 / warm_start set the base temperature with an ABSOLUTE component**, so the builder's own documented use
   `create('a1') + create('a4', time_compression=100)` failed to compile ("set absolutely by both"). They now set
   the `thermal.t_base_k` DEFAULT; a soak or cold start replaces it. `scenario.ambient_c` also replaces it (explicit
   config wins over a scenario default). Note: a calibrated power applied at another base (a soak) gives a larger
   resonance excursion when beta > 0 (s(T) is larger hot); that is physical, the calibration is at the target's base.
2. **`scen_next_break_b` claimed linearity across knots** for a periodic `sampled` component at negative local time
   (a negative `phase_s`): it returned the distance to x = 0 instead of the next knot (error 7.8 on a test table).
3. **Clipping kinks were invisible to `scen_next_break_b`** (a ramp running into a clip limit was reported linear
   across the kink; deviation 0.49 on a [0, 1] signal). New `SI_CLIP` column (sig_i64 now (S, 5)), set at compile
   from conservative value ranges; such signals break every tick. No library scenario is flagged except the swing
   when `traffic_start_stop` and `squelch` are stacked (intended: the clip is what makes them combine).
4. **`CF_HSMOOTH` ignored the `bernoulli_td` map** (2 r (1 - r) curves up to 6x more than r: deviation 80x the
   tolerance on a sine) and **ignored that several smooth components share one signal's tolerance**. Ramps under the
   map (trapezoid, pwl, linear sampled) are curved too and were reported exactly linear; they now get an HSMOOTH
   from 4 r'^2. `map='bernoulli_td'` on a random walk (unbounded) is now an error.
5. An additive component on an element whose default is NaN (`tx.bias_v` not driven) silently produced NaN; now a
   compile error.
6. `squelch` was absolute on `data.swing_scale`, so "squelch during traffic" could not be built; now additive.
7. `reloma_plane` reset the base temperature by 105 K instantly between biases (a sawtooth), which unlocks any loop
   and corrupts the next bias's sweep; now a triangle.
8. `DisturbanceFrame` could not replay aggressor power (`from_frame` of any rendered aggressor scenario raised); it
   now records `aggressor_sites`. It also no longer mutates the caller's `signals` dict.
9. `from_config` now applies the 07 P0-10 compression rule at load (it ran only at compile) and converts
   `scenario.ambient_c` etc. through `mrm.params.units.to_si` (the single SI conversion) instead of its own +273.15.
10. Calibration: BVLS non-convergence now warns; `calibrate_scenario(fit_offset=True)` warns that the fitted offset is
    not applied; `model` may be given per site ({site: {s0_m_per_k, ...}}) because s0 is per victim ring.
11. A periodic `pwl` whose table runs past its period is rejected; `aggressor_direct` takes `victim_ring`; unknown
    `scenario.aggressor` keys raise ScenarioError instead of TypeError.

Added: `training_pattern` (07 P0-11 "training patterns", which were missing) and the A2 reading `ramp_per_period`
(04 D1's verified note favours it on arithmetic; the builder had only the three 01 D7 readings).

Open (not fixed here):

- **Two laser-drift paths.** `mrm.plant.laser` models drift itself (keys `laser.freq_drift_hz_per_s`, ...; off by
  default) and the `laser_drift` scenario adds drift through `laser.dfreq_hz` / `laser.power_delta_db`. Enabling
  both applies it twice; the integrator should pick one (or the engine should refuse both). Their random-walk
  calibrations also differ (assumed in both): scenario P(max over 24 h > limit) ~ 1e-3, plant 24 h rms = limit/2.
- **k(T) residue scaling (07 P2-5, `src_scale` in mrm.plant.thermal) is not in the calibration**: the aggressor
  response is taken at its nominal residues. If the engine scales the aggressor residues with temperature, the
  calibrated power misses the target by that factor (a few % over 100 K).
- Absolute data-path builders (`traffic_start_stop`, `idle`, `ones_density_steps`, `training_pattern`) cannot be
  stacked with each other on the same rings (one absolute definition per element); a combined sequence needs one
  state machine (`_state_pwl` / `pwl`). The error message says so.
- The A2 / A1 temperature targets assume the plotted degC is the ring (or a fixed-fraction thermometer); 07 Q5.
