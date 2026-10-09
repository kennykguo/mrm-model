# Handoff: core timing, channels, counter RNG; control API, registry, fixed point (`core_control_api`)

Status: done. Files: `mrm/core/{timing,channels,rng}.py`, `mrm/control/{api,registry,fxp,lfsr}.py`, tests in
`tests/verification/test_{core_timing,core_channels,core_rng,control_api,fxp}.py` (208 tests after the review, about
12 s with a warm numba cache; no file above 8 s cold. All pass with `NUMBA_DISABLE_JIT=1` too: 202 run in that mode,
because the brute-force LFSR widths depend on the mode).

## 1. Decisions and deviations from the contract (please review)

1. **RNG keying (07 P1-13).** 07 asks for a key of (seed, stream_id, b, ring, tick). It also asks that element b of
   a B = 64 run equal the B = 1 run with seed b. Both hold only if b enters through the seed. The convention is
   therefore `seed_b = base_seed + b` (`rng.batch_seeds`) and `key = stream_key(seed_b, stream_id)`. The Philox
   counter is `(tick_lo, tick_hi, ring, slot << 12 | inner)`. Consequence: element 1 of the run with base 0 and
   element 0 of the run with base 1 share a stream. This gives common random numbers by design; do not pool two
   Monte Carlo sets whose base seeds are closer than B.
2. **Philox4x32-10.** It matches the three Random123 known-answer vectors, in numba and in pure NumPy. Its 32 x 32
   bit products never overflow a uint64, so the bits are identical under `NUMBA_DISABLE_JIT=1`. The draw functions
   have **explicit numba signatures** (key uint64, ring/tick/slot int64). Without them, a key returned to Python
   (an int above 2**63) selects a lazily compiled int64 specialization and overflows.
3. **Stream enum.** It has the 07 ids plus `DATAPATH`, `PATTERN` and `DAC`, which were added for the bit-statistics
   and error-count draws, the ones-density noise and the bias-DAC noise. Parametric names map to
   `category | crc24(name)`: `adc_<channel>`, `scenario_<name>`, `dac_<channel>`.
4. **Channels added** beyond the task list: `tx.heater.applied` and `tx.bias.applied`, (B, N). These are the
   latched DAC registers, which 07 P0-6 calls "applied codes"; P1-22 also puts applied codes in the observation.
   The flags are named exactly `monitor_valid` and `adc_overrange`, both (B, N) bool. `rx.error_count_valid` is a
   separate bool channel.
5. **Channel dtypes come from the converter width.** Codes of up to 16 bits are uint16 (`tx.heater`, `tx.bias`,
   `datapath.bitstat_threshold`). ADC-coded sensors are int32 (07 P0-7), or int64 above 31 bits. The schema allows
   DAC and ADC widths up to 32 bits, and a fixed uint16 would not hold them.
6. **`samples_per_tick` (07 P1-1)** is `ControllerSpec.samples_per_step`, M = ceil(controller period / ADC period).
   An alias property keeps the 07 name.
7. **`ring.monitor.adc_rate_hz` means the rate PER RING.** With `adc_mux.channels_per_adc > 1`, the mux only sets
   the sample phases.
8. **CompositeController (07 P2-7).**
   - Its `latency_ticks` is the largest child latency. A shorter child pipeline is therefore modelled
     conservatively.
   - `round_robin` gives child edge j to ring j % N only. The child sees `obs.active` and the data since its own
     last edge, and only that ring's (B, N) commands are applied.
   - When two children write the same channel in one step, the later child in the list wins.
   - `tick_jitter_s` is **not implemented** (it is optional in 07).
9. **Default owners.** `tx.heater`, `datapath.bitstat_threshold` and `monitor.pga_select` belong to the controller.
   `tx.bias`, `tx.assignment` and `laser.tune` belong to the scenario.
10. **`ChannelHardware` defaults are fallbacks** for tests. Each one carries its provenance in a comment, and the
    presets must own the real values.

## 2. Physics and contract doubts

- **Monitor ADC width against its rate (worth a hypothesis entry).** PAT-2's ADC is dual-slope, with a 2 us window
  of 400 x 5 ns clocks. One conversion therefore resolves on the order of 400 counts, about 9 bits, unless the
  de-integration runs much longer than the window. The 16-bit width (`REF.monitorAdcBits`) is inferred from the
  RX polarization-monitor codes (01 D17), not from the MRM monitor. 16 bits at 500 kHz from this converter does not
  add up. Suggestion: carry `ring.monitor.adc_bits` as a hypothesis, {16 with summed conversions at a lower rate, or
  about 10 at one conversion per window}. Controller noise and lock-in rates depend on this choice.
- `ring.monitor.adc_latency_s = 2e-6` (one window) and `ring.bias_dac.bits = 12` are assumed fallbacks.

## 3. Requests by consumer

### params / realize (schema owner)
- Build the spec with
  `ControllerSpec.from_config(cfg, n_batch=B, nominal=view, scenario_drives=..., scenario_offsets=...)`.
  `from_config` reads `sim.dt_plant_s`, `array.n_rings`, `cfg.n_lines`,
  `controller_interface.{update_rate_hz, bind.<ch>, knowledge, schedule, on_out_of_range}`, `array.bus_order`,
  `channel_plan.ring_to_line`, `channel_plan.lines[].freq_thz`, and the keys of `channels.HARDWARE_KEYS` through
  `cfg.si`/`cfg.get`. Alternatively, build `ChannelHardware.from_config(cfg)` and call `ControllerSpec.build(...)`.
- **Missing schema keys** (requested):
  - `controller_interface.bind.tx.assignment`, `controller_interface.bind.datapath.bitstat_threshold` and
    `controller_interface.bind.monitor.pga_select`, each an enum `controller | scenario`;
  - `laser.tune.present`, `laser.tune.bits` and `laser.tune.lsb_hz` (my proposal; if you rename them, change
    `HARDWARE_KEYS` or tell me);
  - the fields `temp_sensors[].adc_bits` and `temp_sensors[].code_offset_c`;
  - optionally `controller_interface.expose`, a list_str of the sensors the controller may read.
- `nominal`: pass an object with `fingerprint()` or `to_dict()`. `ControllerSpec.fingerprint()` hashes it, which is
  what 07 P1-1 `test_spec_independent_of_seed_and_uncertainty` compares.
- `timing.check_tick_table(entries, dt)` raises one error that names every bad key. You can reuse it at load time.
- realize calls `rng.fill_uniform(..., tick=_entry_id(name), ...)`. The tick must be >= 0, because a negative int64
  has no uint64 value under `NUMBA_DISABLE_JIT=1`.

### plant io (SensorBank / ActuatorBank) and `sim/kernel.py`
- **Event order:** the diagram in the `mrm/core/timing.py` docstring. A command computed at edge k latches at
  k + `latency_ticks` and acts on [k_act, k_act + 1), with k_act = latch + DAC latency. A sample with sample tick s
  covers (s - W, s]. `timing.first_response_tick` gives the expected first observation for
  `test_loop_delay_impulse`.
- **FIFOs:** `fifo_alloc/push/pop_ready/pop_all_ready/peek/count/clear` are numba ring buffers with one queue per
  element. `fifo_capacity(latency, period) = latency // period + 1` is exact if you push before you pop at each
  tick.
- **Observation contents:**
  - Give new-sample flags for EVERY sensor channel. `Observation.new` is strict, and CompositeController ORs the
    flags.
  - `tx.monitor.samples[..., :count]` must be valid and in sample-tick order, with zeros after the count.
  - On overflow (more than M samples), keep the newest M (suggested) and set `adc_overrange` or another flag.
- **Flat ABI:** `spec.layout` (AbiLayout).
  - `obs_i64 = [tick, active_ring, 0, 0, sensor values..., new flags...]`; `act_i64` holds the controller-bound
    actuators. Bools are 0/1.
  - Call the compiled `step` per batch row; `api.call_step_batch` shows the pattern.
  - At reset, pre-fill `act_i64` from the latched registers (`.applied`), or else from `ChannelSpec.init`. Then keep
    it between steps, so that "not written" means "hold". `CompiledController` does exactly this on the Python path.
- **fast_kernel:**
  - `fobs = [tick, N, new[N], code[N], sample_tick[N]]` and `fact = [heater_offset[N], bias_offset[N]]`, in codes
    (`api.fo_*`, `api.fa_*`, `fast_obs_size`, `fast_act_size`).
  - It runs once per released monitor-sample block, before `step` when both fall on the same tick.
  - The offsets are added to the latched command before the DAC latency FIFO and clipped to the channel range.
- **Python path:**
  - Wrap the sensors in `Observation(tick, dt, sensors, new, active)`, which is read-only and strict.
  - Validate every step with `ActionValidator(spec).validate(actions)` and report `saturation_counts()`.
  - Check telemetry with `validate_telemetry`.
  - For a `CompositeController`, set the call period to `ctrl.base_period_ticks(dt)`.
  - `AbiLayout.pack_obs/unpack_obs/pack_act/unpack_act` convert between the dict and flat forms.
- **RNG:** seeds `rng.batch_seeds(base, B)`; key `stream_key(seed_b, stream_id(name))`. For several draws per
  (ring, tick), use a separate slot for each, e.g. the sample index within the tick.

### baselines (`mrm/control/baselines/`)
- Register in `mrm/control/baselines/__init__.py`. `registry.resolve` imports that package on its first miss.
- Use `BaseController` with a `Params` dataclass, the `PerRing(factory)` / `PerDie(factory)` adapters, and
  `CompiledController(name, init, step, params_i64=..., params_f64=...)`.
- In compiled code, find channels with `obs_offset(spec_i64, int(ObsId.X))` / `act_offset(...)` and read parameters
  with `user_i64` / `user_f64`. `fxp` gives fixed-point arithmetic and `lfsr` gives dither signs.

### validation (oracle)
- Give the oracle class `truth_fed = True`. `registry.register` and `registry.resolve` then refuse it (07 P1-1).

### metrics
- The standard telemetry keys are `api.TELEMETRY_KEYS`: `state`, `locked`, `detuning_est_pm`, `integrator` and
  `integrator_saturated`, each (B, N).

## 4. Adversarial review (second pass): fixes and new contract points

Bugs found and fixed in these files (tests added for each):

1. **Compiled actions wrapped before validation.** `CompiledController.step` cast `act_i64` to the channel dtype,
   so a command of -5 reached the plant as 65531 (near full heater power) and 70000 as 4464, with no clip and no
   saturation count. It now returns int64; `ActionValidator` clips, counts and casts. `AbiLayout.unpack_act` has
   `cast=False` for unvalidated output. **Kernel owner:** clip `act_i64` to each channel's [min, max] before the DAC
   FIFO; never cast it to uint16 first.
2. **ABI ids shifted under `expose`.** Temperature ids were assigned in order of the EXPOSED channels, so hiding
   `temp.bank` gave `temp.die` the id `TEMP_ID_BASE + 0`: a compiled controller read the wrong sensor. Ids now come
   from the full channel set (`channels.assign_channel_ids`; `make_abi_layout(..., obs_ids=, act_ids=)`).
3. **Uniform could return exactly 1.0** (probability 2**-53: 2**53 - 0.5 rounds to 2**53), which divides by zero
   in PTRS/BTRS. Clamped to the largest double below 1; every other value is unchanged.
4. **BINV lost precision for small p:** `log(1 - p)` is ~11 % off at p = 1e-16. Now `log1p(-p)`.
5. **`from_config` withheld the grid** when the preset uses `channel_plan.center_thz/spacing_ghz` (no `lines`
   table): `line_grid_hz` was None even with `line_grid` in knowledge, and `ring_to_line` was None instead of the
   schema's identity default. Now `api.nominal_line_grid_hz` follows realize's rule, checked against
   `realize.line_frequencies_hz` on `tests/data/mini.yaml`.
6. **Composite silently dropped a child's `fast_kernel`** (a dither would vanish). Now a TypeError.
7. **Registry hid broken baselines:** any ImportError inside `mrm.control.baselines` became "unknown controller".
   Only a missing package is silent now. `create` also refuses a Truth-fed INSTANCE returned by a factory.
8. Smaller: `expose` of a sample block now also exposes its count channel; the integration window is tick-checked
   only for `afe_type: integrate_dump`; `seconds_to_ticks` rejects counts beyond 2**62; signed code dtypes
   (`laser.tune`, 32 bit) are int32 not int64; `stream_id` suffixes are case-insensitive like the prefix; Poisson
   with lam = inf saturates at 2**62 instead of an undefined float->int conversion; `Lfsr.bits` checks k <= 62.
9. A vacuous test (`test_latency_in_ticks_independent_of_controller_rate` called `first_response_tick` with a
   controller period of 1) now varies the controller period.

Added (contract items that were skipped):
- **07 P2-7 tick jitter.** `ControllerSpec.tick_jitter_ticks` (from `controller_interface.tick_jitter_s`, whole
  ticks, < period), `timing.edge_jitter(key, edge_index, J)` (counter RNG, stream `Stream.SCHEDULE`), M covers
  period + J, and `CompositeController` steps its children on NOMINAL edges when it is called late. **Runner:** run
  edge j at `phase + j*period + edge_jitter(stream_key(seed_b, Stream.SCHEDULE), j, J)`.
- **`fifo_capacity(latency, push_period, pop_period=1)`.** The old formula assumed a pop at every tick. A SensorBank
  that drains its latency FIFO only at controller edges must pass the controller period (brute-force tight test).
- `spec_arrays` header: `SPEC_I_BIAS_MAX`, `SPEC_I_HEATER_LATENCY_TICKS`, `SPEC_I_BIAS_LATENCY_TICKS`,
  `SPEC_I_ADC_WINDOW_TICKS`, `SPEC_I_TICK_JITTER_TICKS` (indices 27..31, append-only).

Contract clarifications for the plant / runner:
- `latest` sensors HOLD their last delivered value between deliveries (the new flag marks a delivery); `sum`
  sensors report only what happened since the last step. `rx.error_count` is held; `rx.error_count_valid` marks a
  delivery. The composite's accumulation relies on this.
- `Observation` arrays are read-only VIEWS valid during `step` only; a controller that keeps data must copy it.

Physics doubt, sharpened: the fallback `ring.monitor.adc_rate_hz` is now 250 kHz, not 500 kHz. A dual-slope
conversion is the 2 us integration plus up to 2 us of de-integration (D <= N = 400 clocks, PAT-2), so one converter
cannot exceed 250 kHz; `tests/data/mini.yaml` already uses 250 kHz. The 16-bit width is still not supported by
PAT-2 (about 400 counts per conversion); keep it a hypothesis in the presets.

Still open: the schema keys requested in section 3 (`controller_interface.bind.tx.assignment` matters most: without
it the WDM supervisor baseline cannot own `tx.assignment`), and `controller_interface.tick_jitter_s` is read by
`from_config` through the existing schema key.
