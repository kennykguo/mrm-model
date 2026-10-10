# Handoff: hardware bridge, golden vectors, golden traces (`hardware`)

Scope: 07 P1-21 (binary lockstep bridge, golden vectors, float/fixed equivalence; the Verilator/cocotb HDL
co-simulation is a documented stretch) and the golden traces of 07 section 4.9. Status: done, except the items in
section 6. Nothing was committed.

## 1. Files

New (owned):
- `mrm/sim/bridge.py`: frame layout generated from the ChannelSpecs, codec, transports (pipes, TCP, socketpair),
  handshake, `BridgeController` (plant side), `serve` (controller side), `spec_for`, `restrict_spec`, the C header
  generator `c_header` and the C reference controller `C_FXP_PI_SOURCE`; `python -m mrm.sim.bridge serve | c-header
  | c-example`.
- `mrm/sim/vectors.py`: `VectorRecorder`, `run_recorded`, `record_vectors` (versioned `.npz` + `$readmemh` `.hex` +
  Verilog `.layout.vh`), `load_vectors`, `replay` (Python protocol), `replay_abi` (compiled ABI, no spec: the RTL
  test-bench view), `compare_actions`, `write_hex` / `read_hex`, `observation_from_flat`.
- `mrm/control/cli.py`: `register_cli` (`mrm golden [--update --reason] [--case] [--list] [--dir]`, `mrm bridge ...`,
  `mrm vectors record | replay | hex`) and the golden-trace machinery (`GOLDEN_CASES`, run, save, load, compare,
  check, update, CHANGELOG line). Already listed in `mrm.cli.PROVIDERS`.
- `tests/golden/`: `g26__static.npz`, `g26__a1_slide10_aggressor.npz` (metrics only), `vectors/fxp_pi__g26__
  aggressor_square.npz` (golden vectors, 55 kB; only the npz is committed, the hex files are generated),
  `test_golden.py`, `README.md`.
- `tests/verification/test_bridge.py`, `test_bridge_c.py`, `test_vectors.py`, `test_vectors_equivalence.py`.
- `tests/hdl/`: `fxp_pi_update.v` (synthesizable Verilog twin of the PI update), `tb_fxp_pi_replay.v`,
  `tb_fxp_pi_update_random.v`, `test_hdl_replay.py` (Icarus only; section 5).
- this file.

Extended (new functions only, nothing existing changed): `mrm/control/fxp.py` gets the reference fixed-point PI
controller: `fx_pi_update`, `fxp_pi_init`, `fxp_pi_step`, `pi_float_bound_codes`, `class FxpPi`, and these names in
`__all__`. The block at the end of the file imports `mrm.control.api` and `mrm.core.channels` (api never imports
fxp; test_control_api's isolation test still passes: no plant/sim/validation import).

## 2. Public API (exact signatures)

`mrm.control.fxp` (additions):
```
fx_pi_update(e, integ, kp, ki, u0, frac, acc_bits, lo, hi, rnd) -> (u, integ_new)     # @njit, int64
fxp_pi_init(spec_i64, spec_f64) -> (p_i64, p_f64, s_i64, s_f64)                        # @njit, 07 P0-9 ABI
fxp_pi_step(p_i64, p_f64, s_i64, s_f64, obs_i64, act_i64) -> None                      # @njit
pi_float_bound_codes(abs_e_sum: float, abs_e: float, frac_bits: int) -> float
class FxpPi(CompiledController):
    __init__(kp=0.0193, ki=0.0148, *, gain_format="Q3.20", acc_format="Q19.20", rounding="half_up",
             setpoint_codes=None, latency_ticks=0, name="fxp_pi")
    from_params(params) (classmethod; registry path "mrm.control.fxp:FxpPi"), kp_raw, ki_raw, kp_q, ki_q,
    frac_bits, gain_format, acc_format, rounding, describe() -> dict,
    reset(spec) (CompiledController.reset after checking that the monitor codes span < 2^16: the 17-bit error)
```
FxpPi: per ring, e = y - y_set on `tx.monitor.latest` (new and valid samples only), positional PI
`integ' = clamp(sat_acc(integ + ki e), (lo-u0) 2^F, (hi-u0) 2^F)`, `u = clamp(round_F(u0 2^F + kp e + integ'), lo, hi)`;
u0 = the latched register at the first step; set-point captured from the first valid sample (or given). It drives
its output every step. It holds an operating point: no acquisition, peak map or recapture (start it at a lock).
Default gains: pi_lock's loop (2.5 kHz crossover, 13 us zero) over the measured g26 nominal lock slope
-10.6 ADC codes per heater code: a design choice for g26 nominal, not a tuned controller.

`mrm.sim.vectors`:
```
FORMAT = "mrm.vectors/1"
class VectorRecorder(controller, *, max_steps: int | None = None)         # Controller protocol; .stream(meta=None)
run_recorded(cfg_or_params, controller, scenario=None, duration_s=1e-3, *, max_steps=None, **run_kwargs) -> RecordedRun
record_vectors(run, path, *, hex: bool = True, meta=None) -> dict[str, Path]   # run: RecordedRun | VectorRecorder | VectorStream
load_vectors(path) -> VectorStream
replay(controller, vec, spec, *, check_spec=True) -> ReplayResult
replay_abi(init_fn, step_fn, vec, *, spec_i64=None, spec_f64=None) -> ReplayResult
compare_actions(vec, act, written=None) -> (n_mismatches, first | None)
clip_held_command(ctrl, spec) -> None          # the engine's clip of a CompiledController's held command (in place)
hidden_warm_start_registers(spec) -> list[str] # bound heater/bias whose '.applied' readback the spec hides
write_hex(path, words, comments=()) -> Path ; read_hex(path) -> np.ndarray ; layout_vh(vec) -> str
observation_from_flat(spec, obs_i64, tick=None) -> Observation ; stream_of(run) -> VectorStream
VectorStream(ticks, obs, act, written, reset_before, layout_rows, layout_fingerprint, spec_fingerprint,
             act_in=None, spec_i64=None, spec_f64=None, meta={}) ; .channel(name, vector="obs"), .field(...)
ReplayResult(act, written, mismatches, first) ; .ok
```

`mrm.sim.bridge`:
```
PROTOCOL_VERSION = 1 ; KIND = {HELLO 1, READY 2, OBS 3, ACT 4, BYE 5, ERROR 6} ; PROFILES = ("minimal", "full")
FrameLayout.from_spec(spec, profile="full", *, sensors=None, actuators=None) -> FrameLayout
    .encode_obs(tick, sensors, new) -> bytes ; .decode_obs(payload) -> (tick, sensors, new)
    .encode_act(tick, actions) -> bytes ; .decode_act(payload) -> (tick, write_mask, actions)
    .describe(), .fingerprint(), .sensors, .actuators, .obs_record_bytes, .act_record_bytes, payload sizes
class BridgeController(transport, *, latency_ticks=0, profile="full", sensors=None, actuators=None,
                       name="bridge", keep_times=False)        # plant side; .stats, .rt_samples, .round_trip_us, .close()
serve(controller, spec, transport, *, max_steps=None) -> dict  # controller side
spawn(cmd, *, env=None, cwd=None, stderr=None) -> StreamTransport ; stdio() -> StreamTransport
SocketTransport(sock, *, timeout_s=None)       # timeout_s: a silent peer raises BridgeError (liveness only)
connect(host, port, *, timeout_s=None) -> SocketTransport ; listen(host="127.0.0.1", port=0) -> socket
write_frame(tr, kind, payload=b"") ; read_frame(tr, *, eof_ok=False) -> (kind, payload) | None
spec_for(cfg, scenario=None, batch=1) -> ControllerSpec        # exactly as runner.run builds it
restrict_spec(spec, sensors) -> ControllerSpec
c_header(layout, *, guard="MRM_BRIDGE_H", spec_fingerprint=None) -> str ; C_FXP_PI_SOURCE: str
    # spec_fingerprint adds MRMB_SPEC_FP (mrm bridge c-header passes it); the C twin then checks the HELLO's spec
add_arguments(ap) ; run_command(args) -> int ; main(argv=None) -> int
```

`mrm.control.cli`: `register_cli(register)`; `GOLDEN_CASES: dict[str, GoldenCase]`; `GoldenCase(name, preset,
scenario, duration_s, runs, scenario_params={}, overrides=(), run_kwargs=DEFAULT_RUN_KWARGS, description="",
pending=None)`; `GoldenRun(label, controller, controller_params={})`; `run_golden_case(case, *, progress=None) ->
GoldenData`; `save_golden(path, data, *, reason)`; `load_golden(path)`; `compare_golden(ref, new, *, rtol=1e-9,
atol_scale=1e-12) -> list[str]`; `check_golden(names=None, directory=None, *, progress=None)`;
`update_golden(names, reason, directory=None, *, progress=None) -> list[str]` (CHANGELOG lines);
`changelog_line(name, data, reason)`; `golden_dir()` (`$MRM_GOLDEN_DIR` or tests/golden), `golden_path(name, dir)`.

## 3. Design

**Bridge frames.** 8-byte header `{'M','B', version, kind, u32 payload_bytes}`; OBS = `u64 tick` + B world records;
ACT = `u64 tick, u32 write_mask, u32 0` + B records. The record layout is generated from the ControllerSpec: one
field per non-bool channel on the narrowest little-endian integer type holding the channel's [min, max]; bool
channels and new-sample flags packed into `flags[N]` (per ring) and `dflags` (per world). g26 (B = 1):
- `minimal` = 07's frame: OBS record 43 bytes `{u16 adc[8] (tx.monitor.latest), u16 temp[1], u16
  heater.applied[8], u8 flags[8], u8 dflags}`, ACT record 16 bytes `{u16 dac[8]}` (+ `u16 bias[8]` when the
  controller owns tx.bias);
- `full`: every exposed sensor (OBS record 539 bytes; flags are u16, 16 bits) and every controller-bound actuator.
- custom (`sensors=`, `actuators=`): in every profile a block channel brings its count and a framed action its
  `.applied` register.
The layout fingerprint covers each field's wire type, offset, count and channel range [min, max] (review fix: the
range was left out, so a C twin built for another heater range with the same u16 wire type was accepted). The
handshake (HELLO / READY) checks protocol version, layout fingerprint, ControllerSpec fingerprint (Python side; the
C twin too when its header has MRMB_SPEC_FP) and the controller latency. Frames carry ticks only; a wall-clock
timeout exists only as an optional lost-peer guard on sockets (`timeout_s`).

**Where the bridge runs.** `BridgeController` is a Python-protocol controller, so a bridged run uses the runner's
Python path unchanged (the kernel stops at each edge). Bit-identity holds because the external side builds the same
spec (fingerprint-checked), sees exactly the framed channels (a smaller frame: `restrict_spec`, ids kept, so a
CompiledController still finds its channels by id; a Python controller reading an unframed channel gets
UndeclaredChannelError; a CompiledController gets offset -1, as with `controller_interface.expose`, so use `full`
unless it reads only framed channels), applies the runner's ActionValidator before framing, and clips a
CompiledController's held command as the kernel and the runner do (`vectors.clip_held_command`). An action written
outside the frame must equal the plant's register from the first step on (its framed `.applied`, else the channel
init; an unframed heater or bias, whose warm-start register is unknown on that side, is refused). Controllers with a
`fast_kernel` and Truth-fed controllers are refused at the handshake.

**Vectors.** Recorded at the Controller-protocol level by a wrapper; for a CompiledController the wrapper mirrors the
two runner details that matter (held command pre-filled from `.applied`; held command clipped to the channel ranges
after each step), so recording is bit-transparent (tested against the in-kernel compiled run). Stored: obs_i64,
act_i64 after the step (raw, before the plant's range check), act_i64 held on entry (compiled), the written-channel
mask, reset marks and the spec arrays, plus a sha256.

**Golden traces.** Three runs per case (open loop, `pi_lock_c`, `fxp_pi`), g26 nominal, seeds 0, B = 1, settled
warm start at the quasi-static lock codes. Integer and bool metrics exactly, floats to rtol 1e-9 (+1e-12 of the
metric's scale), NaN = NaN, text exactly. `--update` needs `--reason` and prints one CHANGELOG line per changed file
(with the 12-digit data hash); `test_every_golden_file_is_in_the_changelog` enforces the line. The compressed soak
is a pending case (`GOLDEN_CASES["g26__soak_compressed"].pending`), listed and skipped with its reason.

## 4. Measurements (shared 8-core M2, AC power, load 2.1-2.4, warm numba cache, 2026-10-10)

FxpPi on g26 nominal, static, 50 ms (5000 edges at the g26 100 kHz controller rate = 10 ticks per edge), best of 3:

| path | us per plant tick | us per controller edge | round trip (median, us) |
|---|---|---|---|
| in-kernel compiled (no bridge) | 9.9 | 99 | - |
| in-process Python path (CompiledController.step from Python) | 27.7 | 277 | - |
| bridge, C peer over pipes, minimal frame | 31.0 | 310 | 13.6 |
| bridge, C peer over pipes, full frame | 32.7 | 327 | 13.6 |
| bridge, Python peer (FxpPi) over pipes, minimal | 54.5 | 545 | 87.5 |
| bridge, Python peer (FxpPi) over pipes, full | 61.1 | 611 | 115 |
| bridge, Python peer over TCP loopback, full (test run) | - | - | 230 |

The transport round trip of the binary protocol is 11-14 us per edge (1.1-1.4 us per plant tick at the g26 rate),
against 07's 10-50 us for a text protocol. Plant-side codec per edge: encode OBS 19 us (minimal) / 37 us (full),
decode ACT 3.4 us. The bridged run is dominated by the Python path's per-edge kernel stop (27.7 us per tick in
process); a bridged C controller costs +3 to +5 us per tick over that. Golden traces: `mrm golden` 6.3 s (static
~1 s, A1 three runs of 130.7 ms ~4.3 s). Icarus replay of the golden vectors (3200 checks): < 1 s.

## 5. Tests (all run through tools/memguard.py)

| file | result |
|---|---|
| tests/verification/test_vectors.py | 16 passed (incl. a NUMBA_DISABLE_JIT=1 replay subprocess) |
| tests/verification/test_vectors_equivalence.py | 6 passed (37 of test_fxp + this file under NUMBA_DISABLE_JIT=1, closed loop excluded) |
| tests/verification/test_bridge.py | 30 passed (19 protocol tests also under NUMBA_DISABLE_JIT=1) |
| tests/verification/test_bridge_c.py | 12 passed (skipped without `cc`) |
| tests/golden/test_golden.py | 11 passed, 1 skipped (pending soak), **1 failed: test_every_golden_file_is_in_the_changelog** until the integrator adds the three CHANGELOG lines of section 7 |
| tests/hdl/test_hdl_replay.py | 5 passed (skipped without iverilog / yosys) |
| existing: test_fxp.py, test_control_api.py, test_params_cli.py | 93 passed |

What they establish:
- **Bit-identity of the bridge**: FxpPi through a thread endpoint (socketpair, both profiles), a Python process
  over pipes (both profiles) and over TCP, the C twin over pipes (both profiles), B = 2 sampled worlds, and a
  bias-owning controller: every trace (detuning at every tick, heater and monitor codes) and every metric equal to
  the in-kernel run. Loud failures: spec, layout, latency, protocol version, magic, a lost tick, a dead peer, a
  controller exception (reported through an ERROR frame), an out-of-range value, a changed unframed actuator.
- **Golden vectors**: recording is transparent; files round-trip (npz, hex, vh); replays bit-exact through the
  Python protocol, the compiled ABI, the ABI without JIT, the C twin (golden file, 400 records) and the Verilog twin
  (Icarus, 3200 heater actions); one LSB of ki is detected; the committed vectors are still produced by today's plant.
- **Fixed point against itself**: `fx_pi_update` against an exact big-integer reference on 20 000 random operand
  sets (all 5 rounding modes, saturation and clamps); the C twin against FxpPi on a 1500-step synthetic stress
  stream per rounding mode (accumulator saturation, integrator clamp and output clamps all hit); the Verilog
  `fxp_pi_update` against Python on 3000 random operand sets (floor and half_up).
- **Float against fixed** (07 "within bounds (Ch17)"): docs/research has no numeric float-vs-fixed rule (06 does not
  list one; 07 cites Ch17 only for the integrator-fraction-bits failure). The bound used is derived in
  `pi_float_bound_codes`: |u_fix - u_float| <= 1 + 2^-(F+1) (sum |e| + |e_k|) codes on the same observations; held
  at every one of 3200 updates (measured difference <= 1 code; the derived bound reaches 1.19 codes at the end of
  the run). Closed loop (same plant, same noise draws): heater codes within 1, detuning within 0.045 pm (asserted
  2 codes, 0.1 pm; one heater code is 0.053 pm at the g26 lock). The Ch17 failure mode is shown: an integrator
  rounded to whole codes stalls for |ki e| < 1/2 (ki = 0.0148, e = 20 codes) while F = 20 bits track the float sum.

## 6. Deviations, doubts, open issues

1. **HDL co-simulation not built (stretch).** cocotb and Verilator are not installed (Icarus 12, vvp and yosys are);
   nothing was installed. The Verilator `.so` with `ctrl_init` / `ctrl_tick` loaded through cffi inside the numba
   kernel (closed-loop co-sim) does not exist. Instead tests/hdl runs a plain-Icarus `$readmemh` replay of the
   golden vectors through a synthesizable Verilog `fxp_pi_update` (the per-ring sequencing is in the test bench) and
   a random-operand check, plus a yosys synthesis check (7510 generic cells, combinational). This verifies the hex
   format, the layout include and the RTL arithmetic; it is not the cocotb flow 07 names. With cocotb installed,
   the same vectors drive a cocotb bench directly (`read_hex` + the `.layout.vh` offsets).
2. **07's frame text vs the generated frame**: `i16 temp[S]` becomes u16 (12-bit unsigned codes: the generator takes
   the narrowest type holding the channel range); the minimal frame adds the latched `.applied` registers of the
   framed actions (a warm start needs them: FxpPi and the C twin take u0 from them) and splits the flags into
   per-ring `flags[N]` and per-world `dflags`; ACT carries a write mask (Python semantics: a channel left out holds).
3. **Python path only.** Bridged and recorded runs take the runner's Python path: no `tick_jitter_s` (runner
   restriction), no `fast_kernel` controllers (their per-sample I/O is not in the edge frame; refused with a clear
   error by VectorRecorder and, since the review, by `serve`, which before ran them silently without the fast
   kernel), ~28 us per tick of Python-path overhead. Long co-simulations would need an in-kernel bridge (the C ABI
   path of 07 P1-21); not built.
4. **Saturation counts** of a bridged run come from the external endpoint's ActionValidator (`serve` returns them;
   equal to the in-process counts, tested); the runner's `saturation_counts_python` of a bridged run is 0, so a
   bridged run's own diagnostics understate saturation. Nothing carries the endpoint's counts back to the plant
   side (a protocol extension, e.g. in BYE, if needed).
   A recorded or bridged CompiledController whose spec hides a bound DAC's `.applied` readback
   (`controller_interface.expose`) starts its held command from the channel init instead of the plant's warm-start
   register: a warning is raised (`vectors.hidden_warm_start_registers`), the run can differ from the in-process
   one. With the default exposure (g26) this does not happen.
5. **FxpPi is a reference, not a validated lock**: g26-nominal gains, no acquisition or recapture; under A1 it loses
   ring 0 at the hot cusp like pi_lock_c (max |detuning| 470 vs 474 pm; heater authority, ENGINE.md section 5).
6. **Golden traces and the environment.** The goldens were written with Python 3.12, numpy/numba/llvmlite of `.venv`
   on arm64 macOS (recorded in each file). A closed loop can follow another valid trajectory after a last-bit
   change (perf.md 5.4), so on another platform or after a deliberate numerical change the golden test fails by
   design; the failure message shows the differing metrics and the environment change. A change of `pi_lock_c`
   (controllers_a) also changes the goldens (its runs are part of them).
7. **Golden vectors depend on the plant too**: `test_golden_vector_file_matches_todays_plant` fails when the plant
   changes; regenerate them (tests/golden/README.md) together with the golden traces.
8. The README/CHANGELOG link is a test, not a hook: an `--update` without the CHANGELOG line fails CI.

## 7. Requests

To the integrator:
1. **Add to docs/CHANGELOG.md** (tests/golden/test_golden.py checks the hashes; it fails until then):
   ```
   - golden: tests/golden/g26__static.npz updated (data sha256 bb95d4ab4626): first golden traces (static lock and the A1 fast aggressor on g26; hardware task, 07 4.9)
   - golden: tests/golden/g26__a1_slide10_aggressor.npz updated (data sha256 2b898a0568ce): first golden traces (static lock and the A1 fast aggressor on g26; hardware task, 07 4.9)
   - golden vectors: tests/golden/vectors/fxp_pi__g26__aggressor_square.npz added (sha256 da23b4c547c2): FxpPi on g26 nominal, direct square aggressor, 400 edges (hardware task, 07 P1-21)
   ```
   (the third line since the review: the test now also enforces tests/golden/README.md's rule that the committed
   golden vectors change only with a CHANGELOG line naming their sha256)
   plus an entry for the new contracts: bridge protocol 1 (frames generated from the ChannelSpecs, profiles
   minimal / full), vectors format `mrm.vectors/1`, golden format `mrm.golden/1`, the FxpPi additions to fxp.py, and
   the HDL deviation of section 6.1.
2. **Export `mrm.sim.record_vectors`** (07 P1-21 names it; `mrm/sim/__init__.py` is not mine). A lazy export keeps
   `import mrm.sim` light:
   ```python
   def __getattr__(name):
       if name == "record_vectors":
           from mrm.sim.vectors import record_vectors
           return record_vectors
       raise AttributeError(f"module 'mrm.sim' has no attribute {name!r}")
   ```
3. **Compressed-soak golden** (07 4.9): fill `GOLDEN_CASES["g26__soak_compressed"]` in mrm/control/cli.py (scenario
   parameters, duration, compression; remove `pending`) once the experiments task lands and the 07 P0-10
   convergence test passes, then `mrm golden --update --case g26__soak_compressed --reason "..."` and the line.
4. Optional: register FxpPi by name (`fxp_pi`) in `mrm/control/baselines/__init__.py` and add it to the scorecard
   CASES; today it resolves as `mrm.control.fxp:FxpPi`.
5. Docs: ENGINE.md section 10 lists "the binary bridge" as not built (now built); ARCHITECTURE section 2 can name
   tests/hdl (Icarus replay) and `mrm/control/cli.py`.
6. Optional (07 P2-11, 05 section 2): an `hdl` extra (`cocotb`) in pyproject.toml, and the pinned lock file, which
   would make the goldens portable between machines.

## 8. Adversarial review (2026-10-10)

Reviewed against 07 P1-21, 4.1 and 4.9; all fixes are in the owned files (no engine, runner or kernel file touched;
the runner diff in the tree belongs to the experiments task). Findings, most severe first:

1. **Fixed: an action written outside the frame was dropped silently on its first value.** `serve` stored the first
   value of an unframed actuator and only refused later changes, so a controller that wrote, e.g.,
   `monitor.pga_select = 1` from its first step on drove an in-process plant at gain 1 and a bridged plant at its
   init 0, with no error. Now the first value must equal the plant's register (framed `.applied`, else the channel
   init; an unframed heater or bias is refused because its warm-start register is unknown on that side).
   Test: test_unframed_actuator_must_equal_the_plant_register, test_unframed_warm_start_register_is_refused.
2. **Fixed: a CompiledController with out-of-range commands was not bit-identical over the bridge.** The kernel and
   the runner clip the held command in place; `serve` did not, so the controller's next step built on the unclipped
   value (shown: with the clip removed the new test fails on heater_code). `serve` now applies
   `vectors.clip_held_command` (shared with VectorRecorder and replay). Test: test_out_of_range_compiled_controller_
   is_bit_identical (in-kernel = Python path = bridge, every trace and metric; endpoint saturation counts = the
   kernel's = the Python path's); test_recording_is_transparent_when_commands_leave_the_range covers the recorder's
   clip, which FxpPi (always in range) never exercised.
3. **Fixed: `serve` ran a controller with a `fast_kernel` silently without it** (the handoff said such controllers
   were refused; only VectorRecorder refused them). Fast-kernel and Truth-fed controllers are now refused at the
   handshake. Test: test_fast_kernel_and_truth_fed_controllers_are_refused.
4. **Fixed: the golden comparison accepted infinities.** The tolerance rtol * max(|a|, |b|) is infinite when one side
   is infinite, so finite -> inf and -inf -> +inf passed (the golden files hold +-inf histogram edges). Non-finite
   values now match only exactly. Test: test_non_finite_values_compare_exactly.
5. **Fixed: the layout fingerprint left out the channel ranges.** Two configs with the same wire types but another
   heater range (code_max 60000 vs 65535) had the same fingerprint, so a C twin built for one (it clamps to
   MRMB_ACT_TX_HEATER_MAX) was accepted by the other. The ranges are now in the fingerprint; `c_header` can also
   embed the ControllerSpec fingerprint (`mrm bridge c-header` does) and the C twin then checks it. Tests:
   test_layout_fingerprint_covers_the_channel_ranges, test_c_refuses_another_spec.
6. **Fixed: a custom frame without the latched registers broke the warm start silently** (FxpPi's u0 became the
   channel init 0 instead of the plant's code). A framed action now brings its `.applied` register in every
   profile. Test: test_custom_frame_is_bit_identical (two-sensor frame = in-kernel run).
7. **Fixed: `mrm golden --update` could not repair a changed case definition** whose metrics were unchanged (it
   skipped the file as "unchanged" while test_golden_file_content kept failing). It now rewrites the metadata and
   asks for no CHANGELOG line (same data hash). Test: test_update_rewrites_a_changed_definition_without_a_changelog_
   line.
8. **Fixed: the committed golden vectors' CHANGELOG rule was not enforced** (README only); the changelog test now
   checks their sha256 too (third line in section 7).
9. Fixed (minor): a corrupted length word could allocate up to 1 GiB (`_MAX_PAYLOAD` now 16 MiB, checked against
   the layout); a silent TCP peer blocked forever (`SocketTransport(timeout_s=)` raises BridgeError; test
   test_silent_socket_peer_times_out); FxpPi.reset refuses a monitor wider than its 17-bit error register (an
   18-bit ADC; test); the Icarus replay took the gain width from the default FxpPi instead of the vectors' meta;
   test helpers now compare every metric, nested dicts included, instead of a few top-level ones.
10. Not changed, documented: a CompiledController on the external side sees offset -1 for an unframed channel and
    may behave differently than in-process with the full spec (bridge.py docstring); a recorded or bridged
    CompiledController whose spec hides a bound DAC's `.applied` starts from the channel init (warning).
    Observation for the runner owner (not in this task's files): `_run_python` captures `held = ctrl.act` once; a
    scenario `fault.controller_reset` calls `ctrl.reset`, which allocates a new `ctrl.act`, so after such a reset the
    runner's clip no longer reaches the controller's held command on the Python path (the kernel path restores its
    own arrays). It matters only for a CompiledController with out-of-range commands after a mid-run reset.

Checked and found sound: the HELLO / READY byte offsets in Python and C; the C rounding modes against
fx_round_shift; fx_pi_update's exactness budget; the float-vs-fixed bound derivation (premise added to its
docstring: the integrator clamp range inside the accumulator); the gains (0.204 / 10.6 = 0.0193, x 10/13 = 0.0148);
the A1 golden phase (P/2: cool minimum at t = 0); spec_for against runner.run; the absence of cocotb and Verilator
(`import cocotb` fails, no `verilator` on PATH), so the HDL stretch is correctly not built.
Timing re-check (20 ms static, best of 2; the experiments task running): at load 3.7, in-kernel 11.9, Python path
29.1, C peer over pipes 34.4 (minimal) / 33.9 (full), Python peer (minimal) 80.7 us per tick; median round trip 13.7
us (C) and 102 us (Python peer): consistent with section 4 (C peer +5 us per tick over the Python path). At load 9
the same runs took 22.3 / 67.1 / 91.9 / 100.3 / 180.8 us per tick (round trip 19 and 229 us): every round trip then
pays two process switches on a busy machine.

Test status after the review (memguard, warm cache): owned files 80 passed, 1 skipped (pending soak), 1 failed (the
CHANGELOG test, until section 7's three lines are added). Fast suite (`-m "not slow"`, whole tree incl. the
experiments task's files): 1622 passed, 74 skipped, 3 xfailed, 4 failed = that CHANGELOG test + the known VAL-8a/b/c
(test_thermal_replication.py), in 353 s at load 8-10. test_kernel_parity.py is in it and passes. No engine file was
touched, so tools/perf_scenarios.py was not rerun; the golden traces passing against today's engine is the stronger
check here. Numba caches cleared after the fxp.py edit: `tools/clear_numba_cache.py fxp` and the cached functions
of tests/verification/test_vectors_equivalence.py (no njit source changed; the edit was a new Python method).

## 9. Integration (2026-10-10, integrator; docs/handoff/integrator.md section 8)

Requests of section 7, resolved:

1. CHANGELOG: the three lines of section 7.1 and the contract entry (bridge protocol 1, mrm.vectors/1, mrm.golden/1,
   the FxpPi additions, the HDL deviation) are in docs/CHANGELOG.md (2026-10-10 integration entry);
   test_every_golden_file_is_in_the_changelog passes.
2. `mrm.sim.record_vectors`: lazy module `__getattr__` in mrm/sim/__init__.py (the code of 7.2).
3. Compressed-soak golden: `GOLDEN_CASES["g26__soak_compressed"]` is live. It is the c = 100 case of the A4
   convergence experiment (experiments/acceptance/a4_soak.yaml, certificate
   runs/acceptance_20261010/20261010T115721Z_d1511028, validated up to 100): g26 with the overrides
   `controller.name=pi_lock_c`, `scenario.name=soak`, `scenario.time_compression=100.0`, `sim.substep_policy=auto`,
   one 1218 s cycle at c = 100 (12.18 s), settled warm start, `score_from_s` 0.005, **pi_lock_c only** (no
   convergence experiment covers open_loop or fxp_pi, and 07 P0-10 allows no compressed result without one). Its
   plant and scenario fingerprints equal the certificate's (8e979786..., 79d9c265...), and all 47 metric arrays it
   shares with the certified run are bit-identical (max |detuning| 126.49580490119054 pm, heater range 26 022 codes).
   New: `GoldenCase.scenario` may be `CONFIG_SCENARIO` ('config'), the scenario section of the case's config with its
   signal defaults (a registry name gets none: g26's bias is then not driven and sits at 1.0 V instead of 1.05 V;
   the static and A1 cases keep their registry names, so their files are unchanged). The soak case's test is marked
   slow (95 s idle, ~3.5 min under load). CHANGELOG line: data sha256 b278f2666f89.
4. FxpPi is registered as `fxp_pi` (by its "module:Class" path in mrm/control/baselines/__init__.py). It is not in
   the scorecard CASES (that needs a scorecard re-run and a new committed scorecard: open for the controllers owners).
   The golden cases keep "mrm.control.fxp:FxpPi" (their definitions are unchanged).
5. Docs: ENGINE.md section 12 (bridge, vectors, golden traces) and section 10 (not built: the Verilator / cocotb
   co-simulation, an in-kernel bridge); ARCHITECTURE section 2 names mrm/control/cli.py, the FxpPi additions and
   tests/hdl.
6. pyproject.toml: optional extra `hdl = ["cocotb"]`. The pinned lock file (07 P2-11) is not written.

Also: mrm/control/fxp.py module docstring (fx_matvec is not the P1-22 MLP helper; rl request). Still open: section 6
items 1, 3, 4 (saturation counts of a bridged run) and 6 (goldens are platform specific). Resolved: the runner
observation of section 8 item 10 (after a scenario `fault.controller_reset` the Python path now clips the
CompiledController's new `act` array; tests/verification/test_runner_fault_reset.py failed before the fix: heater
65535 on the Python path against 44535 in the kernel).
