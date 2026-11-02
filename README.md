# mrm-model · branch `feature/tlc-port`

> **This branch** ports the ECE496 thermal locking controller (TLC, [msilve160/ECE496_project](https://github.com/msilve160/ECE496_project))
> into mrm-model, so it can run closed loop on the 8-ring G26 plant and produce performance and locking
> metrics for the ECE496 project.
>
> **Status: Phases 0-4 framework done, Phase 5 first pass written: one command runs the faithful ECE496 TLC and six comparators through 16 experiments, a Monte Carlo campaign and four model-trust corners on G26. First results write-up in `docs/tlc/SUMMARY.md`; open items in `PLAN.md`. Decisions Q1–Q12 are settled in spec §6.**
> ▶ Results: [`docs/tlc/SUMMARY.md`](docs/tlc/SUMMARY.md) (write-up) · [`docs/tlc/results.md`](docs/tlc/results.md) (tables) · Run: `tools/tlc_run_all.sh` · How it works: [`docs/tlc/WALKTHROUGH.md`](docs/tlc/WALKTHROUGH.md)
> 📄 Spec & plan: [`docs/TLC_PORT_SPEC_AND_PLAN.md`](docs/TLC_PORT_SPEC_AND_PLAN.md) · 📋 Tracker: [`PLAN.md`](PLAN.md)

## Goal of this branch
- **Deliverable:** a framework. One command runs the TLC (and comparators) through the ECE496 scenario set on G26 and writes a metrics report.
- **Not a deliverable yet:** good TLC numbers on G26. The TLC is proven on the ECE496 4-ring plant; here it is ported faithfully, and where it falls short on G26 the report records the failure and the reason.

## ⚠️ Things that need to be addressed

### Decisions (settled 2026-10-31, spec §6; the owner can reopen any)
- [x] **Q1** Meaning of "work better than the 4-ring ECE496 baseline": G26 as the higher-fidelity testbed (side-by-side report), TLC beating ECE496 `BaselinePI` on G26, or TLC-on-G26 beating TLC-on-4-ring?
- [x] **Q2** Which comparators go in the report (`tlc_pi`, `pi_lock_c`, `kalman_lqg_c`, `mpc_c`, `dither_lockin_c`)
- [x] **Q3** Report units: pm, GHz, or both
- [x] **Q4** What to do with the rev 1 code (archive / delete / repair)
- [x] **Q5** How the ECE496 source enters this repo (copy with attribution / submodule / sibling import)
- [x] **Q6** Main now has its own README; this branch section sits on top of it. Before merging to main, move it to `docs/tlc/README.md`?
- [x] **Q7** Which knowledge tokens the TLC gets (`spec.nominal` exposes only the line grid by default)
- [x] **Q8** Pass thresholds: ECE496 R-reqs converted to pm, mrm's 5 pm budget, or both
- [x] **Q9** ECE496 scenarios with no mrm equivalent (common-mode D2, +10 K kick, PD dropout): skip, add to mrm, or emulate in the adapter
- [x] **Q10** Run durations and Monte Carlo size (the Python TLC takes ~3 ms per 100 µs step)
- [x] **Q11** Which results get committed
- [x] **Q12** Push target and reviewer (`origin` is `kennykguo/mrm-model`)

### Problems found in the existing rev 1 port (spec §1)
- [ ] B3 lock-in returns unsigned y1/y2/ζ, so the loop can't tell which side of the lock point it is on
- [ ] B1 tone plan collides: 2f of 1.5/2.0/2.5 kHz lands on other rings' tones (the source plan didn't)
- [ ] B4 UKF doesn't converge on clean self-consistent data (160 pm true → 140 pm estimate)
- [ ] Invented or drifted constants (`p_nominal=42116`, `mod_shift=50 pm`, `0.05 pm/code`, 3 ms tail vs plant 4 ms)
- [ ] The 140 tests don't touch mrm's plant or API; several are written so they can't fail
- Full evidence: [`docs/tlc/REV1_REVIEW.md`](docs/tlc/REV1_REVIEW.md)
- [ ] The rev 1 spec used a non-existent `obs["tx.monitor"]`, the wrong default rate (G26 is 100 kHz), and the wrong telemetry keys

### Found while building (Phases 1-3)
- [x] 8-ring collision-free tone plan (4.0-13.5 kHz; the search rediscovers the source's 4-ring plan exactly)
- [x] Warm start no longer reads the true detuning (initial estimate = the core's setpoint)
- [x] Invalid monitor samples reach the core as missing (NaN)
- [ ] **Owner to confirm:** G26 cannot meet the source's R13 ER floor (2.8-3.0 dB vs 3.5 dB at the inflection), so B6 has no setpoint; the G26 priors drop the floor to 0 dB (the only parameter change to the primary arm)
- [ ] ζ = z2/z1 is weak on G26 (span ≈ 2x its noise; [`docs/tlc/lockin_diag.md`](docs/tlc/lockin_diag.md)), so the UKF leans on the DC channel
- [ ] Faithful `tlc` steady state: 1.6-4.0 pm rms; removing its dither barely changes that (tracking error, not dither). `tlc_pi`: 7-10 pm rms
- [ ] A1: the TLC loses lock (~1 nm) where `pi_lock_c` / `kalman_lqg_c` / `mpc_c` hold rings 1-7 within ~15 pm; under the TEC-held package corner it holds (8.3 pm)
- [ ] Cold start: TLC ACQUIRE ends on a wrong line in 43/64 ring trials [0.54, 0.78] vs 1/64 for `wdm_supervisor`
- [ ] Open matrix rows: R11 (D3 + D2), O5 (B10 learning), the 300-trial campaign
- [ ] Upstream at `ca3f8fe`, some `tests/verification/test_ctrl_c_*` fail with or without this branch (stale `mrm.plant.optics` names)

### G26 facts the port must handle
- [ ] Heater is a 16-bit **voltage** DAC (power ∝ code²); the TLC assumes a power fraction, so the adapter converts
- [ ] Controller `step()` runs every 10 µs with ~2–3 ADC samples per call; the TLC core expects 100 µs blocks, so the adapter runs it multirate
- [ ] `spec.nominal` is knowledge-gated: TLC priors need tokens granted in the experiment YAML
- [ ] Telemetry must be numeric `(B,N)` with the standard keys; mrm's runner doesn't record controller telemetry, so a TelemetryLog is needed
- [ ] Metrics include the TLC's own dither in the error; ±5 pm or more of dither scores as excursions
- [ ] Monitor sees the NRZ-averaged drop, so the TLC's noise-based squelch detector will be blind on G26
- [ ] Cold rings start ~7.6 nm blue and the warm-up ramp crosses foreign lines, so the TLC's ACQUIRE will lock to wrong lines
- [ ] Ring 0 runs out of heater under the A1 aggressor for every controller (`docs/KNOWN_ISSUES.md` #38)
- [ ] Thermal poles 3.2 µs / 13 µs / 4 ms and crosstalk 3 % at 1.52 ms (vs ECE496's 13 / 78 µs and 20 % at 250 µs); the TLC's internal model gains must come from priors

### Housekeeping
- [ ] **Upstream main is broken from `84de49b` (optics revamp) on:** `mrm.sim.kernel` still uses ~60 names that `mrm.plant.optics` no longer defines, so `mrm run` fails. This branch merges main only up to `ca3f8fe`, the last working commit; report it upstream and merge again once main is fixed
- [x] `.venv` created (uv, Python 3.13) per the Install section below (`python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'`)
- [ ] `tests/verification` needs `gymnasium` (4 gym test files fail to collect without it); main later moves them to `archive/rl`
- [ ] First `mrm run` takes ~2 min for numba compilation (cached afterwards; later runs ~1-2 s)

## Plan at a glance
| Phase | Outcome |
|---|---|
| 0 ✅ | Decisions above, env, rev 1 archived, source brought in |
| 1 ✅ | **Walking skeleton:** `mrm run experiments/tlc/steady_tlc.yaml` works end to end with a stub TLC, and a report is generated |
| 2 ✅ | Faithful TLC core; it reproduces ECE496's own results on the ECE496 plant |
| 3 ✅ | TLC core on G26 (priors, open-loop lock-in diagnostic, warm closed loop) |
| 4 ✅ | Full scenario matrix: ECE496 D1–D6 / R-reqs mapped to mrm scenarios |
| 5 | `docs/tlc/results.md`: R-table, ECE496-plant vs G26 side by side, scorecard rows |

---

# mrm: a control-oriented model of a silicon microring-modulator bank

`mrm` simulates a bank of eight silicon microring modulators on one bus waveguide, the heaters and converters around
them, and the outside disturbances (aggressor heaters, package temperature, laser drift, traffic), tick by tick. Its
purpose is to design and score heater wavelength-lock controllers. A controller sees only what the real hardware
would show it: monitor ADC codes in, heater DAC codes out. The default device is the Lightmatter G26 transmit bank
(preset `g26`). Most of its numbers come from public papers, talks and patents or from solver estimates, and every
number in a preset records where it came from.

![Block diagram of one plant tick](docs/architecture.svg)

## Install

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

The optional extras are `extract` (the offline mode, coupler, junction and thermal solvers), `rl` (gymnasium) and
`hdl` (cocotb).

## First run

Run every Python job from the repository root through the memory guard. The first run compiles the plant kernel,
which can take a few minutes.

```sh
.venv/bin/python tools/memguard.py -- .venv/bin/mrm run experiments/examples/steady_pi_lock.yaml
```

This runs the shipped PI lock (`pi_lock_c`) on all eight rings at constant ambient temperature for 40 ms and prints
the lock metrics. [docs/EXPLAINED.md](docs/EXPLAINED.md), "Try it", lists the next runs.

## What to read, in order

Documents:

1. [docs/guide/mrm_guide.pdf](docs/guide/mrm_guide.pdf): the reading guide, written for a reader who knows textbook
   chapters 0 to 5 (waveguides, coupling, the ring resonator). It goes from your add-drop ring to the whole model.
2. [docs/EXPLAINED.md](docs/EXPLAINED.md): the physical picture, and the first runs.
3. [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md): what the model does not do yet.
4. [docs/ENGINE.md](docs/ENGINE.md), sections 1-6: how to run the engine and what happens in one tick.
5. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), section 2: where each part of the code lives.
6. [docs/CODE_STYLE.md](docs/CODE_STYLE.md): how the code is written.

Code (the same order as the package docstrings and docs/ARCHITECTURE.md):

1. `examples/controllers/template_python_controller.py` and `mrm/control/api.py`: what a controller gets and returns.
2. `mrm/plant/optics.py`, `ring_rows_nb`: the add-drop ring field calculation, including backscatter.
3. `mrm/plant/phase.py`, `ring_optics_row`: where the ring's phase, coupling and loss come from.
4. `mrm/plant/selfheat.py`, `solve_ring_temperature`, and `mrm/plant/thermal.py`: self-heating and the heat paths.
5. `mrm/plant/sensing.py`, `monitor_tick_row`: from drop power to a monitor ADC code.
6. `mrm/sim/kernel_ref.py`, `ReferenceEngine.step`: one tick in plain NumPy. `mrm/sim/kernel.py` is the fast copy of
   the same tick; skip it until you need it.
7. `mrm/sim/runner.py`, `run`: one closed-loop run of B worlds.

## Words

| word | meaning |
|---|---|
| world | one sampled set of device parameters (batch index b); B worlds run side by side |
| g26 | the default preset (`mrm/params/presets/g26.yaml`). G25 and G26 (capital G) are Lightmatter device generations |
| preset | a YAML file of device parameters; every number carries its value, provenance and source |
| tick | the plant time step, 1 µs by default; time is an integer count of ticks |
| substep | one thermal step of h ticks (h = 1, or a power of two under the automatic substep policy) |
| edge | a tick at which the controller runs (a controller edge) |
| Truth | the true plant state (detuning, temperatures, powers); never shown to the controller |
| detuning | resonance wavelength minus laser-line wavelength (positive: the resonance is red of the line) |
| relOMA | \|P_thru,1 − P_thru,0\| / P_in: the modulation amplitude relative to the ring's input power |
| oracle | the best relOMA over the detuning at that instant, among the open-loop-stable points |
| scenario | the disturbances of a run as signals over time (aggressor heaters, soak, laser drift, traffic) |
| experiment | a YAML file that names a preset, a controller, a scenario, the start state and the scoring |

## Checks

```sh
.venv/bin/python tools/memguard.py --limit-gb 5 -- .venv/bin/python -m pytest -q -m "not slow" -p no:cacheprovider
.venv/bin/python tools/memguard.py --heavy -- .venv/bin/mrm golden
```

The first is the fast test suite. The second checks that the plant still gives the committed golden traces.
[docs/RENAMES.md](docs/RENAMES.md) lists the names that changed in the readability pass.
