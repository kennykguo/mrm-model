# TLC Port: Spec and Plan (rev 3)

**Target:** `mrm-model` (G26 8-ring plant) · **Source:** `ECE496_project/thermal_control/` (sibling checkout `../ECE496_project`, commit `f98ce84`)
**Branch:** `feature/tlc-port` · **Status:** SPEC REVIEW. No implementation starts until the questions in §6 are answered.

---

## 0. Objective and scope

**Objective.** Build the framework to run the ECE496 thermal locking controller (TLC) closed loop against
mrm-model's G26 plant and collect performance and locking metrics for the ECE496 project.

| In scope (this branch) | Out of scope (later) |
|---|---|
| Faithful port of the TLC (it is proven on the ECE496 plant) | Tuning the TLC to pass on G26 |
| Adapter between the TLC and mrm's controller API | Fixed-point / RTL / `CompiledController` |
| `mrm run`-able experiment files for the ECE496 scenario set | Real-time compute |
| Telemetry capture plus a metrics report in ECE496 terms (R-requirements) and mrm terms | Changing mrm's plant |
| Side-by-side runs: the same TLC on the ECE496 4-ring plant and on G26 | |

**Definition of done:** one command runs the full scenario matrix (§3) for `tlc` and its comparators and
writes a report. Every cell holds a number, a documented failure, or "not mapped (reason)". Poor TLC
numbers are acceptable; missing or unexplained numbers are not.

---

## 1. Why the earlier spec and status were withdrawn (summary)

The full evidence, with file:line citations, is in [`docs/tlc/REV1_REVIEW.md`](tlc/REV1_REVIEW.md).

- **The rev 1 port isn't usable.**
  - B3 lock-in drops the sign (`b3_lockin.py:247`).
  - B1's tone plan collides with itself: 2f lands on other rings' fundamentals.
  - B4's UKF doesn't converge even on self-consistent data.
  - Constants are invented or have drifted.
  - The 140 tests never touch mrm's plant or API, and several are written so they can't fail.
- **Rev 1 misread the API.**
  - There is no `obs["tx.monitor"]`; the inputs are `tx.monitor.samples` and `tx.monitor.count`.
  - G26 runs at 100 kHz (`period_ticks=10`), not 10 kHz.
  - `spec.nominal` exposes only `line_grid` and `ring_to_line` unless the experiment grants more.
  - The standard telemetry keys are `state, locked, detuning_est_pm, integrator, integrator_saturated`.
- **Rev 1 misread the plant.**
  - The heater is a voltage DAC, so power ∝ code².
  - The monitor sees the NRZ-averaged drop, so there is no data-pattern noise to detect squelch from.
  - The metrics score the true detuning including the controller's own dither.
  - Ring 0 runs out of heater under A1 for every controller.
  - Cold rings sit about 7.6 nm blue, and the warm-up ramp crosses foreign lines.

### Source plant vs G26 (what changes for the TLC)
| | ECE496 4-ring plant | G26 (mrm) |
|---|---|---|
| Units | GHz; u = heater power fraction | pm; 16-bit voltage-DAC codes (power ∝ code²) |
| Lineshape | HWHM ~20 GHz (~115 pm) | FWHM ~338 pm |
| Heater poles | 13 µs + 78 µs | 3.2 µs (41 %) + 13 µs (54 %) + 4 ms tail (5 %) |
| Crosstalk | 20 / 6 / 2 % at 250 µs | 3 % nearest at 1.52 ms, 1.5 % far, τ ∝ d² |
| Monitor | 12-bit, 1 MS/s, data-pattern noise | 16-bit, 250 kS/s, integrate-and-dump, NRZ-averaged (no pattern noise) |
| Fast aggressor | D2, ~200 pm/ms, 3 K swing | A1, ~160–180 pm/ms, 88.5 K swing |
| Rings | 4 | 8 |
| Controller call rate | TLC called every 100 µs with 100 PD samples, returns 10 heater samples | `step()` every 10 µs, ~2–3 ADC samples per call |

1 GHz ≈ 5.72 pm at 1310 nm.

---

## 2. Spec

### 2.1 Architecture: port the controller unchanged, adapt the interfaces
```
  mrm runner (1 µs plant, step() every 10 µs)
        │ obs (codes)                                     ▲ {"tx.heater": codes}
        ▼                                                 │
  ┌────────────── mrm/control/tlc/controller.py  @register("tlc") ──────────────┐
  │ Adapter (mrm-specific)                                                      │
  │  • ingest: tx.monitor.samples/count → normalised floats → 100 µs block     │
  │  • every 10th edge: core.step(block) → 10 heater samples (u, power frac)   │
  │  • every edge: emit next sample: u → code = round(code_max·√u)             │
  │  • priors: spec.nominal tokens → PlantParams-equivalent (never mrm.plant)   │
  │  • telemetry: (B,N) numeric, std keys + TLC extras; TelemetryLog sink       │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Core = ECE496 tlc/ + optics.py, unchanged except: N parameterised,          │
  │ params injected, PD sample rate & block length from config                  │
  └─────────────────────────────────────────────────────────────────────────────┘
```
- **The core is the ECE496 algorithm**, kept in the source's units (GHz, power fraction). It stays diffable
  against `../ECE496_project`.
- **The adapter is the only mrm-specific code.** It owns units, the DAC law, sample buffering, the multirate
  schedule, priors and telemetry.
- **Multirate.** mrm calls `step()` every 10 µs; the core runs on every 10th call. The other calls replay
  the 10 pre-planned heater samples, which carry the dither. This is exactly the source's 100 kS/s output
  stream.
- **Batch.** One core instance per world (`B`), run in a loop.

### 2.2 Interface contract
| Item | Value |
|---|---|
| Registration | `tlc` (full TLC) and `tlc_pi` (source `BaselinePI`: same B1/B2/B3/B8, PI only) |
| Reads | `tx.monitor.samples`, `tx.monitor.count`, `tx.heater.applied`, `monitor_valid`, `adc_overrange` |
| Writes | `{"tx.heater": int (B,N)}` within `spec.heater_dac` |
| Knowledge tokens | Q7 (§6) |
| Telemetry, standard | `state` (Mode int), `locked`, `detuning_est_pm`, `integrator`, `integrator_saturated` |
| Telemetry, extras | `rate_est_pm_per_ms`, `sigma_pm`, `dither_amp_pm`, `eff_learned`, `squelch`, `fault`, `core_step_us` |
| Params | `BaseController.Params` mirroring source `ControllerParams` + `core_every_n_edges=10`, `start_mode` (`warm`/`cold`), `force_fault_at_s` |
| Start | `warm`: mrm's settled oracle code → core `warm_start` (code→u, δ from priors). `cold`: core `cold_start` (source ACQUIRE, unchanged) |

### 2.3 Allowed deviations from the source (complete list; anything else is a bug)
1. `N` becomes a constructor argument instead of the module-level `N_RINGS`.
2. Params are injected; the source defaults stay as the defaults.
3. PD rate becomes 250 kS/s with 25 samples per core step; B2 decimation and FIR cutoff are rescaled so the
   lock-in sees the same band.
4. Plant knowledge (lineshape, heater gain, thermal τ, unheated offset) comes from priors, not
   `PlantParams()` defaults.

Squelch detection, cold-start ACQUIRE and dither amplitude stay as in the source. Each is expected to
misbehave on G26 (§1). The report records how, and redesigning them is follow-up work.

### 2.4 Metrics and report
- **mrm metrics** (free from `mrm run`): `max_abs_lock_error_pm`, `rms_lock_error_pm`, `excursions`,
  `locked_final`, `wrong_line_*`, `time_to_lock_s`, `oma_penalty_db`, `heater_*`, `fsr_hop_events`, BER.
- **ECE496 metrics** (computed from recorded traces, per ring, in pm and GHz): p99.9 / peak / rms of the
  true error, recovery time after a disturbance, heater mW, max slew in %FS/ms, command range, lock time,
  wrong-channel locks, plus learned efficiency and coupling for B10.
- **TLC internals** come from the TelemetryLog. mrm's runner records only Truth fields, not controller
  telemetry.
- **Report:** `tools/tlc_report.py runs/<ids>` → `docs/tlc/results.md` + JSON. It holds:
  - the R-table (TLC vs comparators, G26);
  - a side-by-side table (TLC on the ECE496 plant vs TLC on G26);
  - the mrm scorecard rows.

---

## 3. Scenario matrix (ECE496 → mrm)
| ECE496 scenario | Req | mrm scenario | Mapping |
|---|---|---|---|
| steady | R1 | `static`, warm (as in `steady_pi_lock.yaml`) | ✅ direct |
| D1 slow ramp, 10 K/s | R2 | `xpu_power_step` (10–120 K/s) | ✅ close |
| D2 common aggressor | R3, R4 | none common-mode; `aggressor_direct` with a far/central site? | ⚠️ Q9 |
| D2 local | R3, R6 | `a1` (ring 0 victim), `a2` | ✅ (ring 0 heater-limited, KNOWN_ISSUES #38) |
| D3 + D2, 1.5× | R11 | `a1` + uncertainty corners | ⚠️ approximate |
| D4 laser offset / drift | R5 | `laser_drift`, `laser_sweep_emulation` | ✅ |
| D5 laser power ±2.5 % / ±3 dB | R7 | `fibre_loss_step(delta_db=…)` (power into the rings) | ✅ close |
| D6 squelch, all / one ring | R12 | `squelch` / `b1` (`rings=` param) | ✅ (detector expected blind, §2.3) |
| cold start | R8 | `cold_start` / `b5` | ✅ (wrong-line expected, §1) |
| random cold-start campaign | R9 | `mc` of `b5` worlds | ✅ |
| restart one ring | R8 | `line_enable` off→on | ⚠️ restarts the laser, not the controller |
| +10 K kick, loss of lock | R10 | none | ❌ Q9 |
| PD missing 100 ms | R21 | none (no monitor-fault scenario) | ❌ Q9 |
| forced PI fallback | R22 | controller param `force_fault_at_s` | ✅ (controller-side) |
| B10 learning (eff ±15 %, coupling ±20 %) | O5 | robust named corners / LHS worlds | ✅ (truth for scoring from `truth_final`) |

Comparators on G26: `tlc`, `tlc_pi`, `pi_lock_c`, `kalman_lqg_c`, `mpc_c`, `dither_lockin_c` (Q2).

---

## 4. Plan (framework first)

Each phase ends with something runnable. Tests target the framework (contracts, plumbing, fidelity), not TLC
performance on G26.

### Phase 0: groundwork
- [x] Answer §6.
- [ ] Archive the rev 1 `mrm/control/tlc/*` and `tests/tlc/*` (Q4).
- [ ] Reproducible env: `uv venv .venv && uv pip install -e .` (Python ≥ 3.12; numba, h5py; gymnasium optional). Document it in `TEST_INFRA.md`.
- [ ] Bring in the source per Q5.

### Phase 1: walking skeleton (end to end with a stub)
- [ ] Adapter + `@register("tlc")` with a **stub core**: hold the warm code + source-tone dither.
- [ ] TelemetryLog sink; `tools/tlc_report.py` (mrm metrics + ECE496 per-ring stats from traces).
- [ ] `experiments/tlc/steady_tlc.yaml`; `tools/tlc_run_all.sh` (or a `compare` YAML).
- [ ] Contract tests: action dtype/shape/range, telemetry keys/shapes, exactly 10 samples per core step, code↔u round trip, no `mrm.plant`/`mrm.sim` imports under `mrm/control/tlc/`.
- **Exit:** `mrm run experiments/tlc/steady_tlc.yaml` + report works. Every later phase swaps the stub for real parts.

### Phase 2: faithful core + fidelity check on the source plant
- [ ] Port the core (deviations 2.3-1, 2.3-2 only).
- [ ] Fidelity test: the ported core on the **ECE496 plant** reproduces `results/metrics.json` for `steady` and `d2_common` within tolerance (plant imported from the sibling checkout; skipped if absent).
- [ ] The ECE496 runner becomes the "4-ring reference" column of the report.

### Phase 3: core on G26
- [ ] Priors from `spec.nominal`; deviations 2.3-3, 2.3-4.
- [ ] Open-loop diagnostic (formerly the go/no-go spike, now informational): dither on a warm ring, log B3 y1/y2/ζ against the true detuning. It documents the lock-in SNR on G26.
- [ ] Closed loop, warm start, `tlc_pi` and `tlc`.

### Phase 4: scenario matrix
- [ ] One YAML per mapped row of §3 under `experiments/tlc/`, with comparators.
- [ ] Gap scenarios per Q9.
- [ ] Runtime budget check (Python core ~3 ms per 100 µs step; Q10).

### Phase 5: report
- [ ] `docs/tlc/results.md`: R-table, the ECE496-vs-G26 side by side, the scorecard rows, and known failure explanations.
- [ ] README status updated.

---

## 5. Risks
| Risk | Effect | Mitigation |
|---|---|---|
| Python core ~3 ms per 100 µs step × B worlds | MC / long D1 runs take minutes to hours | Short durations; per-world loop first; vectorise only if needed (Q10) |
| Source plant import needs the source's `sys.path` layout (`from params import …`) | Fidelity test is fragile | Test-only fixture that adds `../ECE496_project/thermal_control` to the path |
| Dither ≥ ±5 pm scores as excursions | TLC looks worse than it is | Report the dither envelope separately; Q8 |
| G26 thermal poles ≪ 100 µs | Source ROM gains must come from priors, or the MPC is mis-scaled | Deviation 2.3-4; log model vs truth |
| mrm main moves on | API drift | Pin the mrm commit in the report |

---

## 6. Decisions (2026-10-31, lead: Claude, owner: A. Patel)
The consultation questions are settled as below. Any of them can be reopened by the owner.

| # | Decision |
|---|---|
| Q1 | G26 is the higher-fidelity testbed: TLC-on-4-ring and TLC-on-G26 are reported side by side, with comparators. The primary arm is the faithful port; a disclosed a-priori re-parameterisation (dither amplitude, ζ-model gains from nominal data, never tuned on G26 outcomes) comes later as a second arm |
| Q2 | `tlc`, `tlc_pi`, `pi_lock_c`, `dither_lockin_c`, `kalman_lqg_c`, `mpc_c`; `wdm_supervisor` for cold start only (LQG and MPC are warm-start lock-only) |
| Q3 | pm primary, GHz alongside (1 GHz ≈ 5.72 pm) |
| Q4 | Rev 1 code moves to `archive/tlc_rev1/` (`git add -f`: main ignores `archive/`); its tests are not run |
| Q5 | Copy the source core into `mrm/control/tlc/core/` with Apache-2.0 attribution and the source commit (`f98ce84`) |
| Q6 | Branch section on top of main's README until this branch merges; then move it to `docs/tlc/README.md` |
| Q7 | Primary tier = the model-based baselines' tier: `[line_grid, ring_to_line, model_nominal]`. More access only as a disclosed arm. The report lists the tokens of every arm |
| Q8 | Both threshold sets (ECE496 converted; mrm 5 pm). Two error channels: total true detuning (mrm scoring) and dither-removed tracking error (true detuning notch-filtered at the dither tones) |
| Q9 | Not mapped until Phase 4. Then: common-mode aggressor and +10 K kick through mrm's scenario generators (no plant change; `mrm golden` must still pass); PD dropout at the adapter boundary. All labelled as extensions |
| Q10 | Short runs first (≤ 150 ms; D1 compressed). Cold-start wrong-line campaign: 300 worlds (95 % bound ≈ 1 % at zero events), sized after the Phase 2 step-time benchmark. Every rate gets a Clopper–Pearson interval |
| Q11 | `runs/` stays ignored; commit `docs/tlc/results.md` + JSON only |
| Q12 | Commit locally; pushing to `origin/feature/tlc-port` waits for the owner |

Mandatory model-trust corners (Phase 4): `g26_fast_thermal`, `g26_slow_thermal`, `g26_textbook_thermal`, plus a crosstalk corner still to be identified.

Upstream note: main is broken from `84de49b` (optics revamp: `mrm.sim.kernel` uses ~60 names `mrm.plant.optics` no longer defines). This branch merges main up to `ca3f8fe` only.
