# TLC Port: Spec

**Target:** `mrm-model` (G26 8-ring plant) · **Source:** `ECE496_project/thermal_control/` (sibling checkout
`../ECE496_project`, commit `ae9556b`) · **Branch:** `feature/tlc-port`
**Results:** [`docs/tlc/SUMMARY.md`](tlc/SUMMARY.md) · **Status and open items:** [`PLAN.md`](../PLAN.md)

Section numbers are cited from the code (e.g. "spec 2.3", "spec 6, Q1"); keep them stable.

---

## 0. Objective and scope

Run the ECE496 thermal locking controller (TLC) closed loop against mrm-model's G26 plant, measure it against
mrm's own controllers, and report it in both projects' terms.

- **Faithful arm** (`tlc`): the ECE496 core exactly as published, only adapted at its interfaces (2.3).
- **Second arm** (`tlc_arm2`, `tlc_wdm`): disclosed a-priori adaptations for G26 (Q1), reported beside the faithful
  arm, never in place of it.
- **Out of scope:** fixed-point / RTL, real-time compute, changing mrm's plant.

**Definition of done:** one command (`tools/tlc_run_all.sh`) runs the full matrix (3) and writes the report. Every
cell holds a number, a documented failure, or "not mapped (reason)".

---

## 1. Source plant vs G26

| | ECE496 4-ring plant | G26 (mrm) |
|---|---|---|
| Units | GHz; u = heater power fraction | pm; 16-bit voltage-DAC codes (power ∝ code²) |
| Lineshape | HWHM ~20 GHz (~115 pm) | HWHM ~32 GHz (NRZ-broadened) |
| Heater poles | 13 µs + 78 µs | 3.2 µs (41 %) + 13 µs (54 %) + 4 ms tail (5 %) |
| Crosstalk | 20 / 6 / 2 % at 250 µs | 3 % nearest, then 1.5 % to every ring, ms time constants |
| Monitor | 12-bit, 1 MS/s, data-pattern noise | 16-bit, 250 kS/s, integrate-and-dump, NRZ-averaged (no pattern noise) |
| Fast aggressor | D2, ~200 pm/ms, 3 K swing | A1, ~160–180 pm/ms, 88.5 K swing |
| Rings | 4 | 8 |
| Controller call rate | 100 µs blocks of 100 PD samples, 10 heater samples out | `step()` every 10 µs, ~2–3 ADC samples per call |

1 GHz ≈ 5.72 pm at 1310 nm. Ring 0 runs out of heater under A1 for every controller (`docs/KNOWN_ISSUES.md` #38),
so A1 is judged on rings 1-7.

---

## 2. Spec

### 2.1 Architecture: port the controller unchanged, adapt the interfaces
```
  mrm runner (1 µs plant, step() every 10 µs)
        │ obs (codes)                                     ▲ {"tx.heater": codes}
        ▼                                                 │
  ┌────────────── mrm/control/tlc/controller.py  (registry "tlc") ──────────────┐
  │ Adapter (mrm-specific)                                                      │
  │  • ingest: tx.monitor.samples/count → normalised floats → 100 µs block      │
  │  • every 10th edge: core.step(block) → 10 heater samples (u, power frac)    │
  │  • every edge: emit next sample: u → code = round(code_max·√u)              │
  │  • priors: spec.nominal tokens → PlantParams (never mrm.plant)              │
  │  • telemetry: (B,N) numeric, std keys + TLC extras; TelemetryLog sink       │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Core = ECE496 tlc/ + params.py + optics.py (mrm/control/tlc/core/)          │
  └─────────────────────────────────────────────────────────────────────────────┘
```
- **The core is the ECE496 algorithm**, in the source's units (GHz, power fraction), diffable against the source.
- **The adapter is the only mrm-specific code**: units, DAC law, sample buffering, multirate schedule, priors,
  telemetry. One core per world (`B`).
- **Second-arm code lives beside the core, never in it:** `arm2.py` (a `TLC` subclass) and `hybrid.py`.

### 2.2 Interface contract
| Item | Value |
|---|---|
| Registration | `tlc` (faithful), `tlc_pi` (source `BaselinePI`), `tlc_arm2` (second arm), `tlc_wdm` (WDM supervisor acquires, `tlc_arm2` tracks) |
| Reads | `tx.monitor.samples`, `tx.monitor.count`, `tx.heater.applied`, `monitor_valid` |
| Writes | `{"tx.heater": int (B,N)}` within `spec.heater_dac` (`tlc_wdm` also holds the supervisor's `tx.assignment`) |
| Knowledge tokens | Q7 |
| Telemetry, standard | `state` (Mode int), `locked`, `detuning_est_pm`, `integrator`, `integrator_saturated` |
| Telemetry, extras | `rate_est_pm_per_ms`, `sigma_pm`, `dither_amp_pm`, `eff_learned`, `squelch`, `fault`, `core_step_us` |
| Params | `core`, `core_every_n_edges=10`, `start_mode` (`warm`/`cold`), `dither_shape`, `tone_plan`, `force_fault_at_s`, `pd_fault_*`; second arm only: `cp_overrides`, `pp_overrides`, `arm2` |
| Start | `warm`: the applied code → core `warm_start` (initial estimate = the core's own setpoint, never the true detuning). `cold`: core `cold_start` (source ACQUIRE) |

### 2.3 Allowed deviations from the source in the faithful arm (complete list; anything else is a bug)
1. The ring count is `ControllerParams.n_rings` instead of the module constant `N`.
2. Params are injected; the source defaults stay the defaults.
3. PD rate 250 kS/s with 25 samples per core step (no decimation, FIR cutoff 60 kHz).
4. Plant knowledge (lineshape, heater gain and poles, crosstalk, unheated offset, monitor noise) comes from priors
   derived from the nominal G26 design, not `PlantParams()` defaults. The source's Johnson / shot noise terms (its own
   TIA assumptions) are off: the G26 monitor sigma already is the whole noise, and it sets the squelch floor.
5. 8-ring tone plans: the 250 Hz-grid sine plan (4-13.5 kHz) and, for the sampled dither shapes, the source's four
   out_rate/N tones repeated every four rings (`tone_plan.py`).
6. R13 extinction-ratio floor 0 dB on G26 (owner, 2026-11-09): G26 reaches 2.8-3.0 dB at the inflection point,
   below the source's 3.5 dB, so the source B6 finds no setpoint.

Checks: `tests/tlc/test_core_equivalence.py` (bit-identical to the source on its own plant, sine and triangle
dither) and `test_core_fidelity.py` (the published R1/R3/R4 numbers, on the source plant with its `f98ce84` noise
settings, because the source did not regenerate `results/metrics.json` after adding its noise model).

### 2.4 Metrics and report
- **mrm metrics** (from `mrm run`): `max_abs_lock_error_pm`, `rms_lock_error_pm`, `locked_final`, `wrong_line_*`,
  `time_to_lock_s`, `oma_penalty_db`, `heater_*`, BER.
- **ECE496-style metrics** (from the recorded true-detuning trace, per ring, pm and GHz): p99.9 / peak / rms; the
  same error with the TLC's dither tones removed (`track_*`, Q8); Monte Carlo rates with Clopper-Pearson intervals.
- **Report:** `tools/tlc_report.py` → `docs/tlc/results.md` + `.json`; written summary `docs/tlc/SUMMARY.md`;
  interactive page `docs/tlc/tlc_g26_evaluation.html` (`docs/tlc/evaluation_src/`).

---

## 3. Scenario matrix (ECE496 → mrm)
`experiments/tlc/matrix.yaml`, run by `tools/tlc_matrix.py`.

| ECE496 scenario | Req | Experiment (`experiments/tlc/`) | mrm scenario | Mapping |
|---|---|---|---|---|
| steady | R1 | `steady_tlc_faithful.yaml` | `static`, warm | direct |
| D1 die-temperature ramp | R2 | `xpu_step_tlc.yaml` | `xpu_power_step`, 120 K/s | close |
| D2 local aggressor | R3, R6 | `a1_tlc.yaml` | `a1`, one period | direct (rings 1-7) |
| D4 laser drift | R5 | `laser_drift_tlc.yaml` | `laser_drift`, 24 h limit in 0.2 s | close |
| D5 laser power step | R7 | `power_step_tlc.yaml` | `fibre_loss_step`, +3 dB | close |
| D6 squelch | R12 | `squelch_tlc.yaml` | `squelch` 10-30 ms | direct |
| cold start | R8 | `cold_tlc.yaml`, `cold_wdm.yaml`, `cold_hybrid.yaml` | `static`, cold | direct |
| random cold-start campaign | R9 | `cold_campaign_{tlc,wdm,hybrid}.yaml` | `cold_start`, ambient 0-105 °C, 8 worlds | direct |
| forced PI fallback | R22 | `fallback_tlc.yaml` | `static` + `force_fault_at_s` | controller side |
| D2 common aggressor | R3, R4 | `ext_common_aggressor_tlc.yaml` | die-wide `xpu_power_step`, +3 K at 2200 K/s | extension |
| +10 K kick | R10 | `ext_kick_tlc.yaml` | die-wide `xpu_power_step`, +10 K in 20 µs | extension (common, not local) |
| PD missing | R21 | `ext_pd_fault_tlc.yaml` | `static` + adapter `pd_fault_*` | extension (controller boundary) |
| D3 + D2, 1.5x | R11 | `a1_pd_adjacent_tlc.yaml` | A1 at the PD-side stress site | approximate |
| restart one ring | R8 | — | `line_enable` restarts the laser, not the controller | not mapped |
| B10 learning | O5 | — | needs each world's true heater efficiency | not mapped |

Controllers: `tlc`, `tlc_arm2`, `tlc_pi`, `pi_lock_c`, `dither_lockin_c`, `kalman_lqg_c`, `mpc_c` on every plain
experiment; cold start with `tlc`, `wdm_supervisor` and `tlc_wdm`. Model-trust corners (fast, slow and textbook
thermal, TEC-held package) for steady state and A1, each with corner priors and with nominal priors (mismatch).

---

## 4. Status
Tracked in [`PLAN.md`](../PLAN.md); history in git.

## 5. Known limits
- The Python core takes ~1.5 ms per 100 µs step (8 rings): about 15x slower than real time.
- Most cells are one world; only the cold-start campaigns are Monte Carlo.
- mrm `main` does not build since its optics rewrite (`plant.py:248`, `far_hold`); this branch merges main up to
  `ca3f8fe` (= rewritten `71004ea`). Merge later with `ca3f8fe` as the base.

---

## 6. Decisions (owner: A. Patel)

| # | Decision |
|---|---|
| Q1 | G26 is the higher-fidelity testbed. The primary arm is the faithful port; a disclosed a-priori second arm (G26 adaptations, never tuned per scenario outcome) is reported beside it |
| Q2 | Comparators: `tlc_pi`, `pi_lock_c`, `dither_lockin_c`, `kalman_lqg_c`, `mpc_c`; `wdm_supervisor` for cold start only |
| Q3 | pm primary, GHz alongside |
| Q5 | The source core is copied into `mrm/control/tlc/core/` with Apache-2.0 attribution and the source commit |
| Q6 | Branch section on top of main's README until this branch merges; then move it to `docs/tlc/README.md` |
| Q7 | Knowledge tier = the model-based baselines' tier `[line_grid, ring_to_line, model_nominal]`; `tlc_wdm` adds the WDM supervisor's tokens for acquisition. Every arm's tokens are listed in its experiment |
| Q8 | Both threshold sets (ECE496 2 GHz ≈ 11.4 pm; mrm 5 pm); total error and dither-removed tracking error |
| Q9 | G26-missing disturbances through mrm's scenario generators (no plant change), labelled as extensions |
| Q10 | Short runs (≤ 150 ms); campaigns get Clopper-Pearson intervals |
| Q11 | `runs/` stays ignored; commit the report (`docs/tlc/results.md` + JSON) only |
| R13 | ER floor 0 dB on G26 (2026-11-09) |
| O5 | B10 learning and restart-one-ring not mapped; reported as limits (2026-11-09) |
