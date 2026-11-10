# TLC Port: Status Tracker

**Branch:** `feature/tlc-port` · **Spec & plan:** [`docs/TLC_PORT_SPEC_AND_PLAN.md`](docs/TLC_PORT_SPEC_AND_PLAN.md) (rev 3)
**Source:** `../ECE496_project/thermal_control/` @ `f98ce84`
**Goal:** run the ECE496 TLC against mrm-model's G26 plant and collect metrics. This branch builds the framework; good TLC results are not required yet.

| Phase | What | Status |
|---|---|---|
| 0 | Answer spec §6 questions, archive rev 1 code, reproducible env, bring in the source | **DONE** |
| 1 | Walking skeleton: adapter + stub core + telemetry log + report + `steady_tlc.yaml` | **DONE** (`tools/tlc_run_all.sh`) |
| 2 | Faithful core port + fidelity check on the ECE496 plant | **DONE** (bit-identical; published R1/R3/R4 reproduced) |
| 3 | Core on G26: priors, open-loop lock-in diagnostic, warm closed loop | **DONE** (steady state; findings in spec §4 Phase 3) |
| 4 | Scenario matrix (ECE496 D1–D6 / R-reqs → mrm scenarios) | **framework DONE**; open items below |
| 5 | Results report + README status | **first pass DONE** ([`docs/tlc/SUMMARY.md`](docs/tlc/SUMMARY.md)); refresh after the open items |

Run everything: `tools/tlc_run_all.sh` (~10 min) → `docs/tlc/results.md`.

## Open items
1. **Source update: 3-step sign dither (B1).** The faithful baseline follows the source, so the new dither replaces
   the cosine dither once it is published. Then re-vendor, rerun the matrix and run the TLC arm of item 3. Blocked:
   the change is not on `msilve160/ECE496_project` yet (newest commit is still `f98ce84`).
2. **Upstream main is broken from `84de49b`** (the simulator still references ~60 renamed optics names). This
   branch merges main only up to `ca3f8fe`; report upstream and re-merge once fixed. Some
   `tests/verification/test_ctrl_c_*` tests already fail at `ca3f8fe`, with or without this branch.
3. **300-trial cold-start campaign** (8 trials committed): `tools/tlc_run.py
   experiments/tlc/cold_campaign_tlc.yaml --set experiment.batch=300` (~2 h for the TLC arm), same for `_wdm`.
   The `_wdm` arm is running; the TLC arm waits for item 1.
4. **Model-mismatch corners.** Corner runs give the TLC the corner's own priors; a variant with g26 priors on a
   corner plant is in progress.
5. **Refresh `docs/tlc/SUMMARY.md`** once items 1, 3 and 4 land.
6. **Second arm** (spec 6, Q1): a disclosed a-priori re-parameterisation (dither amplitude, ζ-model gains). In
   scope, after the item 5 refresh is committed.

## Decided
- **R13 ER floor: 0 dB on G26** (owner, 2026-11-09). G26 gives 2.8-3.0 dB ER at the inflection point vs the
  source's 3.5 dB, so the source B6 finds no setpoint; the deviation stays disclosed.
- **O5 (B10 learning) and restart-one-ring (R8 variant): not mapped** (owner, 2026-11-09). Reported as limits of
  the pass. O5 scoring needs each world's true heater efficiency; `line_enable` restarts the laser, not the
  controller.

> ⚠️ The rev 1 status ("Phase 1 & 2 COMPLETE, 140/140") is withdrawn. See spec §1.
