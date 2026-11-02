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
1. **R13 ER floor (owner decision).** G26 gives 2.8-3.0 dB ER at the inflection point vs the source's 3.5 dB, so
   the source B6 finds no setpoint. The G26 priors drop the floor to 0 dB; confirm or overrule.
2. **Upstream main is broken from `84de49b`** (the simulator still references ~60 renamed optics names). This
   branch merges main only up to `ca3f8fe`; report upstream and re-merge once fixed. Some
   `tests/verification/test_ctrl_c_*` tests already fail at `ca3f8fe`, with or without this branch.
3. **O5 (B10 learning)** is not mapped: scoring needs each world's true heater efficiency.
4. **300-trial cold-start campaign** (8 trials committed): `tools/tlc_run.py
   experiments/tlc/cold_campaign_tlc.yaml --set experiment.batch=300` (~2 h for the TLC arm), same for `_wdm`.
5. **Model-mismatch corners.** Corner runs give the TLC the corner's own priors; a variant with g26 priors on a
   corner plant is still to do.
6. **Restart-one-ring (R8 variant)** has no mrm equivalent (`line_enable` restarts the laser, not the controller).
7. **Refresh `docs/tlc/SUMMARY.md`** once items 1, 4 and 5 land (the R13 decision and the 300-trial campaign change its numbers).
8. **Optional second arm** (spec 6, Q1): a disclosed a-priori re-parameterisation (dither amplitude, ζ-model
   gains), only after the faithful results are reviewed.

> ⚠️ The rev 1 status ("Phase 1 & 2 COMPLETE, 140/140") is withdrawn. See spec §1.
