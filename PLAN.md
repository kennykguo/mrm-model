# TLC Port: Status Tracker

**Branch:** `feature/tlc-port` · **Spec & plan:** [`docs/TLC_PORT_SPEC_AND_PLAN.md`](docs/TLC_PORT_SPEC_AND_PLAN.md) (rev 3)
**Source:** `../ECE496_project/thermal_control/` @ `f98ce84`
**Goal:** run the ECE496 TLC against mrm-model's G26 plant and collect metrics. This branch builds the framework; good TLC results are not required yet.

| Phase | What | Status |
|---|---|---|
| 0 | Answer spec §6 questions, archive rev 1 code, reproducible env, bring in the source | decisions DONE (spec §6); rest IN PROGRESS |
| 1 | Walking skeleton: adapter + stub core + telemetry log + report + `steady_tlc.yaml` | IN PROGRESS |
| 2 | Faithful core port + fidelity check on the ECE496 plant | OPEN |
| 3 | Core on G26: priors, open-loop lock-in diagnostic, warm closed loop | OPEN |
| 4 | Scenario matrix (ECE496 D1–D6 / R-reqs → mrm scenarios) | OPEN |
| 5 | Results report + README status | OPEN |

> ⚠️ The rev 1 status ("Phase 1 & 2 COMPLETE, 140/140") is withdrawn. See spec §1.
