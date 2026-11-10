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
| 4 | Scenario matrix (ECE496 D1–D6 / R-reqs → mrm scenarios) | **DONE**; corner variants with corner and nominal priors |
| 5 | Results report + README status | **first pass DONE** ([`docs/tlc/SUMMARY.md`](docs/tlc/SUMMARY.md)); refresh after the open items |

Run everything: `tools/tlc_run_all.sh` (~10 min) → `docs/tlc/results.md`.

## Open items
1. **Source update: 3-step sign dither (B1).** The faithful baseline follows the source, so the new dither replaces
   the cosine dither once it is published; then re-vendor, rerun the matrix and run the TLC arm of item 3.
   Blocked: the change is not on `msilve160/ECE496_project` yet (newest commit is still `f98ce84`).
2. **Upstream main is broken** (checked at `e77b124`, 2026-11-09). Building the engine fails with
   `pack_self_heating_optical_state() got an unexpected keyword argument 'far_hold'` (`mrm/plant/plant.py:241`);
   ~66 old optics names remain in `mrm/sim/kernel.py`, `mrm/sim/kernel_ref.py` and `mrm/plant/plant.py`, and five
   test files fail at import. Main was also force-pushed with rewritten history (old `ca3f8fe` = new `71004ea`), so
   a plain `git merge origin/main` refuses (unrelated histories). Once main is fixed, merge with `ca3f8fe` as the
   base (`git merge-tree --merge-base=ca3f8fe HEAD origin/main`, then a commit with both parents); a trial merge
   this way has no conflicts. To report upstream.
3. **300-trial cold-start campaign** (8 trials committed): `tools/tlc_run.py
   experiments/tlc/cold_campaign_tlc.yaml --set experiment.batch=300`, same for `_wdm`.
   - `wdm_supervisor` arm done (2026-11-10, 24 min): wrong line 12 / 2400 ring trials [0.0026, 0.0087]
     (12 / 300 worlds); not locked at 0.8 s 27 / 2400 [0.0074, 0.016] (24 / 300 worlds).
   - TLC arm (~2 h) waits for item 1.
4. **Corner runs: numbers provisional until item 1.** `matrix.yaml` `corners` has two variants, both with `tlc`,
   `pi_lock_c`, `kalman_lqg_c` and `mpc_c`: `knowledge: corner` (corner priors) and `knowledge: nominal` (g26 priors
   on the corner plant). Run one variant with `tools/tlc_matrix.py --corners-only --knowledge {corner,nominal}`.
   First results: every A1 verdict is the same under both priors, for the TLC (TEC-package ride-through included)
   and for `kalman_lqg_c` / `mpc_c` (rings 1-7 peak 12.6 / 33 / lost / 3.6 pm on fast / slow / textbook / TEC).
5. **Refresh `docs/tlc/SUMMARY.md`** once items 1, 3 and 4 land.
6. **Second arm** (spec 6, Q1): a disclosed a-priori re-parameterisation (dither amplitude, ζ-model gains). In
   scope, after the item 5 refresh is committed.

## Decided
- **R13 ER floor: 0 dB on G26** (owner, 2026-11-09). G26 gives 2.8-3.0 dB ER at the inflection point vs the
  source's 3.5 dB, so the source B6 finds no setpoint; the deviation stays disclosed.
- **O5 (B10 learning) and restart-one-ring (R8 variant): not mapped** (owner, 2026-11-09). Reported as limits of
  the pass. O5 scoring needs each world's true heater efficiency; `line_enable` restarts the laser, not the
  controller.

The rev 1 port and its status ("Phase 1 & 2 COMPLETE, 140/140") are withdrawn; see spec §1 and `archive/tlc_rev1/`.
