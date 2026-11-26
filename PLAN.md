# TLC Port: Status

**Branch:** `feature/tlc-port` · **Spec:** [`docs/TLC_PORT_SPEC_AND_PLAN.md`](docs/TLC_PORT_SPEC_AND_PLAN.md) ·
**Results:** [`docs/tlc/SUMMARY.md`](docs/tlc/SUMMARY.md) (tables: [`docs/tlc/results.md`](docs/tlc/results.md);
page: [`docs/tlc/tlc_g26_evaluation.html`](docs/tlc/tlc_g26_evaluation.html))
**Source:** `../ECE496_project/thermal_control/` @ `ae9556b` · **Updated:** 2026-11-26

Run everything: `tools/tlc_run_all.sh` (~45 min, 151 runs) → `docs/tlc/results.md`.

## Done
- Framework: adapter, faithful vendored core (bit-identical to the source, published R1/R3/R4 reproduced), G26
  priors, scenario matrix with mrm comparators, corners with corner and nominal priors, cold-start campaigns, report.
- Source update `ae9556b` re-vendored (sampled dither shapes, monitor noise model); dither-shape comparison
  ([`docs/tlc/dither_compare.md`](docs/tlc/dither_compare.md)): sine stays the default on G26.
- Second arm (`tlc_arm2`, `tlc_wdm`): capped dither, stiff DC scale, red-side waiting re-lock, WDM-supervisor
  acquisition. In the main matrix beside the faithful arm.

## Open
1. **Power step (R7):** every controller loses lock; the tuned TLC needs blue fall-off detection (`arm2.fall_level`,
   off) plus a re-acquisition the stuck-sensor check (B2) does not block.
2. **Squelch (R12):** 20 pm, level with mrm's controllers; give the TLC the squelch state or freeze on a collapsed
   dither response.
3. **Statistics:** 300-world cold-start campaign for `tlc_wdm`
   (`--set experiment.batch=300` on `cold_campaign_hybrid.yaml`); more worlds per scenario.
4. **Upstream mrm `main` is broken** (`plant.py:248` passes `far_hold` to an optics packer that no longer takes it;
   the kernel still uses the pre-rewrite optics layout; checked at `d6983ca`, 2026-11-25). Merge once fixed, with
   `ca3f8fe` as the base: `git merge-tree --write-tree --merge-base=ca3f8fe HEAD origin/main`, then a commit with
   both parents (a trial merge has no conflicts).
5. **Upstream ECE496:** `results/metrics.json` predates the source's noise model; the fidelity gate runs the source
   plant with the `f98ce84` noise settings until it is regenerated.

## Decided
- R13 ER floor 0 dB on G26; B10 learning (O5) and restart-one-ring not mapped (owner, 2026-11-09).
- The second arm is reported beside the faithful arm, never instead of it (spec 6, Q1).
