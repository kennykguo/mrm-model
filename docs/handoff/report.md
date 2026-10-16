# Handoff: report (`mrm/validation/report.py`, `sensitivity.py`, `cli.py`)

The builder and reviewer of this task could not write this file (the tool refuses report files from subagents).
The integrator kept the record in docs/handoff/integrator.md section 9.4; it is copied here unchanged so the
report owner has a handoff in the usual place. Source of truth: integrator.md section 9.4.


- Files: `mrm/validation/report.py` (generator), `mrm/validation/sensitivity.py` (Sobol / Saltelli + Jansen with
  bootstrap intervals, the binned given-data S1, the u-map of the 07 4.8 set, design injection, tornado, Jacobian SVD
  and the half-prior rule; free of target-file access, so mrm/calibration may import it), `mrm/validation/cli.py`
  (`mrm report`), `tests/verification/test_report.py`, `tests/verification/test_sensitivity.py`,
  `docs/validation/<preset>/` (report_data.json and SVG charts).
- API: `generate(preset="g26", *, g25_preset=None, recompute=None|()|parts, out=None, n_base=128,
  compare_worlds=256, compare_run=None, n_threads=4, stamp=None, log=...) -> Path`; `check(preset=None, out=None) ->
  list[str]`; `render(data, *, preset, g25_preset=None, stamp=None, doc_text=None, charts=True, chart_dir=None) ->
  str`; `compute_predictions | compute_identifiability | compute_sensitivity | compute_ranking`; `companion_g25`;
  `load_data / save_data / data_path / data_dir`; constants `SCHEMA = "mrm.validation.report/1"`, `PARTS`,
  `CAL_PARAMETERS` (18 leaves), `SENS_METRICS`, `SENS_CONTROLLER = "pi_lock_c"`, `SENS_TOP = 8`, `CODE_DEPS`,
  `FIT_RECORD_PRESETS`, `CALIBRATED_FLAGS`. CLI `[--preset] [--g25-preset] [--fast | --recompute a,b] [--check]
  [--out] [--sobol-n 128] [--compare-worlds 256] [--compare-run DIR] [--threads 4]`.
- Deviations (stated in the report): the half-prior rule only in its linearized form (profile likelihoods come from
  the calibration record, section 3.1); the band is a range over the nominal world and the 12 named corners, not a
  probability band; g26_hot_selfheat cannot be applied to the G25 bench (`uncertainty.chip_input_loss` has no
  `ring_plus8dbm` corner there), so G25 rows count 11 corners; Sobol and tornado use pi_lock_c under A1 with
  variation off, conditional indices over the top 8; worlds are injected by patching scipy's LatinHypercube (checked
  against realize per chunk); the Jacobian scale without a printed sigma is a quarter of an interval tolerance or half
  an abs/rel tolerance.
- The reviewer's corrections: section 3.1 now reads the real fit record and audits every target id it used
  (CONTAMINATION for held-out ids; EXC, unknown and fit:false ids flagged); the Jacobian moves a leaf only on plants
  that share its G26 value; the closure-test scan parses `xfail` decorators (the builder's "1 of 12 rows covered" was
  a false match: **0 of 12**); section 4 measures the noise share (at most 5e-9 %) and reports the binned S1 (sums
  0.29-0.63) and the sums of the ST lower bounds (1.09-1.35): interaction dominance is supported; the staleness check
  covers code (`CODE_DEPS`); and **CAL-10 identifies (R_s + 50 ohm) x C_j, not R_s x C_j** (near-null direction 17,
  loadings C_j +0.84, R_s -0.54 in prior half-widths). Every published g26 number was reproduced bit for bit by the
  reviewer's recompute.
- Requests: **closure owner** (tests/closure): one `xfail(strict=True, reason="VALIDATION.md#closure-<id>: ...")`
  test per 07 4.6 row: dlambda-dt, fsr, thickness, q, heater-efficiency, rth, reloma-level, reloma-curvature,
  aggressor-rate, g25-topology, eo-bandwidth, rom-dc-scale (none exists). **params owner**: a public
  `realize(..., u=(B, D))` (or `uncertainty="design"`) so the LatinHypercube patch can go; the `ring_plus8dbm` corner
  of `uncertainty.chip_input_loss` in g25_test_vehicle; fill_fraction has no effect on any calibration replica with
  the analytic junction (zero Jacobian column). **replicas owner**: CAL-9a/b/c recompute the same 8-die protocol 3
  times per pass (one call would cut the Jacobian by a third).
- Open: about 30 % of the section 4/5 worlds start supply-limited (the assumed 150 mA / 0.2 W array heater limit,
  section 8.3); Saltelli S1 is poorly resolved at 128 base rows (ST is resolved; ~1.3 s per world for more);
  BND-3a/b blocked (named controllers), VAL-7c blocked (no de Cea stack).
- Costs measured on g26 (2026-10-10): predictions 515 s, identifiability 1167 s, sensitivity 1412 s, ranking
  1681 s (76 min, peak 1.09 GB); on g26_calibrated (2026-10-16, this stage): 420, 696, 910, 825 s (2856 s with the
  render, peak 1.07 GB). `--fast` ~5 s.

