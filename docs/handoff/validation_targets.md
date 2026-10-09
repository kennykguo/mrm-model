# Handoff from `validation_targets` (frozen calibration / held-out validation / bounds / excluded targets)

Owner of: `mrm/validation/targets/{calibration,validation,bounds,excluded}.yaml`, `mrm/validation/targets/README.md`,
`mrm/validation/targets_io.py`, the frozen-targets header block of `docs/VALIDATION.md`,
`tests/closure/test_targets_schema.py`, `tests/verification/test_validation_isolation.py`.

The split was frozen on 2026-10-09 and re-frozen the same day by the adversarial review, before any replica or
comparison existed (see "Review of 2026-10-09" below). `validation.yaml` and `bounds.yaml` are hashed (file bytes,
CRLF read as LF) in the header of `docs/VALIDATION.md`; any edit is a re-freeze (procedure in
`mrm/validation/targets/README.md`).

## Requests to other owners

1. **params (`mrm/params/sources.yaml`)**: please define these source keys, used by the target files:
   `BU-45CLO, CO22-JLT, CO22-TCPMT, CO24-JOM, DEC19, GF-ECIO, HC, HOTI26, LM-P, LM-T, LM26-P, LM26-S, OCP, PAT-4,
   PECC, SIB, SMTAI, YOLE, textbook-REF`. Tags follow the textbook (`docs/SOURCES_AND_GENERATIONS.md` section 2) and
   the research reports (01 section 1, 04 references, 03 for DEC19). `textbook-REF` stands for the textbook's
   `src/physics/index.ts` REF table where the textbook cites no source tag. `test_source_keys_exist_in_sources_yaml`
   is skipped until the file exists and then fails on any missing key.
2. **pyproject.toml owner**: add `"validation/targets/*.yaml"` and `"validation/targets/README.md"` to
   `[tool.setuptools.package-data] mrm`, or a non-editable install will not ship the target files.
3. **replicas (`mrm/validation/replicas.py`)**: implement every name in `targets_io.REPLICA_NAMES` (22 functions;
   table in the README). Suggested signature `fn(cfg_or_params, target: Target) -> prediction`, reading
   `target.replica_args`, and returning the shape `targets_io.evaluate` expects for the target's tolerance kind
   (README, "Tolerance semantics"). Notes: `notch_er_vs_heater_v` must support all-pass and add-drop (01 D14);
   `relOMA_map` needs `mode` in {oracle, setpoint_lock, fixed_code} and, for VAL-5, must align its loop-open peak to
   69.6 degC (the peak is not scored); `aggressor_trajectory` must use the rectified sine, not a parabola (see
   "Contract issues" 1). `test_replicas_module_implements_every_protocol` is skipped until replicas.py exists.
4. **report (`mrm/validation/report.py`)**: keep the block between `<!-- frozen-targets:begin` and
   `<!-- frozen-targets:end -->` in `docs/VALIDATION.md` byte-identical when regenerating the body. Score with
   `targets_io.evaluate`; count only `Target.scored` entries; show `unverified`, `deferred` and `cross_check` entries
   with their status. VAL-4c and BND-4 rest on the same printed "< 1 dB" spread (allowed: they are both held out).
   VAL-4c is not independent of VAL-4a/b (the map's maximum is the fitted CAL-14 level), so do not count it as a
   third independent pass. A `soft_fail` (BND-1 between 1 and 2 FWHM) is reported as marginal, not as a pass.
   Print the contamination warnings below in the report.
5. **calibration (`mrm/calibration/**`)**: read targets only through `load_calibration_targets()` or
   `load_targets("calibration")` with that literal. The isolation scan also rejects the strings "validation.yaml",
   "bounds.yaml" and "VALIDATION.md" anywhere under `mrm/calibration/`, comments and docstrings included. Do not fit
   CAL-6 (`fit: false`, the H86 value). Fit CAL-4 as shifts relative to 0 V and CAL-7 as the absolute offset (they
   share the 0 V datum; do not count it twice). CAL-12b is now the quartiles of the Guide line-error population
   (not a sigma bound); CAL-12d is sourced to [YOLE] via the textbook and `active`. Record the
   replica mode used for CAL-14 (oracle gives a lower bound on the modulation depth).
6. **params / presets and thermal (interim model)**: see the contamination warnings. Do not set the heater slew limit
   or code limits from the G26 slide-10 demand (EXC-13, 07 P1-11).
7. **scenarios**: the slide-10 aggressor is `T = 113.6 - 88.5 |sin(pi t / 130.7 ms)| degC` (CAL-13a), peak
   |dT/dt| 2127 degC/s, mean 1354 degC/s, with t = 0 at a cusp (T_max); CAL-13a is scored over one period in periodic
   steady state. The soak profile and the laser sweep are scenario inputs (EXC-14, EXC-17).
8. **params / presets (07 P1-6 FSR hypotheses)**: 07's "window A, 1.32 to 1.48 THz" is inconsistent with BND-1 in
   its interior: inside the 2.8 THz TX span the second order 2 x FSR counts, so 1.34 to 1.46 THz puts an order within
   2 FWHM of a line (at 1.40 THz exactly on line 7). Restrict window A to 1.32-1.34 or 1.46-1.48 THz, or demote it to
   a stress hypothesis like "G25 cluster" and "PDK size". `next_order_distance_ghz` (P1-6 diagnostics) must scan all
   orders inside the span, not FSR modulo 400 GHz.
9. **docs/CHANGELOG.md owner** (the file does not exist yet): add the line "2026-10-09 re-freeze of
   validation.yaml and bounds.yaml by the review of the target split, before any comparison; no validation value or
   tolerance changed; BND-1 windows corrected, BND-2 margin text corrected, BND-3 split into BND-3a/3b; reasons in
   the file headers and docs/handoff/validation_targets.md".
10. **replicas**: `aggressor_feasibility` now serves BND-3a (output `signed_error_pct_min_max`, error =
    (I - I_set)/I_set in %, either sign accepted) and BND-3b (output `monitor_deviation_ua_min_max`, I - I_set in uA);
    `adjacent_order_penalty` must scan every order inside the TX span (`orders: all_within_span_plus_2_fwhm`);
    `laser_line_statistics` outputs `max_abs_static_error_ghz` (CAL-12a), `static_error_quartiles_ghz` as
    {median, q25, q75} (CAL-12b), `max_abs_drift_24h_nm` (CAL-12c) and `max_abs_power_drift_24h_pct` (CAL-12d).

## Contamination warnings (research-report fits that used held-out data)

- 03 key finding 5: coupling-dispersion slope s = 1.48 %/nm was fitted to all 8 notch depths, including VAL-2
  (1.4 and 1.5 V). Do not use it as a default or prior; use the femwell slope (1.3-1.5 %/nm, 05 t02) or refit on CAL-2.
- 06 section 4.3 / V5: the Q and dlambda/dT pairs (for example "Q 3600-4400 at 86 pm/K") were matched to loop-open
  contours that include VAL-5's -16 dB contour. Not a Q prior.
- 06 M6: the 169-227 pm swing range comes from the -3.0 dB (VAL-4) and -2.1 dB levels. The swing prior must come
  from CAL-14 (-2.1 dB) alone.
- 07 P1-2 interim thermal model uses the imec pole family; comparing the interim model (not the FEM pipeline) with
  VAL-8 is tautological.

## Contract issues found (07 is normative; these say where it is wrong or doubtful)

1. **07 D6, P0-9 and P1-4 A1** describe the slide-10 aggressor as parabolic segments with a linear +-2200 degC/s
   sawtooth rate. 01 D7 and its verification log items 4-5 corrected this after 07 was written: the trace is a
   rectified sine (0.34 degC rms; a parabola fits 5x worse and would need a 2780 degC/s peak). CAL-13a uses the
   rectified sine; the scenario generator should too ("parabolic segments" in P0-9 should read "rectified sine").
2. **07 4.3 CAL-4** "droop of dlambda/V^2 of 20 to 30 %": the read-offs give 19 % (01) and 22 % (textbook) from 0.3 to
   1.4 V, with 6-9 % uncertainty from the 0.3 V point alone. The 30 % end is not supported by the curve.
3. **07 4.3 CAL-1, CAL-2** conditions "low power, cold, 0 V" are not printed in any source the research reached; the
   targets carry them as `null` with a note, and the replicas must state their assumption.
4. **07 4.4 VAL-6** cannot have a tolerance now and uses the same data as CAL-9; it is `deferred` with `blocked_by`.
5. **07 4.4 VAL-7** n_g = 2.73 is not confirmed by any research report (03 read de Cea 2019 but quotes only
   gamma_0, gamma_lin, gamma_rad, gamma_th, tau and Gamma = 0.65); the Optica full text could not be fetched here. It
   is `unverified` (reported, not scored). 07's question "is 2.4 MHz angular?" is resolved: gamma_th = 1/tau (0.42 us,
   matching the printed 0.5 us); 1/(2 pi 2.4 MHz) = 66 ns would contradict the text.
6. **07 4.4 VAL-8** mixes imec's measured time constants (5 us, 10 us; 15 um modulator) with imec's FE two-pole fit
   of a different device (3.2/13.0 us; 14 um filter, simulated). The fit is solver-against-solver and is a
   `cross_check` (VAL-8e). Its 63 % point is 7.35 us (04 V5 prints 7.3), not about 10 us. Added VAL-8c/d: imec's
   measured efficiencies (222 +- 10 and 843 +- 55 pm/mW).
7. **07 4.4 VAL-3**: CAL-10 is a population mean (30.9 +- 2 GHz at 1 V) while VAL-3's 35.4 GHz is one plotted
   device; added VAL-3b, the same device's ratio BW(2 V)/BW(1 V) = 1.07-1.11.
8. **07 P0-5 test path** `tests/test_validation_isolation.py`: placed in `tests/verification/` per 07 4.1 and the
   task.
9. **Unverified anchors.** Q (CAL-1), P_FSR / V_FSR (CAL-3), the bias ladder (CAL-5), responsivity (CAL-11) and the
   1309.5 nm operating line (CAL-7) all come from the OFC 2025 talk via the textbook; 01 could not reach the talk. They
   are the only sources for these numbers, so they stay `active` with `rechecked: false`.
10. **dlambda/dT evidence** (task note): the 400 x 200 nm strip at dn_Si/dT = 1.94e-4 /K gives 64.3 (06) to
    64.4 pm/K (05). The 66.3 pm/K "ceiling" of 06 is lambda*(dn_Si/dT)/n_g,Si at the C-band 1.86e-4 /K; at the 1310 nm
    1.94e-4 /K the thermo-optic ceiling is 69.1 pm/K, and 71.1 pm/K with expansion at 300 K (02). Recorded in CAL-6.
11. **relOMA k = 0.67 dB/degC^2** is a secant about 0.65 FWHM from the optimum, not a curvature (06 V5); EXC-4.
12. **Units**: the target files use the grammar of `mrm/params/units.py` (`codes`, `k_per_s`; `c`, `db`, `dbm`
    only alone); `"1"` is a targets-only marker for a dimensionless value. `targets_io.to_si` matches
    `mrm.params.units.to_si` for every unit in the files (tested).

## Review of 2026-10-09 (adversarial reviewer; now owner of these files)

Fixed in place (validation.yaml and bounds.yaml re-frozen the same day, before any replica or comparison existed;
no validation value or tolerance changed):

1. **BND-1 windows were wrong (physics).** 07 4.5 and the first BND-1 text list 1.32-1.48 THz as a window keeping
   >= 2 FWHM; that checks FSR mod 400 GHz, i.e. the first order only. Enumerating every ring, orders +-1..+-3 and the
   8 TX lines: windows between 1.0 and 2.9 THz are 1.06-1.08, 1.32-1.34, 1.46-1.48, 1.72-1.88, 2.12-2.28 and
   2.52-2.68 THz. The test had the same first-order-only flaw; it now enumerates all orders and asserts the window-A
   interior fails. BND-1's replica args now require all orders.
2. **BND-3 tolerance could not express its own description.** One band [-1.9, 3.0] % was applied to both the slide-10
   waveform and the 808 degC/s run, while the 808 criterion is "+-0.5 uA" (07 4.5). Split into BND-3a and BND-3b.
   BND-3a's band is asymmetric but the slide's sign convention is unpublished, so a replica using the opposite sign
   would fail spuriously; `evaluate` now supports `either_sign: true` for bands.
3. **BND-2 margin text** said 1.9-2.0 nm under H_solver; linear dlambda/dT from 66-69 to 74-77 pm/K gives 1.76-2.0 nm.
4. **Contract gaps (07 P0-5, 4.3 rule).** 13 entries lacked the condition keys heater_v|code, bias_v, p_in_dbm; the
   four bounds had no sigma; 8 read-off entries had no digitizer record and none recorded the crop. The schema now
   requires all of these and the files carry them ("not recorded" where 01 did not record a crop).
5. **CAL-12b statistic.** "+-15 GHz" (a yield limit; the plotted population has outliers to -21 GHz) was encoded as
   a bound on the sampled sigma, which accepts a spread 3x the plotted one. Now the population quartiles (textbook
   read-off, not re-checked) with the printed limit kept in `components`.
6. **CAL-12d source.** The textbook's authoring brief cites [YOLE] for the 2.5 %/24 h power drift; now `active`,
   `rechecked: false`, like CAL-12c.
7. **CAL-6 attribution**: 05's 64.4 pm/K uses the femwell index weight 1.086, not tidy3d.
8. **targets_io.evaluate**: z used the largest sigma of a list/mapping (CAL-4, CAL-9b) instead of each point's own
   sigma, understating z up to 2x (CAL-9b); now per element. The inequality "residual" was |worst - value| even on a pass
   (a margin reported as a miss); now the violation (0 on pass) with the margin in `detail`. A reversed band
   prediction [max, min] was silently accepted; now an error. NaN/inf predictions now fail explicitly. CAL-13a's
   mixed-unit sigma mapping no longer produces a meaningless z.
9. **Isolation scanner bypasses**: `from ...targets_io import load_targets as lt; lt("validation")`, relative-import
   aliases, `f = targets_io.load_targets`, `getattr(targets_io, "load_all")`, `importlib.import_module(
   "mrm.validation.report")`, `from ...targets_io import *` and `load_excluded` all passed the first scanner; all are
   now flagged and tested. It remains a guard against accidents, not a sandbox.
10. **Hash portability**: `file_sha256` reads CRLF as LF, so a Windows `core.autocrlf` checkout does not look like a
    re-freeze; any other byte change is still detected (tested).

Not changed, but doubtful or open:

- VAL-4a/4b/4c, VAL-5 and CAL-14 rest on slide-9 read-offs that no verifier re-checked; they stay `active`.
- VAL-7b's Gamma = 0.65 depends on de Cea's definition (Poynting-normalized, per 03); a power fraction and an energy
  fraction differ by about 10 % for strips (06 2.5), more than the +-0.05 tolerance. The replica must reproduce the
  paper's definition exactly; the de Cea cross-section is still not in docs/research, so VAL-7a-c cannot run yet.
- BND-3b rests on a plot whose time axis is aliased (EXC-3); a failure by short spikes only is weak evidence.
- CAL-13a is phase-sensitive (t0 = 0 at a cusp); the replica must align its waveform, or the rms is meaningless.
