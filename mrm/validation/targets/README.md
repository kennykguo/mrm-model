# calibration, validation, bounds, and excluded targets

the split separates values used to fit the model from values used to test it.
it was fixed before fitting, so a prediction cannot pass simply because it
repeats its own input. for example, 86 pm/K is an input hypothesis. a thermal
resistance of about 4.7 K/mW calculated as 0.405/0.086 is also a consistency
check, not new evidence.
see docs/ARCHITECTURE.md for the accuracy rules.

| File | Role | Entries | Who may read it |
|---|---|---|---|
| `calibration.yaml` | the only numbers `mrm/calibration` may fit | CAL-1 ... CAL-14 (21 entries with sub-ids) | everyone |
| `validation.yaml` | held-out targets, tolerances fixed now | VAL-1 ... VAL-8 (18 entries) | everyone **except** `mrm/calibration/**` |
| `bounds.yaml` | necessary conditions from the working G26 link | BND-1 ... BND-4 (5 entries — BND-3a/b) | everyone **except** `mrm/calibration/**` |
| `excluded.yaml` | numbers deliberately not fitted and not scored, with reasons (the design review's list plus additions) | EXC-1 ... EXC-18 | everyone |

`validation.yaml` and `bounds.yaml` are **frozen**, including their comments.
SHA-256 hashes, with CRLF normalized to LF, are recorded in `docs/VALIDATION.md`.
`test_validation_isolation.py` checks both hashes and scans calibration code
for accidental access to held-out data.

## how do i load and evaluate targets?

```python
from mrm.validation.targets_io import load_calibration_targets, load_all, evaluate

cal = load_calibration_targets()            # the only call mrm/calibration should make
everything = load_all()                     # report and tests only: {'calibration', 'validation', 'bounds', 'excluded'}
ev = evaluate(everything["validation"][0], prediction)   # Evaluation(scored, passed, level, max_abs_residual, z, ...)
```

`python -m mrm.validation.targets_io check` runs the schema, the split rules, the frozen-hash check and the isolation
scan. `python -m mrm.validation.targets_io header` prints the header block (used only for a re-freeze).

## what does each target field mean?

| Field | Meaning |
|---|---|
| `id` | `CAL-n[a-z]`, `VAL-n[a-z]`, `BND-n`, `EXC-n` — the design review's numbering, with letter suffixes where the review merged several quantities in one row |
| `quantity` | snake_case name — if it ends in a unit token it must be the entry's unit |
| `description` | one sentence |
| `kind` | `scalar`, `interval` ([lo, hi]), `curve` ({x_name, x_unit, x, y}), `set` (named numbers), `parametric` ({form, params} — only `rectified_sine`), `contour_offsets` (VAL-5), `inequality` and `band` (bounds) |
| `value`, `unit` | as printed by the source, in the key-suffix grammar of `mrm/params/units.py` (`nm`, `pm_per_k`, `k_per_s`, `codes`, `db`, ... — `c`, `db` and `dbm` only alone, so a degC/s rate is `k_per_s`). `"1"` marks a dimensionless value — `mixed` is allowed only for a `set` whose keys carry their own unit suffix. `targets_io.to_si` converts with the same conventions as `mrm.params.units` (dB becomes a natural-log power ratio) |
| `sigma` | required — one standard deviation in `unit`, including read-off (digitization) error — `null` when the value is a specification, a printed bound or a derived bound |
| `components`, `population` | the separate readings behind a value, and population statistics (wafer maps, part counts) |
| `conditions` | required keys — `t_c`, `heater_v` or `heater_code`, `bias_v`, `p_in_dbm`, `port` (through, drop, output, none) — also `t_reference` (stage, tec, ring, on_die_sensor, aggressor_trajectory, unknown), `loop` (open, closed, none), and `detuning`, `lambda_nm`, `window_s`, `period_s`, `grid` where relevant. `null` means "not printed" or "not applicable" |
| `generation`, `source`, `locator`, `provenance` | closed sets of ARCHITECTURE.md section 1 — `source` is a key (or list of keys) for `mrm/params/sources.yaml` — `locator` says where in the source |
| `readoff` | required for provenance `read-off` — `{by, image, method, crop}` (the rule — "record the digitizer and the crop") — "not recorded" is allowed where the research did not record it |
| `rechecked`, `verification` | whether the research re-read the source — what the research verification logs (01 ... 06) say, including every `[corrected]` note that changed a number |
| `replica`, `replica_args` | the protocol function in `mrm/validation/replicas.py` that reproduces the measurement, and its arguments |
| `tolerance`, `tolerance_basis` | fixed now (see below) and why |
| `identifies` / `tests` / `bounds` | what the calibration target may identify / what the held-out target tests / what the bound constrains |
| `data_points` | identifiers of the underlying published data (`dataset/item`) — used for the overlap rule |
| `uses` | calibration ids a held-out target takes as inputs (not as tested data) |
| `status` | `active` (scored), `deferred` (id reserved — its data are still calibration data — must list `blocked_by`), `unverified` (reported, not scored, until the value is confirmed), `cross_check` (solver against solver — reported, never counted as validation) |
| `fit` | calibration only — `false` for CAL-6, the H86 hypothesis value, which is never fitted |

Field names and tolerance keys are closed sets (`targets_io.OPTIONAL_TARGET_FIELDS`, `TOLERANCE_KEYS`) — a misspelt
field such as `use:` or `hardvalue:` is a schema error, not silently ignored.

Bias sign — `bias_v` is the **reverse** bias in volts, positive = reverse (the junction-LUT convention of
ARCHITECTURE 5.4). G25 documents print reverse bias as negative volts — "-1 V" is stored as 1.0.

### how are data kept separate?

1. Ids are unique across the four files.
2. No calibration data point is tested by an active, unverified or cross-check held-out entry. A deferred entry may
   share data only with the calibration ids it lists in `blocked_by` (VAL-6 and CAL-9).
3. No excluded data point appears in any other file.
4. `uses` and `blocked_by` name existing calibration ids. Validation and bound entries may share data with each
   other (VAL-4c and BND-4 rest on the same printed "< 1 dB" spread).

## what makes a prediction pass?

| `tolerance.kind` | Prediction | Pass rule |
|---|---|---|
| `abs` | same shape as `value` (scalar, the curve's y list, [lo, hi] for an interval, a mapping for a set) | every \|prediction - target.value\| <= `tolerance.value` |
| `rel` | as `abs` | every \|prediction - target.value\| <= `tolerance.value` x \|target.value\| |
| `interval` | scalar or array (for example a sampled population range) | every element in [`lo`, `hi`] |
| `inequality` | scalar or array | every element satisfies `op value` — with `hard_value`, a miss between `value` and `hard_value` is `soft_fail` |
| `band` | [min, max] of the predicted signal (min <= max) | `lo - slack` <= min and max <= `hi + slack` — with `either_sign: true` the mirrored band [-hi, -lo] also passes (BND-3a — the slide's sign convention is unpublished) |
| `rms_abs` | parametric — a mapping of `t_s` (times from a cusp) and `y` (degC-equivalent values) over one period in periodic steady state — curve — y list | rms(prediction - target) <= `tolerance.value` |
| `contour_asymmetry` | a mapping with `below_peak_k` and `above_peak_k` lists | every offset within `abs` and the required side wider at every level |
| `none`, `pending` | - | not scored |

the report uses `sigma` for z scores, not acceptance. z is the largest absolute
residual divided by its point's sigma. a curve can have a sigma list, and a set
can have a sigma mapping. calibration uses sigmas in its fitting likelihood.
`evaluate` instead uses the declared tolerance for pass or fail. `max_abs_residual` is the size of the miss (0 when an
interval, band or inequality is met — the inequality margin is in `detail`). A NaN or inf prediction is a `fail`.

## which measurement does each replica reproduce?

`targets_io.REPLICA_NAMES` is the authoritative list. Each function takes the plant configuration (or realized
parameters) and the `Target`, applies `replica_args`, and returns a prediction shaped as in the table above.

| Replica | Targets | Protocol |
|---|---|---|
| `loaded_q_from_3db_width` | CAL-1 | low-power laser sweep, Q_L = lambda / FWHM of the through notch |
| `notch_er_vs_heater_v` | CAL-2, VAL-2a/b | laser sweep at each heater voltage, all-pass and add-drop topologies (the G25 transmit topology is ambiguous) |
| `heater_power_per_fsr` | CAL-3a/b | heater power and voltage for one FSR of tuning (median device) |
| `resonance_vs_heater_v` | CAL-4 | resonance shift relative to 0 V against heater voltage |
| `tuning_range_at_v` | VAL-1 | the same at 1.5 V |
| `resonance_shift_vs_bias` | CAL-5 | DC shift against reverse bias, at P_in -> 0 and at +4 dBm |
| `apparent_shift_fixed_code` | CAL-6 | ambient sweep at a fixed heater code with TCR feedback, per dlambda/dT hypothesis |
| `cold_resonance_offset` | CAL-7 | cold-cavity resonance offset from the operating line |
| `monitor_current_at_lock` | CAL-8 | drop-monitor current at the lock point during the aggressor |
| `stabilizer_code_vs_T` | CAL-9a/b/c, VAL-6 | closed-loop heater code over the aggressor, per driver-law hypothesis |
| `eo_bandwidth_at_bias` | CAL-10, VAL-3a/b | small-signal EO response (waveform mode) at a reverse bias |
| `pd_responsivity` | CAL-11 | sampled responsivity range |
| `laser_line_statistics` | CAL-12a-d | sampled static line error (max \|error\| for G26 — quartiles for the Guide population), max \|drift\| and max \|power drift\| over 24 h |
| `aggressor_trajectory` | CAL-13a/b | aggressor power waveform through the thermal ROM, compared in resonance units |
| `relOMA_map` | CAL-14, VAL-4a/b/c, VAL-5, BND-4 | `mode='oracle'` (or `setpoint_lock`) for the loop-closed map, `mode='fixed_code'` for the loop-open map |
| `mode_solve_published_geometry` | VAL-7a/b | `modes.py` on the de Cea et al. 2019 cross-section |
| `self_heating_time_constant` | VAL-7c | `thermal_fem.py`, absorption source in the waveguide core |
| `thermal_step_published_stack` | VAL-8a-e | `thermal_fem.py` and the Foster fit on the imec stack, with our own materials (02) |
| `adjacent_order_penalty` | BND-1 | distance of every other order to the nearest other TX line, in FWHM |
| `order_hop_margin` | BND-2 | the BND-2 inequality for the worst lane |
| `aggressor_feasibility` | BND-3a, BND-3b | best of the listed causal controllers on the aggressor — signed drop error band in % (slide 10), monitor deviation in uA (808 degC/s run) |
| `bistability_sweep` | (none — the design review lists it for verification) | slow heater sweeps up and down at power |

## re-freeze log

| Date | Files | Reason | Comparisons made before? |
|---|---|---|---|
| 2026-10-09 | validation.yaml, bounds.yaml | first freeze | none |
| 2026-10-09 | validation.yaml, bounds.yaml | review of the split — the required condition keys and sigma on every entry — read-off records with crop — VAL-4c dependence note — BND-1 windows corrected (all orders) — BND-2 margin 1.8-2.0 nm — BND-3 split into 3a/3b with `either_sign`. No validation value or tolerance changed | none (no replica existed) |

## how is a frozen target changed?

A change after any comparison has been made is contaminated — the report must say so. Procedure —

1. Edit the file and state the reason in the entry's `verification` field.
2. Replace the block between `<!-- frozen-targets:begin` and `<!-- frozen-targets:end -->` in `docs/VALIDATION.md`
   with the output of `python -m mrm.validation.targets_io header <YYYY-MM-DD>`.
3. Add a line to `docs/CHANGELOG.md` — date, ids changed, reason, and whether any calibration or controller result had
   already been compared with the old values.
4. Run `pytest tests/closure/test_targets_schema.py tests/verification/test_validation_isolation.py`.

Planned re-freezes — VAL-6 becomes active (and CAL-9 shrinks) once the DAC law and full scale are known (question 5 of
docs/QUESTIONS_FOR_LIGHTMATTER.md) —
VAL-7a becomes active once someone re-reads Table 1 of de Cea et al. 2019 and confirms or corrects n_g = 2.73.

## why does the split differ from the design review?

- **CAL-13a** uses the rectified sine T = 113.6 - 88.5 |sin(pi t / 130.7 ms)| (peak 2127, mean 1354 degC/s), not
  the review's parabolic segment with a linear +-2200 degC/s sawtooth rate — the 808 degC/s note of
  docs/research/01_lightmatter_public.md and its verification log item 5 corrected the shape after the review was
  written.
- **Split rows** — CAL-3 (a P_FSR, b V_FSR), CAL-9 (a swing, b lane offsets, c code range), CAL-12 (a-d), CAL-13
  (a slide-10 trajectory, b the 808 degC/s run), VAL-2 (a, b), VAL-3 (a, b), VAL-4 (a, b, c), VAL-7 (a, b, c),
  VAL-8 (a-e). Each sub-entry has one quantity, one tolerance and one replica call.
- **Added** — VAL-3b (bandwidth ratio of the one plotted device, which removes the population-versus-device mismatch of
  VAL-3a), VAL-8c and VAL-8d (imec's measured heater efficiencies without and with undercut), EXC-9 ... EXC-18 (so
  every public number the model could be tempted to fit has one home).
- **VAL-5** scores the contour offsets from the map's peak, not the absolute contour temperatures — the loop-open
  heater code is unpublished, so the replica aligns its peak to 69.6 degC and the peak is not a test.
- **VAL-6** is `deferred` (the review — "set when known") — its data are CAL-9 until the DAC law is known.
- **VAL-7a** (n_g = 2.73) is `unverified` — no research report confirms it and the paper could not be re-read here.
  VAL-7c resolves the review's question — gamma_th = 2.4 MHz is a rate 1/tau (0.42 us, matching the printed 0.5 us), not
  cycles per second (66 ns).
- **VAL-8e** (imec's two-pole Foster fit, 3.2/13.0 us) is `cross_check` — it is imec's FE simulation of a different
  device, so a match is solver against solver. Its 63 % point is 7.35 us (04 V5 prints 7.3), not 10 us.
- **CAL-12b** carries the quartiles of the Guide line-error population (median -2, interquartile -5.5 to +1.5 GHz —
  the textbook's read-off, not re-checked), with the printed +-15 GHz as the yield limit in `components`. The first
  version bounded the sampled sigma by 15 GHz, which reads a yield limit as a standard deviation (corrected at the
  review).
- **CAL-12d** (2.5 % laser power drift per 24 h) is sourced to [YOLE] through the textbook's authoring brief, like
  CAL-12c, and is `active` with `rechecked: false` (the first version said "no source" and marked it `unverified`).
- **BND-1** windows — the review's window A (1.32 to 1.48 THz) checks only the first order. Inside the 2.8 THz TX span the
  second order counts too, so only 1.32-1.34 and 1.46-1.48 THz of window A keep 2 FWHM (at 1.40 THz ring 0's second
  order sits on line 7). The replica must scan every order inside the span.
- **BND-3** is split — BND-3a (slide-10 error band, either sign) and BND-3b (808 degC/s run, +-0.5 uA). One tolerance
  could not express both criteria, and the slide does not define the sign of its error.
- **CAL-6** carries the corrected dlambda/dT evidence — 64.3 to 64.4 pm/K for the 400 x 200 nm strip at
  1.94e-4 /K, and the 66.3 pm/K ceiling holds only at the C-band 1.86e-4 /K (69.1 pm/K at 1.94e-4 /K).
- **EXC-4** records that k = 0.67 dB/degC^2 is a secant about 0.65 FWHM from the optimum, not a curvature (06 V5).

## which earlier fits must stay out of defaults and priors?

Some research-report fits used held-out data. Their results must not become preset defaults or priors —

- 03 key finding 5 fitted the coupling-dispersion slope s = 1.48 %/nm to all 8 notch depths, including the held-out
  1.4 and 1.5 V points (VAL-2). Use the femwell solver slope (1.3 to 1.5 %/nm, 05 t02) or refit on CAL-2 only.
- 06 section 4.3 and V5 matched Q and dlambda/dT to the loop-open contours, including the -16 dB contour of VAL-5.
  Its "Q 3600 to 4400 at 86 pm/K" must not be used as a Q prior.
- 06 M6 inverted the swing from both the -2.1 and the -3.0 dB levels (169 to 227 pm at Q 3500). The -3.0 dB contour
  is VAL-4 — the swing prior must come from CAL-14 (-2.1 dB) alone.
- The interim thermal model takes its poles from the imec family. Comparing the interim model, rather than
  the FEM pipeline, with VAL-8 is tautological.
