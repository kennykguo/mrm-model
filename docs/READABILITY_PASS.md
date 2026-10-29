# readability pass

## what is the goal?

make every Python file under `mrm/` readable to an EE engineer who knows ODEs but is new to this codebase.
include `optics.py` and preserve the user's recent improvements. work sequentially, one file at a time.
do not use subagents. the earlier instruction to launch an agent for every file is superseded.

## how should the code read?

- use ASD-STE100 principles — short sentences, one idea per sentence, active voice, and consistent terms.
- prefer lowercase prose. use dashes instead of colons or semicolons in explanatory text.
- preserve Python syntax, API names, paths, scientific symbols, and correct unit capitalization.
- give each file a short introduction with its purpose and main entry points. use questions when helpful.
- explain non-obvious reasoning, units, array dimensions, timing, assumptions, and helper outputs.
- replace ambiguous local names with descriptive names. keep defined math names in small equations.
- retain names used by source-inspection checks. the LaTeX checks evaluate the Q solver's local `f`.
- split dense expressions only when evaluation order, aliasing, exceptions, and results stay identical.
- preserve clear code. avoid whole-file formatting, repetitive comments, and excessive explanations.

## what must stay the same?

public signatures, imports and exports, config keys, CLI flags, saved fields, channels, metrics, registry names,
metadata strings, random streams, array layouts, dtypes, loops, formulas, and floating-point operation order.
compiled state names remain governed by `mrm/sim/state_spec.py`. report suspected bugs separately.

there are 94 Python files. the other 62 files contain configuration, reference prose, or scientific artifacts.
review their role separately. preserve parsed configuration values and keep serialized scientific data byte-identical.
`docs/CODE_STYLE.md` is the detailed guide. the user's preferences in this document take precedence.

## what is the procedure for each file?

1. read the whole file and its relevant callers or tests.
2. make a focused proposal against the frozen input. preserve any newer user edits when integrating it.
3. compile without imports. compare the executable AST, signatures, and literals with the input.
4. inspect every executable difference. allow only reviewed local renames or equivalent readable steps.
5. apply the file, record its result, and move to the next file. run relevant tests at coherent boundaries.

## what is complete?

94 of 94 Python files are reviewed and applied. executable checks pass after accounting for documented
local renames. the per-file list and hashes are in the manifest. completed files by package —

- calibration — 8 of 8 Python files — complete
- control — 22 of 22 Python files — complete
- core — 4 of 4 Python files — complete
- extraction — 7 of 7 Python files — complete
- params — 9 of 9 Python files — complete
- plant — 11 of 11 Python files — complete
- scenarios — 5 of 5 Python files — complete
- sim — 12 of 12 Python files — complete
- top-level — 4 of 4 Python files — complete
- validation — 7 of 7 Python files — complete
- waveform — 5 of 5 Python files — complete

all file reviews are complete. 62 of 62 non-code files are also reviewed.
comments in presets and the target README are clearer. parsed YAML values are unchanged.
all 37 JSON/NPZ artifacts are byte-identical. generated calibrated presets and frozen held-out
files retain their bytes. their role and structure are recorded separately.
newer user edits to the optics introduction found during the final audit are preserved.

## how will unchanged behavior be checked?

six fixed performance scenarios were saved before editing. compare the final signals with
`tools/perf_compare.py` and require zero numerical differences. baseline `mrm golden` matches all three cases.
compare final tests with the baseline, including failures and skips. do not update goldens to hide differences.

the baseline main suite finished — 2,029 passed, 7 failed, 119 skipped, 14 xfailed,
and 1 xpassed. it excluded the Python bridge file and included 12 C bridge cases. failures cover thermal target checks, the tutorial notebook, and generated variable-map records.
the isolated bridge suite now passes all 42 tests in both baseline and edited code. the earlier full run's
signal 13 did not recur in isolation.

focused plant, core, and control checks pass. their individual counts and logs are in the work folder.
parameter checks pass — 180 in the broad run, then both LaTeX checks after retaining the inspected local name.
scenario checks pass — 64 tests. runner block/start/reset checks pass — 16. registry checks pass — 26.
Gym and compiled-policy checks pass — 42. metrics pass — 7. vector recording/replay pass — 22.
reference-engine parity passes — 23. manifest and CLI checks pass — 19.
extraction checks pass — 121 tests. calibration machinery, target schema, and isolation pass — 111.
sensitivity and oracle checks pass — 19. waveform checks pass — 42. material checks pass — 42.
all 94 proposals pass executable AST checks. public interfaces and runtime literals are preserved.
the earlier private `world_summary.a` signature rename was reviewed separately with its caller.
applied hashes match live files after retaining the newer optics prose. no tests or golden files were changed.
all six numerical cases match exactly — 488 signals, zero differences, no missing arrays or shape changes.
all three golden cases match. the final broad suite finished. six documentation checks exposed examples or
file-layout details removed during condensation. the reward examples, nominal oracle range,
sensor/aggressor statement, recorded peak-seeking behavior, command examples, and run-directory names
are restored. all six follow-up checks pass. damaged documentation references now use plain code names.
the broad run recorded 2,013 passed, 13 failed, 117 skipped, 13 xfailed, and 2 xpassed. its six new
documentation failures used the earlier loaded prose. the revised prose passes all six checks in a fresh run.
the other seven failures match the baseline — three thermal targets, the tutorial notebook, and three
variable-map checks. no new failures remain after the focused rerun.

the comparison covers the same 2,158 test identities after excluding 12 C bridge cases from the baseline
main run. both bridge files pass separately — 42 tests in both versions. two previously skipped KaTeX
checks now pass. one machine-load-dependent performance xfail became an xpass. final executable,
signature, literal, configuration, artifact, live-hash, and whitespace checks pass. the review is complete.

## where is the resumable record?

`/tmp/mrm-readability-20261023/` holds the frozen baseline, original hashes, proposals, individual reports,
AST checks, numerical baselines, and test logs. `manifest.json` records which files are applied.
`agent-instructions.md` describes the original proposal boundaries — its delegation instructions are superseded.
update this document and the manifest after each file. do not restart completed work.

possible pre-existing issues are recorded in the thermal FEM, junction extraction, and experiment reports.
they are static observations, not verified fixes, and remain outside this pass.

## optics walkthrough follow-up

the reader knows lightmatter-capstone chapters 1–5, the two-direction field balance, port injections,
the implicit field derivative, and segment energy/loss integration. explain the physical problem first,
then its mathematical model, then the code. reuse established concepts and keep verification high-level.

continue in file order. check simulation usage and flag reference-only functions. use one best-supported
physical model as the target for cleanup. do not add physics or change parameter defaults during cleanup.

the retired, user-commented zero-backscatter helper and its dangling alias are removed. `ring_rows_nb`
now uses the existing staged exact solver for both zero and nonzero backscatter. imports work again.
the reciprocal and phase-decomposition helpers have brief clarifying comments and unchanged arithmetic.
the reciprocal and phase-decomposition helpers are covered. the four quarter-turn cases are angle
bookkeeping, not a limit of four resonances. TCMT is covered. the workspace helpers are next.
TCMT denominator names now distinguish detuning from decay. its arithmetic is unchanged, and 16 focused
optics checks pass. the steady-state TCMT kernel and comparison API are now in `archive/optics/tcmt.py`.
production accepts exact optics only. archive comparisons remain tested. bit-rate waveform TCMT remains active.
the cached thermal shortcuts remain to be assessed as the walkthrough reaches them. full consolidation is not complete.

67 optics and engine-adapter checks pass. four bus comparisons cover 3,540 values, with maximum
differences of 1.5e-13 relative to each field's scale for zero backscatter and zero for nonzero backscatter.
eight unit-ring comparisons also pass. the guide source compiles. snapshots and comparison results are
in `/tmp/mrm-optics-cleanup-20261025/`. 12 non-soak golden checks pass. the long soak was not rerun.

## what is the stricter comment pass?

the user requested a follow-up style pass after the completed readability review.
use concise, single-line `#` comments in place of explanatory docstrings.
each line holds a complete statement, with no wrapped prose.
prefer lowercase prose and dashes, while preserving code names, equations, and unit capitalization.
use ASD-STE100 principles and leave learning annotations to the reader.
preserve all existing reader annotations and process files sequentially without subagents.

core physics functions also define actual input and output names as `code_name = physics_symbol (units)`.
use `(1)` for dimensionless quantities and positional return entries for unnamed outputs.
check the equations before assigning units, especially field amplitudes, energies, Hz, and rad/s.
mixed-unit packed arrays refer to their field definitions instead of receiving a single unit.

the frozen input, sequential file manifest, proposals, and checks for this follow-up are in
`/tmp/mrm-comment-style-20261023/`. the prior readability baseline remains separate.
the follow-up is complete - 94 of 94 Python files are applied and checked.
2,859 explanatory docstrings now use source comments. runtime strings remain strings.
38 core physics functions define their actual input and output names, equation symbols, and units.
the existing reader annotations, including newer optics edits made during the pass, are preserved.

all 94 files compile and pass executable AST comparison. the reviewed executable differences preserve
nine CLI help strings as explicit literals and let scenario descriptions read source comments.
four existing test files now read comments instead of docstrings, with their numerical assertions retained.
literal code examples retain their syntax. prose uses dashes, and wrapped comment fragments are joined.

verification passed - 164 initial integration checks, 357 physics checks, and 80 scenario, CLI, and
documentation checks. the final eight comment contracts also pass after the wording review.
the physics checks cover optics, phase, implicit self-heating, thermal states, electrical behavior,
sensing, materials, optical waveforms, and compiled/reference engine parity.
the XML reports and per-file AST, annotation, CLI-help, and hash checks are in the follow-up work folder.

## what did the scatterer-helper follow-up correct?

the unit definitions above covered 38 selected functions, rather than every physics helper.
on 2026-10-25, the user identified missing definitions in `_interpolated_scatterer_amplitude`.
its comments now define both inputs, both returned names, their equation symbols, and their units.
they also define the per-order reflection mapping and interpolation weights.
the style guide explicitly includes private physics helpers. existing reader annotations are preserved.
the executable AST is unchanged, and all five focused optics checks pass.

## how was the remaining optics coverage checked?

the `_exact_line` omission exposed incomplete coverage after the single-helper correction.
the follow-up now inventories all 75 functions in `mrm/plant/optics.py`, including private helpers and methods.
every signature parameter has a name, symbol or role, and units or a typed field-layout reference.
every return value and modified output buffer has its units or field layout defined.
generic numerical helpers preserve caller units, and packed arrays identify their mixed-unit fields.
normalized power fractions, stored energy per input watt, absolute powers, and phase slopes are distinguished.

the signature-based coverage check passes for all 75 functions. all 53 optics tests pass.
the executable AST, function-body comments, and reader markers match the start of this follow-up.
the detailed inventory, contracts, snapshots, and checks are recorded in the work folder named by
`/tmp/mrm-optics-unit-audit-workdir`.
the style guide now requires a signature inventory before a pass is reported complete.

## steady-state TCMT archive - 25 october 2026

`archive/optics/tcmt.py` retains the unchanged TCMT kernel, derivatives, and comparison API.
production accepts exact optics only and rejects stale TCMT requests.
legacy config keys and compiled slots remain to preserve saved layouts.
the waveform time-domain model and energy-decay conversions remain active.
the guide and variable-map descriptions now distinguish these uses.
the next walkthrough section is the workspace helpers.

verification - 186 Python checks and 13 variable-map checks pass after correcting one edited test fixture.
this includes all 12 non-soak golden checks. the long soak was not rerun.
four bus comparisons cover 3,540 values and match the pre-archive snapshot exactly.
eight ring-row comparisons pass. the archived kernel and exact field-solver ASTs are unchanged.
the guide compiles, and generated source links and mathematical notation pass their checks.
one singular-ring test fails on both the pre-archive snapshot and the final code.
`test_degenerate_row_raises_on_both_paths` expects the general adapter to reject a perfectly lossless,
uncoupled ring at resonance. the general adapter does not raise the expected error.
this separate edge case was not changed during the archive work.
snapshots, numerical comparisons, and test logs are in `/tmp/mrm-tcmt-archive-20261025/`.

## physical quantity naming - 25 october 2026

follow the quantity through its defining expression before choosing its name.
separate field coefficients, power fractions, attenuation exponents, rates, and derivatives.
use derivative names that identify the numerator and the differentiation variable.
keep public keyword arguments, saved keys, packed layouts, and the reader's annotations.
work sequentially and record each reviewed rename by function.
verify each change by reversing only the identifier mapping and comparing executable ASTs.
then run the affected physics and engine checks.

work folder - the path in `/tmp/mrm-quantity-names-workdir`.
status - quantity names updated in 19 files, including optics, thermal, electrical, sensing, waveform, and control calculations.
the same physical quantities use consistent names in the scalar, staged, cached, and reference paths.
right-hand-side entries use `cw_derivative_rhs` and `ccw_derivative_rhs`.
the squared reflection coefficient uses `scatterer_reflectance`.
absorbed shares of intrinsic loss remain distinct from absorbed power fractions of the optical input.
public keyword arguments, saved keys, and packed layouts are preserved.
all 94 Python files compile and match their starting executable AST after undoing only identifier renames.
all 75 optics function contracts match their argument names.
reader annotations remain intact, and generated comments use the new identifiers.
verification complete - the broad physics, engine, extraction, parameter, and controller run passed 710 tests with 4 expected failures.
the focused final-revision run passed 104 tests with 1 expected failures.
source hashes match the checked files. no numerical equations, operation order, or runtime literals changed.
the naming rules and representative quantities are in [physical quantity names](QUANTITY_NAMES.md).

## workspace walkthrough - 25 october 2026

the workspace constants, make_workspace, and make_bus_outputs are covered.
cw means complex workspace and includes both propagation directions.
res stores the unit-input response. the two real result blocks hold binary data levels, not thermal branches.
_ring_has_backscatter and its compatibility alias have no callers in the current repository.
skip this unused helper in the walkthrough.
the COMPLEX_WORKSPACE_ROW_SCATTERER_FIELD_REFLECTION_COEFFICIENT comment now identifies a complex field amplitude rather than a power reflectance.
this comment edit preserves the executable AST.
continue at _exact_stages - derive its closed-form loop solve from the established two-direction balance.

## walkthrough pacing - 26 october 2026

skip a function when it only optimizes calculations already explained.
state the equivalent implementation and confirm whether any approximation is introduced.
continue to explain new physical assumptions, approximations, and numerical methods.
_exact_stages and _exact_stages_slope rearrange the established exact field and derivative calculations.
ring_rows_nb prepares those same quantities and has no new physics.
evaluate_ring_optical_outputs connects the unit response to actual input power, binary data averages, monitor power, and heat.
its phase-to-wavelength derivative uses the chain rule with input power and optical parameters held fixed.
dpabs_dlambda_res_nb repeats that absorbed-power and slope calculation without writing all outputs.
next new topic - the bounds used by the implicit thermal solver, starting at pabs_bounds_nb.
leave the user's parallel source readability changes untouched.

## optics storage names and layout - 26 october 2026

prefer explicit names over abbreviated field codes.
row, header, cache, and view names now state their storage role and the quantity stored there.
all 204 storage assignments in optics.py are grouped after imports and before the first function.
this includes output layouts, normalized responses, workspaces, packed headers, shared views, and caches.
`_as_unsigned_array_index` retains the original unsigned 64-bit cast.
all row positions, counts, array shapes, dtypes, and numerical expressions are preserved.
public view-field strings and recorded signal names are preserved.
references in code, tests, tools, and readable source descriptions use the explicit names.

work folder - the path in `/tmp/mrm-optics-row-names-workdir`.
the inverse AST check covers 203 renamed identifiers in 23 Python files.
it separately preserves a concurrent removal of the unused `_ring_has_backscatter` helper and its old alias.
reader annotations are preserved, including new annotations added during the pass.
verification complete - 233 focused tests passed, one expected failure remained, and one error-path test failed.
`test_degenerate_row_raises_on_both_paths` fails at the same assertion in an isolated copy of the starting source.
the focused run therefore has no new failures from the storage pass.
all workspace and output shapes, row counts, and dtypes match the original layout.
the source index was refreshed for the moved declarations and the explicit identifiers.

## optics type contracts - 26 october 2026

all 72 authored functions, including properties and the workspace validator, declare each parameter and return type.
input/output comments give the type, dtype, dimensions, physical symbol, and units.
parameters, returned tuple items, tuple type positions, and input/output comments use separate lines.
array aliases identify precision and rank. workspace aliases add their capacity contract.
the binary-data dictionary and both response records have explicit field types.

the staged ring solvers and ring_rows_nb validate workspace dtype, dimensions, and capacity before access.
larger buffers and strided views remain valid. no eager numba signatures or new dependencies were added.
the calculation bodies, defaults, argument order, and reader annotations match the starting source after removing only annotations and the new validation calls.

work folder - the path in `/tmp/mrm-optics-types-workdir`.
the 20 contract checks pass with JIT enabled and with JIT disabled.
the focused integration run passes 191 tests with one expected failure and the previously reproduced lossless-ring exception failure.
no new numerical failures were found.
benchmarks remain about 4.6 us for one level, 8.4 us for two levels, and 13.5 us with phase derivatives.
see [optics input and output types](OPTICS_TYPES.md) for the data layouts and validation boundary.

## optics comment placement - 27 october 2026

all 72 authored functions place each description on the same line as its parameter or returned type.
direction prefixes are removed. names, symbols, units, meanings, and type descriptions retain their wording.
tuple return annotations show each position separately. shared array aliases remain available.
493 identical repeated return comments above selfheat_views are reduced to one description per position.
the source index reads the signature comments and retains support for existing header comments elsewhere.

the executable AST and resolved type hints match the frozen source.
all descriptions and other comments, including user reading notes, are preserved.
the per-function manifest, snapshot, and checks are in the folder named by /tmp/mrm-optics-inline-contracts-workdir.

## teaching precision - 27 october 2026

start with the concrete question, physical situation, and purpose before introducing technical shorthand.
explain exactly what a bound permits the solver to do and what a failing bound means.
distinguish disabled physical effects, uncertain physical parameters, and approximate numerical safeguards.
never imply that backscatter-inclusive code means the current preset enables backscatter.
state finite numerical test coverage separately from a general mathematical guarantee.
explain line shapes in words before naming them or giving their equations.
the reader prioritizes understanding the accuracy tradeoff over hearing an optimization rationale.

## slope-bound walkthrough - 27 october 2026

pabs_bounds_nb is covered - peak power and curvature, including the tested backscatter limits.
pabs_slope_bound_nb finds the largest absolute Airy loss slope and adds a 10 percent margin.
explain the curve first, then its slope, the maximum-slope location, and each corresponding code block.
relate the temperature-slope bound to lagged cross-ring heat flow, separately from forward optical propagation.
the startup cross-ring gain check uses a separate Lorentzian estimate in realize._selfheat_lag_gain.
do not claim that this specific helper directly supplies that startup check.
next function - bus_finalize_nb and its forward-output and returned-reflection helpers.

## bus output walkthrough - 27 october 2026

bus_finalize_nb delegates to independent forward-output and returned-reflection calculations.
_bus_pout_nb is covered - conditional zero/one powers at the final bus output, downstream transmission, and mean bus power.
the reverse loop accumulates forward attenuation and is not a backward optical field solve.
conditional powers freeze the current thermal state while selecting one ring's voltage level.
independent data allows products of mean transfers to give the exact average within the single-pass model.
existing tests enumerate all two-ring binary combinations and check conditional outputs.
next function - _bus_refl_nb, including the two-crossing average and limits from omitted repeated reflections.

## returned-power walkthrough - 27 october 2026

_bus_refl_nb returned_power_w is covered through the saved returned-reflection output.
explain two crossings using power transmission fractions and average their square under the same-data-state assumption.
the original laser power and per-ring reflection fractions combine with a round-trip attenuation product.
waveguide segments enter twice. upstream rings enter through their averaged squared power transmission.
the implementation reconstructs this moment from the saved mean, zero-level, and one-level powers.
returned powers add without inter-ring field interference or repeated inter-ring reflections.
the same upstream data state on both crossings is a modeling assumption and not guaranteed at 50 Gb/s.
next - the final reflection_pair_sum and FORWARD_POWER_RELATIVE_ERROR_BOUND assignments in _bus_refl_nb.

## ring output function name - 27 october 2026

ring_lines_nb is now evaluate_ring_optical_outputs. all active call sites use the descriptive name.
the previous name remains an alias to the same compiled function.
the comments explain the response per input watt, binary data averages, actual powers, and totals across wavelengths.
the executable AST matches the frozen source after normalizing only the rename and alias.
the snapshot and change checks are in the folder named by /tmp/mrm-ring-optical-outputs-workdir.

## repeated-reflection estimate - 27 october 2026

_bus_refl_nb is covered through the final pair-sum and forward-power error estimate.
start from an extra forward-going path with two reflections, then derive the two-ring geometric field sum.
explain field reflection magnitudes versus reflected power fractions before forming pair products.
the all-pairs identity avoids listing each pair explicitly.
NRZ uses RMS reflection amplitudes and is a leading-order data-averaged estimate, not a universal data-pattern guarantee.
the reported estimate does not correct the optical outputs.
p1 is encoded in the saved mean power and is not an argument to _bus_refl_nb.
explain mathematics and its purpose before mapping expressions to code.
next function - bus_propagate_nb, which orchestrates the previously covered forward ring walk.

## variables and adapter walkthrough - 27 october 2026

define every symbol before its first use, with meaning and units.
numerical examples must trace quantities from model inputs or solved outputs through the reported result.
reflection power fractions come from solved reflected power divided by local arriving power.
bus_propagate_nb only orchestrates the already-covered forward walk and finalization.
skip its implementation and the packed-array view helpers, which add no physics.
_set_trial_temperature_phase is covered - reference temperature, quadratic shift, local phase conversion, and returned tuning slope.
its initial-ring power reset restarts the bus sweep; later inputs come from upstream commits.
next substantive function - absorbed_power_at_trial_temperature.

## trial heating and commit walkthrough - 28 october 2026

absorbed_power_at_trial_temperature is covered - update phases at a proposed endpoint temperature, evaluate absorbed power and resonance sensitivity, then convert that sensitivity to W/K.
its cached zero-backscatter branch is an equivalent calculation for zero coefficients and can be skipped as implementation optimization.
trial evaluation may modify working storage but does not advance physical time or accept the thermal endpoint.
commit_optics_at_solved_temperature saves the chosen ring outputs, forwards power through the next bus segment, and finalizes the last ring.
the derivative is used to improve temperature guesses and does not propagate heat.
next - selfheat_bounds and its phase-to-temperature bound conversion, skipping repeated bounds implementations.

## phase-to-temperature bound conversion - 28 october 2026

selfheat_bounds and the equivalent selfheat_bounds_general/sh_bounds conversion are covered.
dt_max_k is a temperature excursion in K, not a time step.
bound inputs retain all laser power through upstream rings and account for waveguide losses, within the single-pass model.
explain maximum phase sensitivity across lines and maximum tuning slope over the stated temperature interval.
temperature curvature has two contributions - bending of optical absorption with phase and bending of phase with temperature.
the reference general trial/commit helpers repeat prior calculations and are used for verification. skip their bodies.
following cache, rotation, and workspace routines should be skipped when they only optimize established equations.
flag the optional far-line hold separately because it introduces a real linear-response approximation.

## mathematical exposition and rendering - 28 october 2026

the user prefers explicit mathematical derivations over omitted chain-rule or inequality steps.
this supersedes the earlier preference to avoid expanding derivatives when the expansion explains the calculation.
use the order - purpose, variable definitions and units, assumptions, derivation, bound, numerical example, code mapping, accuracy limits.
define each symbol before its first equation and distinguish resonance wavelength from laser wavelength.
state which quantities remain fixed and which temperature interval a bound covers.
show the chain rule, absolute values, and bound substitutions before stating the final bound.
use simple standalone LaTeX displays and short aligned calculations.
avoid long underbraces, embedded prose labels, crowded displays, and unnecessarily large delimiters.
for multiple laser lines, retain per-line phase dependence where needed instead of implying one common phase with identical conversion.

## far-line hold - 28 october 2026

skip pure cache and phase-rotation implementations; the far-line hold is a genuine approximation and must be explained.
it is off by default and is implemented only on the zero-backscatter path.
_far_hold_entry checks the Airy denominator at both data levels, requiring at least 65 times its resonant minimum.
it caches response values and phase/amplitude partial derivatives at a hold point.
held evaluations use first-order Taylor response in phase and round-trip field transmission.
round-trip group delay scales normalized stored energy separately; incoming power and absorbed-loss shares are applied at current values.
reference phase drift and trial rotation are each limited to 5e-4 rad; amplitude drift is limited to 3e-5.
coupler changes or drift invalidate the cached point. excessive trial rotation disables hold and uses the full response.
worked single-level loss example uses t1=.95, t2=.99, a=.98, phi=1 rad, dphi=3e-4 rad, da=1e-5.
linear loss=0.004474315308764865; exact loss=0.0044743176361430655.
at 2 mW input and heat fraction .8, absorbed-power error is about 3.7 pW for this example only.
validation scope is separate from universal guarantees. the default stays off for accuracy-first work.
next - identify any remaining new concepts in response reuse and bound updates, then finish public wrappers without repeating known calculations.

## cache function names - 28 october 2026

cache-management functions now start with cache_.
the names cover coupler and loss factors, Airy refresh, saved response reads, output writes, matching, invalidation, and engine wavelength keys.
concurrent optics consolidation moved most helpers to archive/optics/retired_cached_approximations.py and removed the engine cache-refresh wrapper.
the archived implementation retains the new names and previous aliases. remaining production cache helpers retain the prefix.
cache reuse retains the equations. the optional far-line approximation remains separately identified.
the archived optics AST matches the frozen source after normalizing the function names and aliases.
the name manifest, snapshots, location inventory, and checks are in the folder named by /tmp/mrm-cache-function-names-workdir.

## one production optical path - 28 october 2026

the latest user direction supersedes the earlier plan to retain a separate exact Airy cache path.
every ring now uses the general two-direction field solver, including zero backscatter.
retired Airy caches, polynomial rotation, and far-line hold are preserved in
`archive/optics/retired_cached_approximations.py` solely for explicit numerical reference comparisons.
production does not import the archive.
Plant optics_far_hold, pack_selfheat_args far_hold, and ReferenceEngine airy selectors are removed.
the runner rejects a stale optics_far_hold option, and low-level stale far-line headers fail explicitly.
physical backscatter parameters and defaults are unchanged.
exact reuse of an identical final trial remains; it does not choose another model or approximation.
retired packed-layout slots remain reserved for compatibility.
all actual field, derivative, optical-output, and bound equations retain their executable ASTs.
a zero complex determinant now raises an explicit singular-field error.
136 Python checks and 13 variable-map checks pass, including all 12 non-soak goldens.
one existing expected failure remains, and two long-soak cases were not run.
1,800 numerical comparison values pass. nonzero-backscatter comparisons match exactly;
zero-backscatter absorbed-power differences stay below 1.9e-16 W.
two golden records changed only at rounding level and are documented in docs/CHANGELOG.md.
a warmed one-ring, eight-line NRZ heating call takes about 0.25 us in the archived specialization
and 1.70 us in the general path. this is an inner-call measurement, not a whole-run timing.
see docs/OPTICS_CONSOLIDATION.md and /tmp/mrm-optics-shortcut-retirement-20261028/validation.json.
the next walkthrough can skip the retired optimization helpers entirely.

## remaining optics interfaces - 28 october 2026

finish the file in code order after selfheat_bounds_general.
pass_through_power_to_next_ring repeats the established bus handoff and finalization.
cache match and clear only manage exact final-trial reuse.
pack_selfheat_args prepares float values and integer offsets, validates and broadcasts inputs, initializes reference parameters, and reserves outputs and workspaces.
its phase sensitivity uses the supplied laser wavelength and group delay. it does not solve temperature or advance time.
_evaluate_ring_input_array and _evaluate_ring_response wrap the established differentiated field solve.
RingResponse stores complex port fields, directional energies, intrinsic loss, and power derivatives.
BusResult stores powers rather than complex port fields. intrinsic loss and absorbed heating remain distinct.
ring_exact is a public single-ring interface used by validation replicas and tests, rather than the compiled thermal inner loop.
ring_scattering_matrix has no simulation caller, including through ring_smatrix. skip its body under the user's rule.
bus_propagate is used by the controller model and wraps the established bus calculation.
absorbed_power_and_resonance_slope has no simulation caller, including through dpabs_dlambda_res. skip its repeated body.
validation helpers and compatibility aliases introduce no physical model.
the optics.py physical calculations have now been covered. the next new topic is the safeguarded temperature root search in selfheat.py.

## phase walkthrough begins - 28 october 2026

start phase.py in code order after completing optics.py.
covered module purpose, imports, RP_* storage indices, PhaseParams and its setup methods, and bus_t2_from_plant_params.
PhaseParams prepares fixed parameters and voltage tables. it does not advance state.
reference resonance and intrinsic loss already include the junction at reference bias.
explain ring-level junction index and free-carrier loss differences relative to that bias before describing cached reference values.
distinguish table per-length quantities from ring-level outputs. defer their electrical derivations to electrical.py.
B counts simulated worlds, N rings, and G voltage-grid points. array dimensions are software layout.
bus attenuation follows dP/dz = -alpha_bus P and gives power transmission exp(-alpha_bus d).
this is a power fraction, not a field amplitude fraction.
the first segment is unity because input loss is handled upstream. downstream segments share the supplied spacing. the final segment is unity here.
existing reference-state checks verify zero shift, resonance phase, field attenuation, absorbed fraction, coupling, and FSR. no new tests were run for this explanation.
next function - thermal_shift_m, followed by local_thermo_optic. derive the integral with distinct T_ref and T_0 before mapping to code.

## equation-free visual atlas - 28 october 2026

read EXCALIDRAW.md and the new supplied clipboard drawing before creating the atlas.
preserve the supplied drawing unchanged as 00-ring-model-reference.excalidraw.
add 25 native editable diagrams covering the physical concepts discussed in this chat.
use Excalidraw font family 5 at size 36, dark sketch strokes, and spacious transparent shapes.
add no mathematical equations. label geometry, directions, graphs, and calculation flows in plain language.
keep physical schematics separate from solver and control-flow drawings.
provide individual native files, a combined atlas with 26 named frames, SVG/PNG previews, and an offline gallery.
the atlas is in docs/diagrams/optics and the layout generator is tools/build_excalidraw_atlas.py.
export previews through the official Excalidraw renderer with the reference font loaded.
inspect rendered layouts and check text bounds for collisions.
check all 26 gallery selections, local-file image loading, editable-file targets, enlargement, and mobile overflow.
curves remain schematic. illustrative measurement dots are not experimental data.
no production simulation code changed for this request.

## equation-only visual companion - 28 october 2026

create docs/diagrams/optics-math as a separate companion to the labelled concept drawings.
use short topic labels, rendered LaTeX equations, and symbol definitions with units.
include no explanatory paragraphs on the equation sheets.
cover topics 00 through 25 and add 17a for phase envelopes.
keep the full ring model, zero-backscatter limit, scalar thermal illustration, Lorentzian study model, and archived TCMT labelled separately.
include the finite-step gain, safeguarded trial candidates, held-power solution, linear input ramp, phase-to-temperature bounds, and statistical link model.
render 127 expressions on 27 sheets with MathJax. KaTeX also accepts every expression.
store vector SVG equation images in Excalidraw and preserve their LaTeX in customData, source.json, equations.tex, and individual latex files.
provide individual native sheets, a combined atlas, SVG/PNG previews, and an offline gallery.
check image assets, native frame contents, label and equation overlap, gallery navigation, and local-file loading.
no simulation code changed. creating equation sheets does not establish that the user has understood the next root-solver lesson.

## thermal tuning law and rebasing - 29 october 2026

covered calculate_thermal_resonance_shift_m and rebase_thermal_tuning_coefficients.
these retain thermal_shift_m and local_thermo_optic as aliases after the user's readability renames.
explain the temperature-to-resonance map with bias fixed before introducing code.
T_ref anchors the known resonance. T_0 anchors the specified tuning slope. neither is necessarily ambient or the previous temperature.
define s(T) = d lambda_r/dT and derive the linear-in-temperature slope model and its integrated quadratic shift.
use the integral between T_ref and T, retaining the lower-limit term.
illustrative example - T_0=300 K, T_ref=310 K, T=330 K, s_0=75 pm/K, beta=.001/K gives 1530 pm shift.
rebasing is exact algebra within the quadratic model, not a linear-only approximation or a time integration step.
at T_prev=330 K, s_local=77.25 pm/K and beta_local=.001/1.03 per K. a 2 K increment gives 154.65 pm, including .15 pm curvature contribution.
derive s_local*beta_local = s_0*beta to show curvature is retained.
engine uses the previous ring temperature as the trial-phase reference and passes these rebased coefficients to the optical adapter.
state the fitted-law limitation and the nonzero denominator assumption for rebasing.
existing phase test checks rebased shift differences and finite-difference slope. no tests rerun for this explanation.
next - evaluate_ring_resonance_shifts and its workspace equivalent. explain the voltage contribution and junction lookup before mapping code. skip only duplicate allocation wrapper mechanics.

## optics, phase, and thermal style - 29 october 2026

completed the optics audit fixes and applied the same style to phase.py and thermal.py sequentially.
all 151 authored functions and methods have typed same-line physical contracts, including nested helpers.
function names, workspace names, and phase column names state their physical role. prior API names remain aliases.
record fields, configuration keys, numeric layouts, arithmetic, defaults, and loop order remain unchanged.
comments use short lowercase statements with dashes. existing reader annotations remain intact.
detailed thermal equations and assumptions are retained in docs/THERMAL_IMPLEMENTATION_NOTES.md.
the normalized calculation AST matches the starting source in all three files.
272 physics checks pass with one expected failure and the separately reproduced legacy TCMT-requesting phase failure.
seven engine and self-heating integration checks pass.
see docs/PHYSICS_READABILITY.md for the complete function inventory.
work folder - the path in /tmp/mrm-physics-readability-workdir.
