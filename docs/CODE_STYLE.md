# code style - write for a human reader

The reader is an engineer with an EE background who knows the ring resonator (coupling, round trip, resonance,
FSR, Q) but not this code base. Every comment, name and file layout should help that reader. Write in short, plain
sentences (close to ASD-STE100: one idea per sentence, active voice, no chains of clauses).

## comments

- Use ASD-STE100 principles for explanatory prose - short sentences, one idea per sentence, active voice, and
  consistent technical terms. Prefer lowercase text. Use a dash instead of a colon or semicolon in prose.
  Preserve Python syntax, literal examples, paths, scientific notation, API names, and correct unit capitalization.
- Say **what** the code does and **why**, in words. Do not repeat what one line of code already says.
- Use `#` comments for explanations. Do not use triple-quoted docstrings for explanatory prose.
- Put one complete statement on each comment line. Do not wrap a statement onto another line.
- Keep each statement brief. Use ASD-STE100 principles, rather than claiming formal compliance with its vocabulary.
- Put purpose comments above the definition, before any decorators. Put parameter and return descriptions beside their types.
- Preserve the reader's annotations. Do not add questions, `##` headings, or `###` annotations for the reader.
- Give units for every physical quantity (K, W, m, s, V, A, pm, codes).
- For core physics functions, including private helpers, map actual input and output names to equation symbols and explicit units.
  Use `# code_name = physics_symbol (units) - brief meaning - type description` beside the matching parameter or return type.
  Omit direction prefixes. Keep each description on one line.
  Use `(1)` for dimensionless values. Preserve scientific case, such as `T`, `K`, `W`, `Hz`, and `kappa_i`.
  Use `return[0]`, `return[1]`, or a returned dictionary key when an output has no single local name.
  Identify mixed-unit packed arrays by their field definitions. Do not assign one unit to a mixed array.
  For generic helpers, state that output units follow the input units instead of inventing a physical unit.
- Check coverage against every function signature in the requested scope, including private helpers and methods.
  Each argument needs a definition. Each returned value or modified output buffer needs its units or field layout.
  A few selected examples do not establish coverage. Keep a per-function inventory and check it before reporting completion.
- Write equations in plain text and define symbols close to the equation.
  For example, `# g = g (K/W) - temperature response to absorbed power over this substep - type float scalar`.
- Do not cite internal planning codes such as "07 P0-9", "perf.md R2-1" or "ENGINE_BRIEF section 3". Say the rule
  itself. If a design document really holds the details, name the file and the section title once:
  "see docs/ENGINE.md, section 4 'What happens in one tick'". This applies to code comments, docstrings and the
  living docs (docs/*.md, README.md). The historical records in docs/handoff/ and docs/research/ keep their
  codes.
- Keep physics sources in a readable form (author and year, for example "de Cea 2019"). Keep the test and
  validation IDs (CAL-14, VAL-8), but always add a few words that say what they check.
- Do not keep history notes ("formerly", "the old site") unless they are needed to read saved data.
- Start each module with brief purpose and entry-point comments. Put long derivations in docs/.
- Start function comments with a brief purpose. Add input/output definitions and non-obvious constraints where useful.
- Declare the input and output type in each contract comment. Give the dtype and dimensions for NumPy arrays.
- Put each contract comment on the same line as its parameter or returned type. Give each tuple position its own type and comment.
- Put one input, output, parameter declaration, or returned tuple item on each line.
- Distinguish type annotations from runtime checks. Validate workspace dtype, dimensions, and capacity before array access.
- Runtime strings remain strings. Preserve CLI help and other behavior that formerly read explanatory docstrings.

## names

- Prefer verbosity over ambiguity in names, including constants and packed-array row indices.
- Name functions by their action and physical result. Prefer `evaluate_ring_optical_outputs` over `ring_lines_nb`.
- Start cache-management function names with `cache_`, followed by the action or stored quantity.
- Cache reuse preserves the equations. Identify an optional model approximation explicitly in its name or description.
- Keep storage layout and compilation details out of function names when they do not explain the task.
- Spell out the stored quantity in row constants and include `ROW` in the name.
- Spell out `CLOCKWISE`, `COUNTERCLOCKWISE`, `COMPLEX_WORKSPACE`, and `REAL_WORKSPACE` in row constants.
- Identify the independent variable in derivative rows, such as `ROUND_TRIP_PHASE_DERIVATIVE`.
- Name the mathematical quantity itself - a field propagation factor, a power fraction, a loss exponent, or a rate.
- Distinguish a complex field coefficient from its squared magnitude and from an attenuation coefficient per metre.
- Name derivatives by both quantities, such as `d_quarter_propagation_dphi` and `d_loss_fraction_da_rt`.
- Do not call a complex derivative a phase slope when it is the derivative with respect to phase.
- Use bounds, sums, ratios, conjugates, and reciprocals in names when those operations define the quantity.
- Use descriptive names - `ring_temp_k`, `heater_power_w`, `n_rings`, `lock_code`.
  Keep a conventional equation symbol only when its meaning is clear and a nearby comment gives its units.
  Expand names that can be confused with another quantity, even inside a small function.
- Put the unit at the end of the name: `_w`, `_k`, `_m`, `_s`, `_v`, `_a`, `_pm`, `_hz`, `_code`.
- Import modules by their own name (`from mrm.plant import optics`), not by a cryptic alias (`_op`), unless two
  names clash.
- Constants are UPPER_CASE and have a comment with their meaning and unit.
- Names of the compiled-kernel arrays (KParams, KState) come from one place, `mrm/sim/state_spec.py`. Rename them
  there, and the kernel, the variable map (`tools/gen_variable_map.py`) and the tests follow.
- **Do not rename** anything that a user types or that is written to disk: YAML and config keys, CLI commands and
  flags, experiment kinds, trace and metric names in saved files, golden-file keys. If such a name must change, keep
  the old name working as an alias (see `ENUM_ALIASES` in `mrm/params/schema.py`).

## layout of a file

Module comments, imports, constants, then the public functions and classes, then private helpers. Use brief section
comments for related code. Do not add decorative boxes or reader annotations.
Keep storage layouts, row indices, header offsets, cache fields, and their counts together before the functions.
Distinguish an index from the value stored at that index. Comments on storage fields give the stored value's units.

    # self-heating solve

Each package `__init__.py` lists its files in reading order with one comment per file.

## logic

A readability change must preserve results. Compare executable ASTs for comment-only edits, and run checks for
CLI help, source inspection, and any other behavior that reads documentation. When executable physics changes,
also compare fixed numerical scenarios with `tools/perf_compare.py`, run `mrm golden`, and compare the test suite
with its baseline. In numba code, preserve floating-point operation order, dtypes, and loops.
Report a suspected bug with evidence and fix it in a separate change.
