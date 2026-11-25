# how is code readability checked?

the first two stages run on every push and pull request. the third helps a reviewer decide whether a reader can understand one changed function on its own.

| stage | question | scope and result |
| --- | --- | --- |
| 1 · Python errors | is a name undefined or an import unused? | **required.** Ruff scans repository Python source and stub files. each finding gives a file, line and rule. [Ruff configuration](stage_1_ruff/ruff.toml) |
| 2 · function structure | can i find the purpose, the inputs and results, and a comment where the reasoning changes? | **required.** the checker inspects every definition in `optics.py` and `thermal.py`. it checks category prefixes, same-line contracts, units, types, shapes, local axis definitions, and comments at calculations, branches and early exits. [rule reference](stage_2_structure/README.md) |
| 3 · meaning | what would i still have to work out to understand why this step follows? | **advisory.** a model reads one selected function in isolation. it consults related code to verify a suspected gap and proposes a one-line comment. a human decides which edits to make. [review instructions and examples](stage_3_semantics/README.md) |

for example, `# handle the zero gain case.` can satisfy the placement rule at an early return. the reader still needs to know why iteration can stop. in the self-heating equation, zero gain makes the new optical temperature rise zero, so the known temperature already solves the step. stage 3 asks whether the comment gives that reason.

## what should i run?

from the repository root, install the [development dependencies](../README.md#how-do-i-run-the-model), then run the same command as CI.

```sh
.venv/bin/python linting/required.py
```

the runner executes both required stages even when one fails. the exit code is 0 only when both pass. [CI workflow](../.github/workflows/lint.yml)

| task | command from the repository root |
| --- | --- |
| count Ruff findings by rule | `.venv/bin/python -m ruff check . --statistics` |
| check the configured physics modules | `.venv/bin/python -m linting.stage_2_structure.check` |
| read one function with source line numbers | `.venv/bin/python -m linting.stage_3_semantics.excerpt mrm/plant/selfheat.py solve_ring_temperature` |
| request an advisory review of that function | `.venv/bin/python -m linting.stage_3_semantics.model_review mrm/plant/selfheat.py solve_ring_temperature` |
| score the three calibration examples | `.venv/bin/python -m linting.stage_3_semantics.model_review --calibrate` |
| test the linting system locally | `.venv/bin/python -m pytest -q linting/tests` |

only the live review and calibration commands need an authenticated Codex CLI. the tests exercise the model runner with local fixtures.

## what is the current status?

on 2026-11-25 all 57 linting tests passed. the required repository check still fails at stage 1.

| stage | result |
| --- | --- |
| Python errors | **472 open findings** across the repository. Ruff reports 46 unused imports (`F401`) and 426 undefined names (`F821`). 420 of the undefined-name findings refer to names registered dynamically in `mrm/sim/kernel.py`. both physics modules in stage 2 pass Ruff. |
| function structure | **0 findings across 151 definitions** in the configured modules. optics has 56 functions and methods. thermal has 95. the separate runtime annotation tests also cover phase. |
| meaning | the last recorded calibration classified all three versions of the zero-gain comment correctly. it missed 0 of 2 issues and proposed 0 rewrites of 1 adequate comment. the same three examples were used to tune the prompt. independent calibration remains open. |

## where does each part live?

- **stage 1** - the two Ruff rules, Python file types and exclusions live in [ruff.toml](stage_1_ruff/ruff.toml). Ruff also respects repository ignore files and inline suppressions. archives and generated run outputs are excluded.
- **stage 2** - edit the default file list, category registry and statement limit in [rules.toml](stage_2_structure/rules.toml). the [checker](stage_2_structure/check.py) implements the diagnostics.
- **stage 3** - the [excerpt tool](stage_3_semantics/excerpt.py) extracts a function and supplies the three calibration variants. the [model runner](stage_3_semantics/model_review.py) applies the [review prompt](stage_3_semantics/reviewer_prompt.md).
- **shared** - [required.py](required.py) runs stages 1 and 2. [tests](tests/) covers the required runner, structural rules, runtime annotations, and semantic review examples.
