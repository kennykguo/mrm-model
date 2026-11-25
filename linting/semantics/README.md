# stage 3 - can a reader explain this function on its own?

use this standard while writing a comment and when reviewing it. assume the reader knows Python and equations. ask **what would they still have to work out for themselves?** model findings are advisory. [three-stage overview](../README.md)

## how does the review work?

1. read only the function's opening comments, decorators, signature and body. at each non-obvious equation, choice or early exit, ask why the next step follows.
2. point to the words in the function that answer that question. accept an adequate explanation wherever it appears in the function. if you had to supply a missing premise, identify the line where the reader needs it.
3. check the suspected gap against related model code. outside evidence can establish the physics. only words inside the function can explain it to the isolated reader.
4. report the source line, unanswered question, missing or incorrect relationship, and the shortest accurate single-line `#` comment. when writing the code, put that comment next to the decision and reread the function.

## what counts as a useful finding?

in `solve_ring_temperature`, **why can the solver return `t_other` when `g == 0`?** the model solves `T = T_other + g * P_abs(T)`. for finite absorbed power, zero gain removes the optical temperature rise on this step. the absorption callback still supplies the power and its temperature derivative at the known temperature.

the three calibration cases present the entire solver. only the comment before the guard changes.

| case | comment | judgment |
| --- | --- | --- |
| A | `# handle the zero gain case.` | **vague.** it names the condition and leaves the reason for returning unstated. |
| B | `# zero gain makes the optical temperature rise zero this step, so t_other already solves the equation.` | **clear.** it explains how the condition makes the returned temperature valid. accept it. |
| C | `# the ring absorbs no light.` | **incorrect.** absorbed power can be positive when the thermal gain is zero. |

## how do i run it?

from the repository root, print a function to review it yourself. supply a qualified name such as `Class.method` for a method. use `--case A`, `--case B` or `--case C` in place of the file and function to inspect a calibration variant.

```sh
.venv/bin/python -m linting.semantics.excerpt mrm/plant/selfheat.py solve_ring_temperature
```

to request a model review, use an authenticated Codex CLI. the first pass receives only the complete function and its local header. if it flags a gap, a fresh pass checks the candidate against up to two same-file callers or callees. supply `--verify-with FILE FUNCTION` to choose more relevant evidence.

```sh
.venv/bin/python -m linting.semantics.model_review mrm/plant/selfheat.py solve_ring_temperature \
  --verify-with mrm/plant/thermal.py physics_separate_known_heat_and_source_gain_for_observer
```

each pass runs in an empty directory with tools disabled. the runner attaches the cited source lines to its report and rejects responses that use tools. confirmed findings include the proposed comment. candidates with insufficient evidence appear under `unverified_lines`. a completed review exits 0, including when it has findings. an invocation or validation error exits 2. edits are left to the author. [reviewer prompt](reviewer_prompt.md)

## how is the reviewer calibrated?

```sh
.venv/bin/python -m linting.semantics.model_review --calibrate
```

the isolated pass receives the full solver and the guard line. the runner withholds the expected judgment, case label and other variants. it scores exact classifications, missed issues and needless rewrites. an unverified issue counts as missed. classifying an incorrect explanation as vague loses the exact classification even if the reviewer proposed a correction. use `--model NAME` to select another available model.

on 2026-11-25 the first scored run with `gpt-6-luna` accepted all three comments and missed both issues. after the prompt asked for the causal link and a counterexample to unsupported claims, the recorded run classified all three correctly. it missed 0 of 2 issues and proposed 0 rewrites of 1 adequate comment. measuring broader accuracy requires independently labeled examples.
