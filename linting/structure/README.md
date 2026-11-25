# stage 2 - function structure

can a reader find the purpose, physical quantities and explanation for each decision inside the function? this required stage inspects Python syntax and comment placement. stage 3 evaluates the meaning of those comments. [three-stage overview](../README.md)

## which functions are checked?

the default scope is every function definition in `mrm/plant/thermal.py` and `mrm/plant/optics.py`. this includes public and private functions, methods, properties, static methods, class methods, async functions and Numba kernels. decorators and visibility give no exemptions. nested definitions are inspected and produce a violation on the enclosing function.

the default file list lives in [rules.toml](rules.toml). adding a file there enrolls it in the default check and its regression test. explicit paths on the command line replace the default list for that run. each path must name an existing Python file. the checker reads source without importing it.

the naming rule applies to `def` and `async def`. variables, classes, arguments, serialized field names, import aliases and assignments are outside that rule. existing public names and Python hooks such as `__post_init__` can bind directly to a category-prefixed implementation. the alias shares the original callable or descriptor.

the separate [runtime annotation test](../tests/test_type_contracts.py) resolves annotations in optics, thermal and phase. phase retains its earlier annotation checks. the category and body rules apply to the two default modules.

## how should a function start?

use `category_subject` or `_category_subject` for a private implementation. the first word must exactly match a category in the registry. the subject starts with a lowercase letter. subsequent words use lowercase letters or digits separated by single underscores. use a verb in the subject when it clarifies the operation. the name carries the category. write a purpose comment that explains what the function achieves.

| prefix | choose it when the caller wants | example |
| --- | --- | --- |
| `physics_` | a physical quantity, response, bound or time update | `physics_linear_hold_lag_weights` |
| `solver_` | unknown fields or states satisfying a set of equations | `solver_exact_ring_fields_and_energy` |
| `cache_` | a saved calculation checked, reused or invalidated | `cache_matches_last_optical_evaluation` |
| `packing_` | a flat buffer built or viewed as structured arrays | `packing_view_self_heating_optical_inputs` |
| `validation_` | confirmation that data or a calculation satisfies a condition | `validation_thermal_step_array_shapes` |
| `assembly_` | a constructed model, layout, workspace or heat path | `assembly_thermal_system` |
| `io_` | a model artifact located, read, written or verified | `io_load_rom` |
| `helper_` | an index, selection, normalized array or other supporting operation | `helper_shared_or_world_row_index` |

choose the prefix from the result the caller needs. a Numba kernel with `cache=True` can still be a field solver. a function that hands back views into packed memory belongs to packing even if the views contain physical quantities. the registry also assigns exclusive subject verbs. for example, `helper_solve_ring_fields` fails because `solve` belongs to the solver category. to introduce a category, add its meaning and exclusive verbs to the registry and add a real example here.

## what causes a diagnostic?

| rule | condition checked |
| --- | --- |
| `syntax` | Python cannot parse the file. |
| `name` and `category` | the definition breaks the prefix or spelling rule, or its first subject word is reserved for another category. |
| `purpose` and `summary` | the contiguous comment block directly above the first decorator or definition lacks a purpose comment, or the function uses a docstring. |
| `signature` and `type` | a parameter or return annotation is absent, multiple parameters share a declaration line, or fixed tuple result types share a line. |
| `contract` and `local` | a parameter or result lacks a same-line `# name = symbol (units) - meaning - type storage` comment, or points to a definition above, below or elsewhere. use `(1)` for dimensionless quantities. for `None`, give the effect or failure condition. result comments can use a descriptive name, `return`, or the corresponding `return[index]` for a tuple item. |
| `storage`, `shape` and `axis` | a simple scalar or recognized precision-specific array annotation disagrees with its storage description, an array description lacks shape, length, capacity or broadcast information, or a symbolic axis in a shape, capacity or broadcast tuple lacks a local `# axes -` definition. |
| `nested` and `return` | a function contains another function, the file contains a lambda, or two values in a returned tuple start on the same line. |
| `path` and `branch` | a compound statement, alternative branch or early exit lacks an explanation at that path. a guard comment can explain its immediate return, raise, break or continue. other exits need their own explanation. |
| `choice` | a statement selects between expressions using `x if condition else y` without a local comment. |
| `math` | a calculation group reaches arithmetic or a recognized numerical call before its first explanation. a blank line or completed compound statement starts a new group. the numerical call list is in [check.py](check.py). |
| `block` | more than six consecutive statements in the same suite share one explanation. a new branch starts its own count. |
| `delegation` | a body consisting of a single return of a directly named function call lacks a comment explaining what the call supplies. |

a qualifying comment starts with `# ` and contains at least three alphabetic words. signature contracts, axis declarations, tooling directives, reader marks and bare category labels do not count. a header never satisfies an in-body check. the six-statement limit is a ceiling. explain a change of physical reasoning where it happens, even when two equations are adjacent.

## how should comments read?

use short single-line `#` comments in lowercase plain English. keep the case required by units, symbols, code identifiers and proper names. give each line one complete thought. use a hyphen for a prose label. keep Python punctuation and mathematical notation intact. leave personal reader annotations to the reader. controlled vocabulary, physical accuracy and sentence clarity are review criteria.

## how do i run the checker?

from the repository root, run the default scope or pass the path to one Python file.

```sh
.venv/bin/python -m linting.structure.check
.venv/bin/python -m linting.structure.check mrm/plant/thermal.py
```

a finding gives the file, line, rule and reason. the exit code is 0 for a clean run, 1 for findings and 2 for invalid input.
