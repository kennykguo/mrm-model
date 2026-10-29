# optics input and output types

all authored functions in `mrm/plant/optics.py` declare parameter and return types.
each description sits on the same line as its parameter or returned type, without a direction prefix.
the comments state the data type, array dtype, and dimensions.
each parameter, returned tuple item, and return type position has its own line.
physical symbols and units remain in the same contract comments.

`N` is the ring count, `L` is the laser-line count, and backscatter has one coefficient per stored order.

| data | dtype | structure |
|---|---|---|
| complex workspace | complex128 | matrix with capacity for 8 rows and L lines |
| real workspace | float64 | matrix with capacity for 27 rows and L lines |
| normalized ring response | float64 | matrix with capacity for 8 rows and L lines |
| round-trip delays | float64 | vector with capacity for L entries |
| outputs per ring and line | float64 | array with shape (16, N, L) |
| outputs per ring | float64 | matrix with shape (6, N) |
| outputs per line | float64 | matrix with shape (3, L) |
| packed optical values | float64 | vector with mixed-unit parameter, cache, and workspace blocks |
| packed optical indices | int64 | vector with header fields, offsets, and bus ordering |

`BinaryDataLevels` describes the binary-data dictionary.
`p1` is required, while `phi1`, `a1`, and `eta1` are optional.
`RingResponse` and `BusResult` declare the types of their returned fields.
functions that return `None` update the buffers described in their parameter comments.

annotations provide a visible contract and editor support.
shape metadata alone does not validate an array at runtime.
the staged ring solvers and their row-input entry point call `_validate_ring_workspace_buffers` before accessing buffers.
that check rejects incorrect precision, rank, row capacity, and line capacity.
larger allocations and strided views remain valid.
existing Python entry points still normalize array-like inputs and validate physical ranges.

Numba retains lazy compilation and its existing error models.
see [Numba compilation rules](https://numba.readthedocs.io/en/stable/reference/jit-compilation.html) and [NumPy typing](https://numpy.org/doc/2.4/reference/typing.html) for the distinction between annotations and compiled types.
