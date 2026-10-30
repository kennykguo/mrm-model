# verify visible type contracts and safe rejection of incompatible ring workspace buffers.
from __future__ import annotations

import ast
from pathlib import Path
import re

import numpy as np
import pytest

from mrm.plant import optics, phase, thermal


# every physics function declares same-line physical and type contracts, including nested helpers.
@pytest.mark.parametrize("module", [optics, phase, thermal], ids=["optics", "phase", "thermal"])
def test_all_optics_functions_declare_types(module):
    source = Path(module.__file__).read_text()
    lines = source.splitlines()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        args += [arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None]
        assert all(arg.annotation is not None for arg in args), node.name
        assert node.returns is not None, node.name
        assert len({arg.lineno for arg in args}) == len(args), node.name
        for arg in args:
            eval(ast.unparse(arg.annotation), vars(module))
            pattern = rf'# {re.escape(arg.arg)} = [^\n]+ - type '
            assert re.search(pattern, lines[arg.lineno - 1]), (node.name, arg.arg)

        def check_return_type(annotation):
            if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
                check_return_type(annotation.left)
                check_return_type(annotation.right)
            elif isinstance(annotation, ast.Subscript) and ast.unparse(annotation.value) == 'tuple':
                types = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
                assert len({item.lineno for item in types}) == len(types), node.name
                for item in types:
                    check_return_type(item)
            else:
                eval(ast.unparse(annotation), vars(module))
                assert re.search(r'# [\w\[\]]+ = [^\n]+ - type ', lines[annotation.lineno - 1]), node.name

        check_return_type(node.returns)
        assert ast.get_docstring(node) is None, node.name
        for statement in ast.walk(node):
            if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Tuple):
                assert len({value.lineno for value in statement.value.elts}) == len(statement.value.elts), node.name


@pytest.fixture
def workspace_buffers():
    complex_workspace, real_workspace, _ = optics.make_workspace(2)
    delays = np.full(2, 1e-12, dtype=np.float64)
    response = np.zeros((optics.RING_RESPONSE_ROW_COUNT, 2), dtype=np.float64)
    return (
        complex_workspace,
        real_workspace,
        delays,
        response,
    )


@pytest.mark.parametrize('position', range(4))
def test_workspace_validation_rejects_wrong_precision(workspace_buffers, position):
    buffers = list(workspace_buffers)
    dtype = np.complex64 if position == 0 else np.float32
    buffers[position] = buffers[position].astype(dtype)
    with pytest.raises(TypeError, match='storage'):
        optics._validate_ring_workspace_buffers(2, *buffers)


@pytest.mark.parametrize('position', range(4))
def test_workspace_validation_rejects_wrong_rank(workspace_buffers, position):
    buffers = list(workspace_buffers)
    buffers[position] = buffers[position][None] if position == 2 else buffers[position].ravel()
    with pytest.raises(ValueError, match='dimensional'):
        optics._validate_ring_workspace_buffers(2, *buffers)


@pytest.mark.parametrize('position', range(4))
def test_workspace_validation_rejects_short_line_capacity(workspace_buffers, position):
    buffers = list(workspace_buffers)
    buffers[position] = buffers[position][:1] if position == 2 else buffers[position][:, :1]
    with pytest.raises(ValueError, match='capacity'):
        optics._validate_ring_workspace_buffers(2, *buffers)


@pytest.mark.parametrize('position', [0, 1, 3])
def test_workspace_validation_rejects_missing_rows(workspace_buffers, position):
    buffers = list(workspace_buffers)
    buffers[position] = buffers[position][:-1]
    with pytest.raises(ValueError, match='rows'):
        optics._validate_ring_workspace_buffers(2, *buffers)


# valid strided views and reusable larger allocations keep working.
def test_workspace_validation_accepts_strided_capacity():
    complex_workspace, real_workspace, _ = optics.make_workspace(6)
    delays = np.full(6, 1e-12, dtype=np.float64)
    response = np.zeros((optics.RING_RESPONSE_ROW_COUNT, 6), dtype=np.float64)
    optics._validate_ring_workspace_buffers(
        2,
        complex_workspace[:, ::2],
        real_workspace[:, ::2],
        delays[::2],
        response[:, ::2],
    )


@pytest.mark.parametrize('kernel', [optics._exact_stages, optics._exact_stages_slope])
def test_staged_solver_checks_capacity_before_writing(workspace_buffers, kernel):
    complex_workspace, real_workspace, delays, response = workspace_buffers
    complex_workspace.fill(7 + 3j)
    real_workspace.fill(17)
    with pytest.raises(ValueError, match='capacity'):
        kernel(2, complex_workspace, real_workspace, delays, response[:, :1])
    np.testing.assert_array_equal(complex_workspace, 7 + 3j)
    np.testing.assert_array_equal(real_workspace, 17)


# the row entry point must check buffers before it populates workspace rows.
def test_ring_row_input_checks_capacity_before_writing(workspace_buffers):
    complex_workspace, real_workspace, delays, response = workspace_buffers
    complex_workspace.fill(7 + 3j)
    real_workspace.fill(17)
    with pytest.raises(ValueError, match='capacity'):
        optics.ring_rows_nb(
            np.zeros(2),
            np.zeros(2),
            np.full(2, 0.1),
            np.full(2, 0.02),
            np.full(2, 0.95),
            delays,
            np.zeros(1, dtype=np.complex128),
            False,
            3,
            True,
            complex_workspace,
            real_workspace,
            response[:, :1],
        )
    np.testing.assert_array_equal(complex_workspace, 7 + 3j)
    np.testing.assert_array_equal(real_workspace, 17)
