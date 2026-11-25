# check annotations that must resolve in the numerical modules at runtime.
from __future__ import annotations

import ast
from pathlib import Path
import re

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
