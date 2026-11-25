# print one function with its own context for a semantic review.
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import tokenize

ROOT = Path(__file__).resolve().parents[2]
SOLVER_PATH = ROOT / "mrm/plant/selfheat.py"
CASES = {
    "A": "# handle the zero gain case.",
    "B": "# zero gain makes the optical temperature rise zero this step, so t_other already solves the equation.",
    "C": "# the ring absorbs no light.",
}


def isolated_function(source, qualified_name):
    lines = source.splitlines()
    body = ast.parse(source).body
    parts = qualified_name.split(".")
    for index, name in enumerate(parts):
        expected = (ast.FunctionDef, ast.AsyncFunctionDef) if index == len(parts) - 1 else (ast.ClassDef,)
        matches = [node for node in body if isinstance(node, expected) and node.name == name]
        if len(matches) != 1:
            raise ValueError(f"expected one definition of {qualified_name}, found {len(matches)} at {name}")
        node = matches[0]
        body = node.body
    first = min([node.lineno, *(item.lineno for item in node.decorator_list)])
    opening = lines[first - 1]
    indent = opening[:len(opening) - len(opening.lstrip())]
    while first > 1 and lines[first - 2].startswith(indent + "#"):
        first -= 1
    return first, lines[first - 1:node.end_lineno]


def case_excerpt(case):
    first, lines = isolated_function(SOLVER_PATH.read_text(), "solve_ring_temperature")
    guards = [i for i, line in enumerate(lines) if line.strip() == "if g == 0.0:"]
    if len(guards) != 1 or guards[0] == 0 or not lines[guards[0] - 1].strip().startswith("#"):
        raise ValueError("update the calibration fixture to match the zero-gain branch")
    index = guards[0] - 1
    indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
    lines[index] = indent + CASES[case]
    return first, lines


def main():
    parser = argparse.ArgumentParser(description="Read one function in isolation or review one blind calibration case.")
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("function", nargs="?", help="function name or Class.method")
    parser.add_argument("--case", choices=CASES, help="print one zero-gain calibration case")
    args = parser.parse_args()
    if args.case and (args.path or args.function):
        parser.error("select either a calibration case or a file and function")
    try:
        if args.case:
            first, lines = case_excerpt(args.case)
        else:
            if not args.path or not args.function:
                parser.error("supply a file and function name or select --case")
            with tokenize.open(args.path) as stream:
                first, lines = isolated_function(stream.read(), args.function)
    except (OSError, SyntaxError, ValueError) as error:
        parser.error(str(error))
    for number, line in enumerate(lines, start=first):
        print(f"{number:5}  {line}")


if __name__ == "__main__":
    main()
