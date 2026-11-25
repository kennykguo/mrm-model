# check whether physics functions explain their inputs, outputs and decisions locally.
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import io
from pathlib import Path
import re
import tokenize
import tomllib

ROOT = Path(__file__).resolve().parents[2]
RULES = tomllib.loads(Path(__file__).with_name("rules.toml").read_text())
TARGETS = tuple(ROOT / path for path in RULES["scope"]["files"])
CATEGORIES = RULES["categories"]
MAX_STATEMENTS_PER_EXPLANATION = RULES["comments"]["max_statements_per_explanation"]
FUNCTION_NAME = re.compile(r"_?([a-z]+)_([a-z][a-z0-9]*(?:_[a-z0-9]+)*)")
VERB_CATEGORIES = {verb: category for category, spec in CATEGORIES.items() for verb in spec["exclusive_verbs"]}
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
SHAPE = re.compile(r"(?:shape|capacity at least|broadcast to)\s*\(([^)]*)\)")
AXIS = re.compile(r"\b[A-Z](?:_[a-z]+)?\b")
NUMERICAL_CALLS = {
    "abs", "cos", "diff", "dot", "einsum", "exp", "expm1", "interp", "log", "log1p",
    "matmul", "maximum", "minimum", "norm", "prod", "sin", "solve", "sqrt", "sum",
}


@dataclass(frozen=True)
class Finding:
    line: int
    rule: str
    reason: str


# exclude reader marks, tooling directives, category tags and signature contracts.
def meaningful(comment: str) -> bool:
    return (
        comment.startswith("# ")
        and len(re.findall(r"[A-Za-z]+", comment)) >= 3
        and not comment.startswith(("# type:", "# noqa", "# axes -", "# todo", "# TODO", "# (todo)"))
        and not re.fullmatch(r"# [a-z]+ function\.?", comment)
        and " - type " not in comment
        and not re.match(r"# (?:see |same as |as above|as below|similar to )", comment, re.I)
    )


def is_calculation(node):
    if isinstance(node, ast.BinOp):
        return True
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            return node.func.attr in NUMERICAL_CALLS
        if isinstance(node.func, ast.Name):
            return node.func.id in NUMERICAL_CALLS
    return False


class Checker:
    def __init__(self, source: str):
        self.lines = source.splitlines()
        self.comments = {
            token.start[0]: token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        }
        self.findings = []

    def report(self, line, rule, reason):
        self.findings.append(Finding(line, rule, reason))

    def header(self, node):
        line = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)]) - 1
        comments = []
        while line > 0 and self.lines[line - 1].strip().startswith("#"):
            comments.append(self.lines[line - 1].strip())
            line -= 1
        return comments

    def contract(self, node, name):
        comment = self.comments.get(node.lineno, "")
        if re.search(r"(?:defined|described|shown) (?:above|below|elsewhere)", comment, re.I):
            self.report(node.lineno, "local", f"define {name} here so its contract stands alone")
        key = re.escape(name)
        if name.startswith("return"):
            key = rf"(?:{key}|(?!return\b)[A-Za-z_]\w*)"
        pattern = rf"^# {key} = (?:None|.+?\([^)]*\).*) - .+ - type .+"
        if not re.fullmatch(pattern, comment):
            self.report(node.lineno, "contract", f"define {name} with symbol, units, meaning and storage type")
            return
        storage = comment.split(" - type ")[-1]
        annotation = node.annotation if isinstance(node, ast.arg) else node
        if isinstance(annotation, ast.Name):
            declared = annotation.id
            if declared in {"float", "int", "complex", "bool", "str"} and not re.match(rf"{declared}\b", storage):
                self.report(node.lineno, "storage", f"the storage description of {name} must agree with {declared}")
            precision = re.fullmatch(r"(Float64|Int64|Complex128|Bool)(?:Array|Vector)", declared)
            if precision and not re.search(rf"\b{precision[1]}\b", storage, re.I):
                self.report(node.lineno, "storage", f"the storage description of {name} must include {precision[1].lower()}")
        if re.search(r'ndarray|array-like|numpy array', storage, re.I):
            if not re.search(r'shape|capacity|length|broadcast|\([A-Z]\w*,|same size', comment, re.I):
                self.report(node.lineno, "shape", f"declare the axes or broadcasting rule for {name}")

    def return_contracts(self, node, name="return"):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # a union of tuples describes alternate sets of returned values.
            if any(isinstance(part, ast.Subscript) and ast.unparse(part.value) == "tuple"
                   for part in (node.left, node.right)):
                self.return_contracts(node.left)
                self.return_contracts(node.right)
                return
        if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "tuple" and isinstance(node.slice, ast.Tuple):
            items = node.slice.elts
            if not any(isinstance(item, ast.Constant) and item.value is Ellipsis for item in items):
                if len({item.lineno for item in items}) != len(items):
                    self.report(node.lineno, "signature", "declare each returned quantity on its own line")
                for index, item in enumerate(items):
                    self.contract(item, f"return[{index}]")
                return
        self.contract(node, name)

    def function(self, node):
        header = self.header(node)
        if not any(meaningful(comment) for comment in header):
            self.report(node.lineno, "purpose", "explain what this function does above its first decorator or definition")
        match = FUNCTION_NAME.fullmatch(node.name)
        if match is None or match[1] not in CATEGORIES:
            self.report(node.lineno, "name", "start with a registered category followed by a descriptive snake_case name")
        else:
            role, subject = match.groups()
            verb = subject.split("_", 1)[0]
            required_role = VERB_CATEGORIES.get(verb)
            if required_role is not None and required_role != role:
                self.report(node.lineno, "category", f"the verb {verb} belongs to {required_role}")
        if ast.get_docstring(node) is not None:
            self.report(node.lineno, "summary", "write the summary above the definition using # comments")
        args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        args.extend(arg for arg in (node.args.vararg, node.args.kwarg) if arg)
        if len({arg.lineno for arg in args}) != len(args):
            self.report(node.lineno, "signature", "put each parameter on its own line")
        for arg in args:
            if arg.annotation is None:
                self.report(arg.lineno, "type", f"annotate {arg.arg}")
            self.contract(arg, arg.arg)
        signature = "\n".join(self.lines[node.lineno - 1:node.body[0].lineno - 1])
        axes = {axis for shape in SHAPE.findall(signature) for axis in AXIS.findall(shape)}
        definitions = " ".join(comment for comment in header if comment.startswith("# axes -"))
        for axis in sorted(axes):
            if not re.search(rf"\b{axis} = [a-zA-Z]", definitions):
                self.report(node.lineno, "axis", f"define the array axis {axis} in this function's header")
        annotation = node.returns
        if annotation is None:
            self.report(node.lineno, "type", "annotate the return type")
        else:
            self.return_contracts(annotation)
        for statement in ast.walk(node):
            if statement is not node and isinstance(statement, FUNCTIONS):
                self.report(statement.lineno, "nested", "move the nested function to module or class scope")
            if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Tuple):
                items = statement.value.elts
                if len({item.lineno for item in items}) != len(items):
                    self.report(statement.lineno, "return", "write each returned value on a separate line")
        self.suite(node.body, node, main=True)

    # inspect each branch independently so a sibling branch cannot supply its explanation.
    def suite(self, statements, owner, main=False):
        previous = statements[0].lineno - 2
        run = 0
        explained = False
        for index, statement in enumerate(statements):
            line = statement.lineno
            suites = [value for _, value in ast.iter_fields(statement)
                      if isinstance(value, list) and value and isinstance(value[0], ast.stmt)]
            if isinstance(statement, (ast.Try, ast.TryStar)):
                suites.extend(handler.body for handler in statement.handlers)
            if isinstance(statement, ast.Match):
                suites.extend(case.body for case in statement.cases)
            header_end = min((suite[0].lineno - 1 for suite in suites), default=statement.end_lineno)
            if any(not self.lines[i - 1].strip() for i in range(previous + 1, line)):
                explained = False
            documented = any(meaningful(self.comments.get(i, "")) for i in range(previous + 1, header_end + 1))
            exit_path = isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue))
            early = exit_path and (not main or index < len(statements) - 1)
            primary = statements is getattr(owner, "body", None)
            if not main and index == 0 and primary and early:
                documented |= any(meaningful(self.comments.get(i, "")) for i in (owner.lineno - 1, owner.lineno))
            if not main and index == 0 and not primary:
                # elif is its own if statement. else, except, finally and case need a local rationale.
                if not documented:
                    self.report(line, "branch", "explain the outcome of this alternative path")
            if (suites or early) and not documented:
                self.report(line, "path", "explain the decision or why this exit is valid")
            if not suites and any(isinstance(n, ast.IfExp) for n in ast.walk(statement)) and not documented:
                self.report(line, "choice", "explain why the calculation selects between these inputs")
            if (isinstance(statement, ast.Return) and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Name) and not documented and not explained
                    and len(statements) == 1 and main):
                self.report(line, "delegation", "explain what result this call supplies to the caller")
            if documented:
                run = 0
                explained = True
            # arithmetic needs a rationale within the calculation, including short functions.
            if not suites and not explained and any(is_calculation(n) for n in ast.walk(statement)):
                self.report(line, "math", "explain what this calculation means within the function body")
                explained = True
            run += 1
            if run > MAX_STATEMENTS_PER_EXPLANATION:
                self.report(line, "block", "explain the next calculation or state change")
                run = 1
            for suite in suites:
                self.suite(suite, statement)
            previous = statement.end_lineno
            if suites:
                explained = False


def check_source(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [Finding(error.lineno or 1, "syntax", error.msg)]
    checker = Checker(source)
    for node in ast.walk(tree):
        if isinstance(node, FUNCTIONS):
            checker.function(node)
        elif isinstance(node, ast.Lambda):
            checker.report(node.lineno, "nested", "give this calculation a named module or class function")
    return sorted(checker.findings, key=lambda finding: finding.line)


def main():
    parser = argparse.ArgumentParser(description="Check function contracts, calculations and control flow for local explanations.")
    parser.add_argument("paths", nargs="*", type=Path, default=TARGETS)
    findings = 0
    for path in parser.parse_args().paths:
        if path.suffix != ".py" or not path.is_file():
            parser.error(f"expected an existing Python source file - {path}")
        try:
            with tokenize.open(path) as stream:
                source = stream.read()
        except (OSError, UnicodeError, SyntaxError) as error:
            parser.error(f"cannot read {path} - {error}")
        for finding in check_source(source):
            print(f"{path}:{finding.line}: {finding.rule} - {finding.reason}")
            findings += 1
    print(f"{findings} readability findings")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
