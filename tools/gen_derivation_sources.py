# index source definitions without importing or running the model.
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/variables/source-index.json"
GREEK = {name for name in "alpha beta gamma delta epsilon eta theta kappa lambda mu nu pi rho sigma tau phi psi omega Gamma Delta Phi Omega".split()}


def math_name(name):
    if name == "TWO_PI":
        return r"2\pi"
    if name == "C0":
        return "c"
    if name.startswith("Delta_"):
        quantity = math_name(name[6:])
        return r"\Delta " + quantity if quantity else None
    parts = name.split("_", 1)
    head = "\\" + parts[0] if parts[0] in GREEK else parts[0]
    if len(parts) == 1:
        return head if len(head) == 1 or head.startswith("\\") else None
    if not (len(head) == 1 or head.startswith("\\")):
        return None
    suffix = "\\" + parts[1] if parts[1] in GREEK else r"\mathrm{" + parts[1].replace("_", ",") + "}"
    return head + "_{" + suffix + "}"


def expression_tex(node, names, assignments=None, seen=frozenset()):
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if assignments and node.id in assignments and node.id not in seen:
            return expression_tex(assignments[node.id], names, assignments, seen | {node.id})
        return math_name(node.id) if node.id in {"TWO_PI", "C0"} else None
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return str(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = expression_tex(node.operand, names, assignments, seen)
        return ("-" if isinstance(node.op, ast.USub) else "+") + value if value else None
    if isinstance(node, ast.BinOp):
        left = expression_tex(node.left, names, assignments, seen)
        right = expression_tex(node.right, names, assignments, seen)
        if not left or not right:
            return None
        if isinstance(node.op, ast.Div):
            return rf"\frac{{{left}}}{{{right}}}"
        if isinstance(node.op, ast.Pow):
            return rf"\left({left}\right)^{{{right}}}"
        operators = {ast.Add: "+", ast.Sub: "-", ast.Mult: r"\,"}
        operator = operators.get(type(node.op))
        return rf"\left({left}{operator}{right}\right)" if operator else None
    if isinstance(node, ast.Call) and node.args:
        argument = expression_tex(node.args[0], names, assignments, seen)
        if not argument:
            return None
        name = dotted(node.func)
        if name in {"np.asarray", "float", "complex"}:
            return argument
        operation = name.rsplit(".", 1)[-1]
        if operation == "log1p":
            return rf"\ln\left(1+{argument}\right)"
        if operation == "expm1":
            return rf"\left(\exp\left({argument}\right)-1\right)"
        if operation == "sqrt":
            return rf"\sqrt{{{argument}}}"
        if operation in {"log", "exp", "sin", "cos", "tan"}:
            command = "ln" if operation == "log" else operation
            return rf"\{command}\left({argument}\right)"
    return None


def symbol_tex(text):
    try:
        node = ast.parse(text.replace("^", "**"), mode="eval").body
    except SyntaxError:
        return None
    names = {n.id: math_name(n.id) for n in ast.walk(node) if isinstance(n, ast.Name)}
    return expression_tex(node, {k: v for k, v in names.items() if v})


def contracts(comments):
    result = []
    pattern = r"^(input(?:/output)?|output) - ([\w\[\]]+)(?: = (.+?))? \(([^()]*)\)(?:\s*[-,].*|$)"
    for comment in comments:
        match = re.match(pattern, comment)
        if not match or not match[3] or match[3] == "None":
            continue
        direction, name, symbol, unit = match.groups()
        tex = symbol_tex(symbol)
        if tex:
            result.append({"direction": direction, "name": name, "symbol": tex,
                           "symbol_text": symbol, "unit": unit, "annotation": comment})
    return result


def leading_comments(lines, node):
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 2
    comments = []
    while start >= 0 and (not lines[start].strip() or lines[start].lstrip().startswith("#")):
        line = lines[start].strip().lstrip("#").strip()
        if line:
            comments.append(line)
        start -= 1
    return list(reversed(comments))


def function_contracts(lines, node, comments):
    # typed signatures keep each description beside its parameter or returned type.
    if node.returns is None:
        return contracts(comments)
    args = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    last_parameter_line = max((arg.end_lineno for arg in args), default=node.lineno)
    annotations = []
    for line_number in range(node.lineno, node.returns.end_lineno + 1):
        comment = lines[line_number - 1].partition("#")[2].strip()
        if not re.match(r"[\w\[\]]+ = .+ - type ", comment):
            continue
        direction = "input" if line_number <= last_parameter_line else "output"
        annotations.append(f"{direction} - {comment}")
    # a union can repeat the same output description in more than one tuple variant.
    return contracts(list(dict.fromkeys(annotations))) if annotations else contracts(comments)


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = dotted(node.value)
        return f"{left}.{node.attr}" if left else ""
    return ""


def source_record(path, lines, node):
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return {"path": path, "line": node.lineno, "end": node.end_lineno,
            "code": "\n".join(lines[start - 1:node.end_lineno])}


def build():
    records, imports, functions, constants = {}, {}, {}, {}
    paths = sorted((ROOT / "mrm").rglob("*.py"))
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        module = relative[:-3].replace("/", ".").removesuffix(".__init__")
        lines = path.read_text().splitlines()
        tree = ast.parse("\n".join(lines))
        aliases = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        imports[module] = aliases

        def visit(body, parents=()):
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = ".".join((*parents, node.name))
                    qualified = f"{module}.{name}"
                    if qualified in functions:
                        qualified = f"{qualified}@{node.lineno}"
                    comments = leading_comments(lines, node)
                    doc = ast.get_docstring(node)
                    summary = next((c for c in comments if not c.startswith(("input", "output", "|", "-", "desc", "todo"))), "")
                    record = {
                        "id": f"fn:{qualified}", "qualified": qualified, "name": node.name,
                        "module": module, "family": "function", "domain": (module.split(".") + ["core"])[1],
                        "meaning": summary or (doc.splitlines()[0] if doc else node.name.replace("_", " ").strip()),
                        "comments": comments, "signature": f"{node.name}({ast.unparse(node.args)})",
                        "source_ref": source_record(relative, lines, node),
                        "arguments": [a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)],
                        "returns": [], "calls": [], "record_refs": [], "sets": [], "contracts": function_contracts(lines, node, comments),
                    }
                    # nested definitions have their own records and call scopes.
                    stack = list(node.body)
                    string_refs, assignments, return_nodes = set(), {}, []
                    while stack:
                        inner = stack.pop()
                        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            continue
                        if isinstance(inner, ast.Call):
                            call = dotted(inner.func)
                            if call:
                                record["calls"].append(call)
                            if call.endswith(".set") and inner.args:
                                record["sets"].append(ast.unparse(inner.args[0]))
                        if isinstance(inner, ast.Return) and inner.value is not None:
                            record["returns"].append(ast.unparse(inner.value))
                            return_nodes.append(inner.value)
                        if isinstance(inner, ast.Assign) and len(inner.targets) == 1 and isinstance(inner.targets[0], ast.Name):
                            key = inner.targets[0].id
                            assignments[key] = inner.value if key not in assignments else None
                        if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                            string_refs.add(inner.value)
                        stack.extend(ast.iter_child_nodes(inner))
                    record["calls"] = sorted(set(record["calls"]))
                    record["returns"] = sorted(set(record["returns"]))
                    record["record_refs"] = sorted(string_refs)
                    input_names = {c["name"]: c["symbol"] for c in record["contracts"] if c["direction"] != "output"}
                    for contract in record["contracts"]:
                        if contract["direction"] != "output":
                            continue
                        expression = assignments.get(contract["name"])
                        if contract["name"] == "return" and len(return_nodes) == 1:
                            expression = return_nodes[0]
                        if expression is not None:
                            tex = expression_tex(expression, input_names, assignments)
                            if tex:
                                contract["formula_tex"] = contract["symbol"] + "=" + tex
                    records[record["id"]] = record
                    functions[qualified] = (module, parents, node)
                    visit(node.body, (*parents, node.name))
                elif isinstance(node, ast.ClassDef):
                    visit(node.body, (*parents, node.name))
                else:
                    nested = [child for child in ast.iter_child_nodes(node) if isinstance(child, ast.stmt)]
                    if nested:
                        visit(nested, parents)

        visit(tree.body)
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Name) or not target.id.isupper():
                    continue
                name = target.id
                qualified = f"{module}.{name}"
                comment = lines[node.end_lineno - 1].partition("#")[2].strip()
                record = {"id": f"const:{qualified}", "qualified": qualified, "name": name, "module": module,
                          "family": "code constant", "domain": (module.split(".") + ["core"])[1],
                          "meaning": comment or name.replace("_", " ").lower(),
                          "value": ast.unparse(node.value), "source_ref": source_record(relative, lines, node)}
                records[record["id"]] = record
                constants[qualified] = node.value

    edges = set()
    variables = json.loads((ROOT / "docs/variables/data.json").read_text())
    known = {n["id"] for n in variables["nodes"]}
    record_sources, record_locations = {}, {}
    preset = ROOT / "mrm/params/presets" / f"{variables['meta']['preset']}.yaml"
    preset_lines = preset.read_text().splitlines()
    stack = []
    for index, line in enumerate(preset_lines):
        match = re.match(r"^( *)([A-Za-z_][\w.-]*):(?:\s|$)", line)
        if not match:
            continue
        indent, key = len(match[1]), match[2]
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key))
        identity = "cfg." + ".".join(part for _, part in stack)
        if identity in known:
            end = index + 1
            while end < len(preset_lines):
                following = preset_lines[end]
                if following.strip() and not following.lstrip().startswith("#") and len(following) - len(following.lstrip()) <= indent:
                    break
                end += 1
            record_locations[identity] = {"path": preset.relative_to(ROOT).as_posix(), "line": index + 1,
                                          "end": end, "code": "\n".join(preset_lines[index:end])}
    registry_path = ROOT / "mrm/sim/state_spec.py"
    registry_lines = registry_path.read_text().splitlines()
    registry_entries = {}
    for call in ast.walk(ast.parse("\n".join(registry_lines))):
        if isinstance(call, ast.Call) and call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
            registry_entries.setdefault(call.args[0].value, []).append(call)
    for record in variables["nodes"]:
        if record["id"].startswith("cfg."):
            continue
        matches = registry_entries.get(record["id"].rsplit(".", 1)[-1], [])
        exact = [call for call in matches if len(call.args) > 1 and isinstance(call.args[1], ast.Constant) and call.args[1].value == record.get("meaning")]
        if len(exact) == 1 or len(matches) == 1:
            call = exact[0] if len(exact) == 1 else matches[0]
            record_locations[record["id"]] = source_record("mrm/sim/state_spec.py", registry_lines, call)

    for record in records.values():
        if record["family"] != "function":
            continue
        for target in record.pop("sets"):
            try:
                expression = ast.parse(target, mode="eval").body
            except SyntaxError:
                continue
            if isinstance(expression, ast.Name):
                expression = constants.get(f"{record['module']}.{expression.id}")
            if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
                identity = f"cfg.{expression.value}"
                if identity in known:
                    record_sources.setdefault(identity, []).append(record["id"])

    def resolve(name, module, parents):
        parts = name.split(".")
        for size in range(len(parents), -1, -1):
            candidate = ".".join((module, *parents[:size], name))
            if candidate in functions or candidate in constants:
                return candidate
        if parts[0] in imports[module]:
            return ".".join((imports[module][parts[0]], *parts[1:]))
        return name if name.startswith("mrm.") else ""

    for qualified, (module, parents, node) in functions.items():
        record = records[f"fn:{qualified}"]
        for call in record["calls"]:
            resolved = resolve(call, module, parents)
            if resolved in functions and resolved != qualified:
                for candidate in functions:
                    if candidate == resolved or candidate.startswith(resolved + "@"):
                        edges.add((f"fn:{candidate}", record["id"], "call"))
        for string in record.pop("record_refs"):
            if f"cfg.{string}" in known:
                edges.add((f"cfg.{string}", record["id"], "source reference"))
        for name in {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}:
            resolved = resolve(name, module, parents)
            if resolved in constants:
                edges.add((f"const:{resolved}", record["id"], "code constant"))

    for qualified, expr in constants.items():
        module = qualified.rsplit(".", 1)[0]
        for name in {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}:
            resolved = resolve(name, module, ())
            if resolved in constants and resolved != qualified:
                edges.add((f"const:{resolved}", f"const:{qualified}", "code constant"))

    # Attach registered calculations to their source functions without treating calls as equations.
    for record in variables["nodes"]:
        for writer in record.get("updated_by", []) if isinstance(record.get("updated_by"), list) else []:
            candidates = [q for q in functions if q == writer or q.endswith(f".{writer}")]
            if len(candidates) == 1:
                edges.add((f"fn:{candidates[0]}", record["id"], "written by"))
                record_sources.setdefault(record["id"], []).append(f"fn:{candidates[0]}")

    quantities = []
    for record in list(records.values()):
        if record["family"] != "function":
            continue
        inputs = []
        for contract in record.pop("contracts"):
            identity = f"contract:{record['qualified']}:{contract['name']}"
            output = contract["direction"] == "output"
            description = " - ".join(contract["annotation"].split(" - ")[2:]).strip()
            quantity = {"id": identity, "family": "function output" if output else "function input",
                        "name": description or record["meaning"], "physics": True,
                        "meaning": description or record["meaning"], "annotation": contract["annotation"],
                        "symbol": contract["symbol"], "symbol_text": contract["symbol_text"],
                        "unit": contract["unit"], "qualified": record["qualified"],
                        "source_ids": [record["id"]], "aliases": [contract["name"], contract["symbol_text"], record["name"]],
                        "scope": "notation declared in this function’s comments; argument values are supplied by callers"}
            if contract.get("formula_tex"):
                quantity["formula_tex"] = contract["formula_tex"]
                quantity["scope"] += "; this equation is rendered from the supported scalar source expression"
            quantities.append(quantity)
            if output:
                for input_id in inputs:
                    edges.add((input_id, identity, "function input"))
            else:
                inputs.append(identity)
    for quantity in quantities:
        records[quantity["id"]] = quantity

    return {"meta": {"files": len(paths), "functions": len(functions), "constants": len(constants),
                     "contracts": len(quantities),
                     "note": "static source definitions and potential direct calls, not a dynamic execution trace"},
            "nodes": list(records.values()), "edges": sorted(edges),
            "record_sources": record_sources, "record_locations": record_locations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(build(), ensure_ascii=False, separators=(",", ":")) + "\n"
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            raise SystemExit("source-index.json is stale - run python3 tools/gen_derivation_sources.py")
        print("source index is current")
    else:
        OUTPUT.write_text(text)
        print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
