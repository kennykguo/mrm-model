# ask a small model to review one isolated function and verify only the gaps it identifies.
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess
import tempfile

from .excerpt import CASES, ROOT, case_excerpt, isolated_function

HERE = Path(__file__).resolve().parent
PROMPT = (HERE / "reviewer_prompt.md").read_text()
FIELDS = ("line", "question", "evidence_lines", "judgment", "relationship", "comment")
VERDICTS = ("clear", "vague", "incorrect", "unverified")
EXPECTED = {"A": "vague", "B": "clear", "C": "incorrect"}
PROPERTIES = {key: {"type": "string"} for key in FIELDS}
PROPERTIES.update(line={"type": "integer"}, evidence_lines={"type": "array", "items": {"type": "integer"}}, judgment={"type": "string", "enum": list(VERDICTS)})
STEP = {"type": "object", "properties": PROPERTIES, "required": list(FIELDS), "additionalProperties": False}
SCHEMA = {"type": "object", "properties": {"steps": {"type": "array", "items": STEP}},
          "required": ["steps"], "additionalProperties": False}


def numbered(excerpt):
    first, lines = excerpt
    return "\n".join(f"{first + index:5}  {text}" for index, text in enumerate(lines))


def ask_codex(model, payload):
    # an empty working directory and fresh session prevent repo instructions and prior answers from entering the evidence.
    with tempfile.TemporaryDirectory(prefix="function-review-") as directory:
        schema = Path(directory) / "schema.json"
        schema.write_text(json.dumps(SCHEMA))
        command = ["codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
                   "--skip-git-repo-check", "--sandbox", "read-only", "--json", "--model", model,
                   "--output-schema", str(schema), "-c", 'web_search="disabled"',
                   "-c", 'approval_policy="never"', "-c", 'model_reasoning_effort="medium"']
        for feature in ("shell_tool", "unified_exec", "multi_agent", "plugins", "apps", "image_generation", "view_image"):
            command.extend(["--disable", feature])
        result = subprocess.run(command + ["-"], cwd=directory, input=PROMPT + "\n" + json.dumps(payload),
                                text=True, capture_output=True, timeout=150)
        events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        if result.returncode:
            errors = [e.get("message") or e.get("error", {}).get("message") for e in events
                      if e.get("type") in ("error", "turn.failed")]
            raise ValueError(next((message for message in reversed(errors) if message), "Codex invocation failed"))
        messages = []
        for event in events:
            item = event.get("item", {})
            if item and item.get("type") not in ("agent_message", "reasoning"):
                raise ValueError("discarded a review that used a tool or outside context")
            if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                messages.append(item["text"])
        if not messages:
            raise ValueError("model returned no completed review")
        return json.loads(messages[-1])["steps"]


def checked_steps(steps, excerpt, requested=None, verifying=False):
    first, lines = excerpt
    seen = set()
    for step in steps:
        line = step["line"]
        judgment = step["judgment"]
        if not first <= line < first + len(lines) or line in seen:
            raise ValueError(f"invalid or duplicate source line {line}")
        seen.add(line)
        if judgment not in VERDICTS or (judgment == "unverified" and not verifying):
            raise ValueError(f"invalid {judgment} judgment for this pass")
        cited = step["evidence_lines"]
        if not step["question"].strip() or any(not first <= number < first + len(lines) for number in cited):
            raise ValueError(f"missing question or unsupported local quotation at line {line}")
        if judgment == "clear" and (not cited or step["comment"] or step["relationship"]):
            raise ValueError(f"a clear step needs local evidence and no rewrite at line {line}")
        if judgment in ("vague", "incorrect") and (not step["relationship"].strip()
                or not step["comment"].startswith("# ") or len(step["comment"].splitlines()) != 1):
            raise ValueError(f"missing relationship or single-line correction at line {line}")
    if requested is not None and seen != set(requested):
        raise ValueError("model did not review exactly the requested lines")
    return [dict(step, evidence=[lines[n - first].strip() for n in step["evidence_lines"]]) for step in steps]


def local_references(path, qualified_name):
    # direct same-module calls establish a checkable relationship without guessing other objects' types.
    tree = ast.parse(path.read_text())
    definitions = {}
    pending = [("", tree.body)]
    while pending:
        owner, body = pending.pop()
        for node in body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            name = f"{owner}.{node.name}" if owner else node.name
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions[name] = node
            else:
                pending.append((name, node.body))
    edges = {}
    for name, node in definitions.items():
        owner = name.rpartition(".")[0]
        targets = []
        for call in (n.func for n in ast.walk(node) if isinstance(n, ast.Call)):
            if isinstance(call, ast.Name):
                targets.append(call.id)
            elif (owner and isinstance(call, ast.Attribute) and isinstance(call.value, ast.Name)
                  and call.value.id in ("self", "cls")):
                targets.append(f"{owner}.{call.attr}")
        edges[name] = [other for other in targets if other in definitions and other != name]
    callers = [name for name, callees in edges.items() if qualified_name in callees]
    neighbors = list(dict.fromkeys(callers + edges.get(qualified_name, [])))
    return [(path, name) for name in neighbors[:2]]


def review_function(model, excerpt, references, focus=None, ask=ask_codex, target=None):
    payload = {"pass": "isolated", "function": numbered(excerpt)}
    if focus is not None:
        payload["focus_line"] = focus
    steps = checked_steps(ask(model, payload), excerpt, [focus] if focus is not None else None)
    candidates = [step for step in steps if step["judgment"] != "clear"]
    if not candidates:
        return steps
    # read the physical evidence only after recording what the isolated reader could infer.
    evidence = []
    if not references and target is not None:
        references = local_references(*target)
    for path, name in references:
        evidence.append({"function": name, "source": numbered(isolated_function(Path(path).read_text(), name))})
    if not evidence:
        return [dict(step, judgment="unverified", comment="") if step in candidates else step for step in steps]
    verification = dict(payload, **{"pass": "verification", "candidates": candidates, "model_context": evidence})
    verified = checked_steps(ask(model, verification), excerpt, [s["line"] for s in candidates], verifying=True)
    replacements = {s["line"]: s for s in verified}
    return [replacements.get(step["line"], step) for step in steps]


def calibration(model, ask=ask_codex):
    outcomes = []
    for case in CASES:
        excerpt = case_excerpt(case)
        guard = excerpt[0] + excerpt[1].index("    if g == 0.0:")
        # use the real heat-gain calculation, never the answer key, for verification.
        reference = (ROOT / "mrm/plant/thermal.py", "physics_separate_known_heat_and_source_gain_for_observer")
        step, = review_function(model, excerpt, [reference], guard, ask)
        outcomes.append({"case": case, "expected": EXPECTED[case], "observed": step["judgment"], "review": step})
    positives = [r for r in outcomes if r["expected"] != "clear"]
    negatives = [r for r in outcomes if r["expected"] == "clear"]
    return {"model": model, "cases": outcomes, "labeled_issues": len(positives), "adequate_comments": len(negatives),
            "missed_issues": sum(r["observed"] in ("clear", "unverified") for r in positives),
            "needless_rewrites": sum(r["observed"] in ("vague", "incorrect") for r in negatives),
            "exact_classifications": sum(r["observed"] == r["expected"] for r in outcomes),
            "unverified": sum(r["observed"] == "unverified" for r in outcomes)}


def main():
    parser = argparse.ArgumentParser(description="Advisory semantic review with a small Codex model.")
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("function", nargs="?")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--verify-with", nargs=2, action="append", default=[], metavar=("FILE", "FUNCTION"))
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    try:
        if args.calibrate:
            if args.path or args.function or args.verify_with:
                parser.error("calibration supplies its own function and verification context")
            report = calibration(args.model)
        else:
            if not args.path or not args.function:
                parser.error("supply the changed function's file and qualified name")
            excerpt = isolated_function(args.path.read_text(), args.function)
            steps = review_function(args.model, excerpt, args.verify_with, target=(args.path, args.function))
            report = {"file": str(args.path), "function": args.function, "model": args.model,
                      "findings": [s for s in steps if s["judgment"] in ("vague", "incorrect")],
                      "unverified_lines": [s["line"] for s in steps if s["judgment"] == "unverified"]}
        print(json.dumps(report, indent=2))
    except (ValueError, KeyError, TypeError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(2, f"review unavailable - {error}\n")


if __name__ == "__main__":
    main()
