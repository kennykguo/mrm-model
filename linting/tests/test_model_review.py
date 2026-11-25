# exercise review sequencing and scoring without sending source to a model during pytest.
import pytest

from linting.semantics import model_review as review


def step(line, evidence, judgment):
    return {"line": line, "question": "why does the guard settle the temperature?",
            "evidence_lines": [evidence], "judgment": judgment,
            "relationship": "" if judgment == "clear" else "the gain multiplies absorbed power",
            "comment": "" if judgment == "clear" else "# zero gain gives zero new optical rise this step."}


def test_verification_receives_context_only_after_a_candidate(tmp_path):
    source = tmp_path / "physics.py"
    source.write_text("# extra verification evidence.\ndef thermal_gain():\n    return 0.0\n")
    payloads = []
    def ask(model, payload):
        payloads.append(payload)
        return [step(12, 11, "vague")]
    excerpt = (11, ["# handle this gain.", "if gain == 0:"])
    result = review.review_function("stub", excerpt, [(source, "thermal_gain")], 12, ask)
    assert payloads[0] == {"pass": "isolated", "function": review.numbered(excerpt), "focus_line": 12}
    assert "extra verification evidence" in payloads[1]["model_context"][0]["source"]
    assert result[0]["evidence"] == ["# handle this gain."]


def test_adequate_explanation_does_not_trigger_context_read(tmp_path):
    excerpt = (5, ["# zero gain removes the new heat, so this temperature already solves the equation."])
    result = review.review_function("stub", excerpt, [(tmp_path / "absent.py", "missing")],
                                    5, lambda model, payload: [step(5, 5, "clear")])
    assert result[0]["judgment"] == "clear"


def test_missing_context_leaves_a_candidate_unverified():
    result = review.review_function("stub", (2, ["# handle zero gain."]), [], 2,
                                    lambda model, payload: [step(2, 2, "incorrect")])
    assert result[0]["judgment"] == "unverified"
    assert result[0]["comment"] == ""


def test_grounding_rejects_an_out_of_range_citation_or_missing_focus():
    excerpt = (10, ["# handle zero gain."])
    with pytest.raises(ValueError, match="quotation"):
        review.checked_steps([step(10, 500, "clear")], excerpt, [10])
    with pytest.raises(ValueError, match="requested"):
        review.checked_steps([], excerpt, [10])


def test_calibration_counts_missed_issues_and_needless_rewrites():
    decisions = iter(["clear", "vague", "vague", "incorrect", "incorrect"])
    def ask(model, payload):
        assert "expected" not in payload and "case" not in payload
        line = payload["focus_line"]
        return [step(line, line - 1, next(decisions))]
    result = review.calibration("stub", ask)
    assert result["labeled_issues"] == 2
    assert result["adequate_comments"] == 1
    assert result["missed_issues"] == 1
    assert result["needless_rewrites"] == 1
    assert result["exact_classifications"] == 1


def test_default_verification_uses_a_real_caller_or_callee(tmp_path):
    path = tmp_path / "physics.py"
    path.write_text("def parent():\n    return active()\ndef active():\n    return gain()\ndef gain():\n    return 1\n")
    assert review.local_references(path, "active") == [(path, "parent"), (path, "gain")]


def test_private_method_can_use_its_local_callers_and_callees(tmp_path):
    path = tmp_path / "model.py"
    path.write_text("class Ring:\n    def advance(self):\n        return self._rise()\n"
                    "    def _rise(self):\n        return gain()\ndef gain():\n    return 1\n")
    assert review.local_references(path, "Ring._rise") == [(path, "Ring.advance"), (path, "gain")]
