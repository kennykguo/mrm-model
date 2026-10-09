"""Configuration snapshots and variants must not mutate their source model."""
import copy

from ringlab.config import load


def test_resolved_snapshot_has_no_mutable_controller_or_evidence_aliases():
    original = load({"controller": {"name": "step", "parameters": {"initial_code": 2660}},
                     "evidence": {"ring.group_index": {"status": "assumed", "source": "test"}}})
    before = copy.deepcopy(original.resolved())
    snapshot = original.resolved()
    snapshot["controller"]["parameters"]["initial_code"] = 100
    snapshot["evidence"]["ring.group_index"]["source"] = "changed"
    snapshot["ring"]["group_index"] = 8
    assert original.resolved() == before


def test_controller_variants_leave_base_and_repeated_digests_unchanged():
    original = load({"controller": {"name": "step", "parameters": {"initial_code": 2660}}})
    before = original.digest()
    overrides = {"controller.name": "step", "controller.parameters": {"initial_code": 2500}}
    first = load(original, overrides)
    hold = load(original, {"controller.name": "hold", "controller.parameters": {"code": 2400}})
    second = load(original, overrides)
    assert original.digest() == before
    assert first.digest() == second.digest()
    assert first.controller["name"] == "step"
    assert hold.controller["name"] == "hold"
    overrides["controller.parameters"]["initial_code"] = 100
    assert first.controller["parameters"]["initial_code"] == 2500
