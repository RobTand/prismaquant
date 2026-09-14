"""A scalar expert endpoint has neither whole-stack nor sample authority."""
import pytest

from prismaquant.allocator import require_no_research_exact_member_scalar
from prismaquant.tessera_campaign import (
    research_exact_member_scope, select_anchor_groups,
)
from tools.dispatch_tessera_campaign import MergeRefused, merge_payloads


def _selection(members):
    return {"groups": [{"key": "s:stack", "members": members}]}


def test_exact_member_requires_the_full_original_group():
    resolved = {"s:stack": ["stack.0.down", "stack.0.up"]}
    partial = _selection(["stack.0.down"])
    with pytest.raises(RuntimeError, match="has members"):
        select_anchor_groups(partial, resolved, where="--units")
    complete = _selection(resolved["s:stack"])
    assert select_anchor_groups(complete, resolved, where="--units") == ["s:stack"]
    scope = research_exact_member_scope(complete, resolved, "stack.0.down")
    assert scope == {"member": "stack.0.down", "full_group_key": "s:stack",
                     "full_group_size": 2, "purpose": "research_scalar_endpoint",
                     "allocator_payload": False, "stack_estimate": False}
    with pytest.raises(RuntimeError, match="outside selected group"):
        research_exact_member_scope(complete, resolved, "other")


def test_exact_member_rejects_unproven_sample_and_consumers():
    resolved = {"s:stack": ["stack.0.down", "stack.0.up"]}
    sampled = _selection(resolved["s:stack"])
    sampled["groups"][0]["sampled"] = ["stack.0.down"]
    with pytest.raises(RuntimeError, match="unsampled whole group"):
        research_exact_member_scope(sampled, resolved, "stack.0.down")
    scalar = {"provenance": {"research_exact_member_scope": {"member": "stack.0.down"}}}
    with pytest.raises(SystemExit, match="no allocator or stack-estimate licence"):
        require_no_research_exact_member_scalar(scalar)
    from prismaquant.tessera_campaign import SCHEMA
    with pytest.raises(MergeRefused, match="cannot be merged"):
        merge_payloads({"row": {"schema": SCHEMA, **scalar}}, census={}, capture_sha256="0" * 64)
