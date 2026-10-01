"""A live absolute path alias still binds the Stage A namespace (Refs #1088)."""
from __future__ import annotations

import json

import pytest

from prismaquant import stage_a_retirement as retirement


@pytest.mark.parametrize("alias_kind", ["dot-root", "dot-child", "link-root", "link-child"])
def test_absolute_binding_alias_cannot_certify_no_live_consumer(tmp_path, alias_kind):
    space = tmp_path / "retired"
    space.mkdir()
    checkpoint = space / "checkpoints" / "boundary-2" / "checkpoint.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('{}')
    if alias_kind.startswith("dot"):
        transit = tmp_path / "transit"
        transit.mkdir()
        alias = transit / ".." / space.name
    else:
        alias = tmp_path / "linked-run"
        alias.symlink_to(space, target_is_directory=True)
    if alias_kind.endswith("child"):
        alias = alias / "checkpoints" / "boundary-2" / "checkpoint.json"
    binding = tmp_path / "consumer.json"
    payload = json.dumps({"checkpoint": {"path": str(alias)}})
    binding.write_text(payload)

    with pytest.raises(retirement.RetirementRefused, match="a live binding holds the run") as refused:
        retirement.check_bindings([binding], space=space, digests={})
    assert str(binding) in str(refused.value) and str(alias) in str(refused.value)
    assert checkpoint.read_text() == '{}'
    assert binding.read_text() == payload
    assert not retirement.retirement_record_path(space).exists()


def test_canonical_sibling_and_nonpath_string_are_unrelated(tmp_path):
    space = tmp_path / "retired"
    space.mkdir()
    outside = tmp_path / "retired-other"
    outside.mkdir()
    binding = tmp_path / "consumer.json"
    binding.write_text(json.dumps({"path": str(outside / "checkpoint.json"),
                                   "qname": "model.layers.0.attn.q_proj"}))
    assert retirement.check_bindings([binding], space=space, digests={}) == 1


def test_lexically_inside_binding_still_refuses_if_symlink_points_outside(tmp_path):
    space = tmp_path / "retired"
    space.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (space / "linked-out").symlink_to(outside, target_is_directory=True)
    binding = tmp_path / "consumer.json"
    binding.write_text(json.dumps({"path": str(space / "linked-out" / "checkpoint.json")}))
    with pytest.raises(retirement.RetirementRefused, match="a live binding holds the run"):
        retirement.check_bindings([binding], space=space, digests={})
