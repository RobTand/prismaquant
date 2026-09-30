"""Ambiguous JSON cannot establish a negative live-binding proof."""
from __future__ import annotations

import json

import pytest

from prismaquant import stage_a_retirement as retirement


@pytest.mark.parametrize("duplicate", ["path", "binding"])
def test_duplicate_member_cannot_hide_a_retirement_binding(tmp_path, duplicate):
    root = tmp_path / "bindings"
    root.mkdir()
    space = tmp_path / "retired"
    space.mkdir()
    live = json.dumps(str(space / "checkpoints" / "boundary-2"))
    if duplicate == "path":
        raw = '{"path": ' + live + ', "path": "/another/run"}'
    else:
        raw = ('{"binding": {"path": ' + live
               + '}, "binding": {"path": "/another/run"}}')
    document = root / "consumer.json"
    document.write_text(raw)

    with pytest.raises(retirement.RetirementRefused, match="binding.*duplicate") as refused:
        retirement.check_bindings([root], space=space, digests={})
    assert str(document) in str(refused.value)
    assert document.read_text() == raw
    assert not retirement.retirement_record_path(space).exists()
