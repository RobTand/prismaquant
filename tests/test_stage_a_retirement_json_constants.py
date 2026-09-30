"""Non-standard JSON constants cannot establish a negative binding proof."""
from __future__ import annotations

import pytest

from prismaquant import stage_a_retirement as retirement


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_binding_scan_refuses_nonstandard_json_constants(tmp_path, constant):
    root = tmp_path / "bindings"
    root.mkdir()
    space = tmp_path / "retired"
    space.mkdir()
    document = root / "consumer.json"
    raw = '{"binding": ' + constant + '}'
    document.write_text(raw)

    with pytest.raises(
        retirement.RetirementRefused, match="binding.*non-standard JSON"
    ) as refused:
        retirement.check_bindings([root], space=space, digests={})
    assert str(document) in str(refused.value)
    assert document.read_text() == raw
    assert not retirement.retirement_record_path(space).exists()
