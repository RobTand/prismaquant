"""AQUA requested coverage must derive from the bytes its plan authenticates.

Metadata-only public gate replay with existing tiny local campaign fixtures.
A real SHA-256 read finishes before an atomic producer-style path replacement;
no digest is fabricated, no PB/GPU submission or pricing is executed.
"""
from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import pytest

from prismaquant import tessera_campaign_table as table
from tests.test_aqua_campaign_submit import MEASURED, _cost_payload, _workspace


@pytest.mark.parametrize("replace_after_hash", [False, True],
                         ids=["unchanged-bound-table", "replaced-after-real-hash"])
def test_requested_cells_use_the_authenticated_planned_table(
    tmp_path, monkeypatch, replace_after_hash,
):
    workspace = _workspace(tmp_path)
    plan = json.loads(workspace["plan"].read_text())
    bound = plan["inputs"]["merged_cost"]
    path = Path(bound["path"])
    original = _cost_payload(workspace["names"])
    wanted = frozenset((name, fmt) for name in workspace["names"] for fmt in MEASURED)
    payload = copy.deepcopy(original)
    events = []
    real_hasher = table.file_sha256hex
    assert real_hasher(path) == bound["sha256"]
    if replace_after_hash:
        # Unit membership and the union of formats remain unchanged. Only one
        # original planned cell disappears, the exact denominator the gate owns.
        del payload["costs"][workspace["names"][0]][MEASURED[0]]
        replacement = pickle.dumps(payload)

        def hash_then_replace(candidate, *args, **kwargs):
            actual = real_hasher(candidate, *args, **kwargs)
            if Path(candidate) == path and not events:
                assert actual == bound["sha256"]
                temporary = path.with_name(path.name + ".replacement")
                temporary.write_bytes(replacement)
                temporary.replace(path)
                changed = real_hasher(path)
                assert changed != actual
                events.append({"authenticated": actual, "replacement": changed})
            return actual

        monkeypatch.setattr(table, "file_sha256hex", hash_then_replace)
    try:
        record, cells = table.aqua_requested_cells(plan, payload, list(MEASURED))
    except table.ScopeRefused:
        # Same-byte parsing may retain the authenticated original (and refuse
        # the narrowed candidate), or a fence may refuse the replacement.
        if not replace_after_hash:
            raise
        return
    assert cells == wanted, (
        "Coverage narrowed to a table the declared SHA never authenticated: "
        f"missing={sorted(wanted - cells)}, replacements={events}, record={record}")
    assert record["requested_cells"] == len(wanted)
