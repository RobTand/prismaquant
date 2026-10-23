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


@pytest.mark.parametrize("kind", ["census", "merged_cost"], ids=["json", "pickle"])
@pytest.mark.parametrize("replace_after_hash", [False, True],
                         ids=["unchanged", "replace-after-owned-hash"])
def test_public_scope_parses_its_authenticated_owned_bytes(
    tmp_path, monkeypatch, kind, replace_after_hash,
):
    workspace = _workspace(tmp_path)
    plan = json.loads(workspace["plan"].read_text())
    bound = plan["inputs"][kind]
    path = Path(bound["path"])
    payload = _cost_payload(workspace["names"])

    def scope():
        if kind == "census":
            return table.joint_campaign_scope(plan)
        return table.aqua_requested_cells(plan, payload, list(MEASURED))

    expected = scope()
    if kind == "census":
        replacement = json.loads(path.read_text())
        replacement["text_sha256"] = "c" * 64
        replacement = json.dumps(replacement).encode()
    else:
        replacement = copy.deepcopy(payload)
        del replacement["costs"][workspace["names"][0]][MEASURED[0]]
        replacement = pickle.dumps(replacement)
    events = []
    real_hasher = table.bytes_sha256hex

    def hash_then_replace(raw):
        actual = real_hasher(raw)
        if replace_after_hash and actual == bound["sha256"] and not events:
            temporary = path.with_name(path.name + ".replacement")
            temporary.write_bytes(replacement)
            temporary.replace(path)
            changed = real_hasher(path.read_bytes())
            assert changed != actual
            events.append((actual, changed))
        return actual

    monkeypatch.setattr(table, "bytes_sha256hex", hash_then_replace)
    assert scope() == expected
    assert bool(events) is replace_after_hash


@pytest.mark.parametrize("kind", ["census", "merged_cost"], ids=["json", "pickle"])
def test_public_scope_refuses_bytes_changed_before_owned_read(tmp_path, kind):
    workspace = _workspace(tmp_path)
    plan = json.loads(workspace["plan"].read_text())
    path = Path(plan["inputs"][kind]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(table.ScopeRefused, match="not the bound"):
        if kind == "census":
            table.joint_campaign_scope(plan)
        else:
            table.aqua_requested_cells(
                plan, _cost_payload(workspace["names"]), list(MEASURED))
