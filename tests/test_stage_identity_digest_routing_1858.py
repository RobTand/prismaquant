"""Native launch coordinates and retained scientific Stage A roster bytes."""
from __future__ import annotations

import hashlib
import json

import pytest

from prismaquant import stage_a_head as head
from prismaquant import stage_a_produced_output as produced


def test_generation_uses_full_native_launch_not_removed_pb_owner(tmp_path):
    from produced_native_fixtures import publication
    owner = publication(tmp_path)
    owner.producer['launch_sha256'] = '0123456789abcdef' + 'b' * 48
    assert owner.generation == '0123456789abcdef'


@pytest.mark.parametrize('kind,boundary,group,probe', [
    ('boundary', 0, 0, None), ('cotangent', 12, 3, 0), ('checkpoint', 2, 4, 9),
])
def test_batch_coordinates_bind_the_full_launch(tmp_path, kind, boundary, group, probe):
    from produced_native_fixtures import publication
    owner = publication(tmp_path)
    for tail in ('a', 'b'):
        launch = '0123456789abcdef' + tail * 48
        owner.producer['launch_sha256'] = launch
        raw = json.dumps([kind, boundary, group, probe, launch],
                         separators=(',', ':'), ensure_ascii=True).encode()
        readable = f'b{boundary}' + ('' if probe is None else f'p{probe}')
        expected = f'stagea-{kind}-{readable}-g{group}-0123456789abcdef-{hashlib.sha256(raw).hexdigest()[:12]}'
        assert owner.batch_id_for(kind=kind, boundary_index=boundary,
            group_index=group, probe_index=probe) == expected


@pytest.mark.parametrize('field,value', [
    ('boundary_index', -1), ('boundary_index', True), ('group_index', 1.0),
    ('group_index', False), ('probe_index', -1), ('probe_index', True),
    ('kind', ''), ('kind', 'a/b'), ('kind', 1), ('kind', '\ud800'),
])
def test_native_batch_refuses_invalid_coordinates(tmp_path, field, value):
    from produced_native_fixtures import publication
    owner = publication(tmp_path)
    with pytest.raises(produced.BoundaryProducedBindingError):
        owner.batch_id_for(**{'boundary_index': 0, 'group_index': 0, field: value})


def _roster_inputs(names, *, capsule=None):
    return head.stage_a_roster(
        dict.fromkeys(names), capsule=capsule, plan_sha256="plan",
        prepared_sha256="prepared", read_manifest_sha256="reads",
        calibration_shape=(4, 8), campaign_scope="local",
    )


@pytest.mark.parametrize("names,payload", [
    ([], b""), (["b", "a"], b"a\nb\n"),
    (["layer.💡", "layer.é"], b"layer.\xc3\xa9\nlayer.\xf0\x9f\x92\xa1\n"),
    (["a\nb", "a"], b"a\na\nb\n"), ([10, 2], b"2\n10\n"),
])
def test_roster_preserves_sorted_spelling_trailing_lf_and_full_hex(names, payload):
    assert _roster_inputs(names) == (hashlib.sha256(payload).hexdigest(), "local")


def test_roster_native_sorting_and_utf8_refusals():
    with pytest.raises(TypeError):
        _roster_inputs(["a", 1])
    with pytest.raises(UnicodeEncodeError):
        _roster_inputs(["\ud800"])


def test_roster_capsule_delegation_is_unchanged(monkeypatch):
    from prismaquant import joint_forward_campaign, joint_forward_resume

    calls = []
    document = {"campaign": "opaque"}
    names = {"b": None, "a": None}

    def read_capsule(path, digest):
        calls.append(("read", path, digest))
        return document, None

    def resolve_campaign(value, **arguments):
        assert value is document
        assert arguments == {
            "plan_sha256": "plan", "prepared_sha256": "prepared",
            "read_manifest_sha256": "reads", "formats_by_qname": names,
            "calibration_shape": [4, 8],
        }
        assert arguments["formats_by_qname"] is names
        calls.append(("resolve",))
        return {"unit_roster_sha256": "c" * 64, "campaign_scope": "capsule"}

    monkeypatch.setattr(joint_forward_resume, "_read", read_capsule)
    monkeypatch.setattr(joint_forward_campaign, "resolve_forward_campaign", resolve_campaign)
    assert head.stage_a_roster(
        names, capsule={"path": "opaque", "sha256": "bound"},
        plan_sha256="plan", prepared_sha256="prepared", read_manifest_sha256="reads",
        calibration_shape=(4, 8), campaign_scope="local",
    ) == ("c" * 64, "capsule")
    assert calls == [("read", "opaque", "bound"), ("resolve",)]


def test_roster_uses_existing_text_owner_without_changing_frame(monkeypatch):
    calls = []
    monkeypatch.setattr(head, "text_sha256hex", lambda text: calls.append(text) or "f" * 64)
    assert _roster_inputs(["b", "a"]) == ("f" * 64, "local")
    assert calls == ["a\nb\n"]

