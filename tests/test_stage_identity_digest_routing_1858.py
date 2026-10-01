"""Old byte profiles and owner routing for generic Stage A/B identities."""
from __future__ import annotations

import hashlib
import json
from typing import cast

import pytest

from prismaquant import stage_a_head as head
from prismaquant import stage_a_produced_output as produced
from prismaquant import stage_b_prep_io as preparation


def _identity_publisher(owner: object = "owner"):
    publication = object.__new__(produced.BoundaryProducedPublication)
    publication.instance = {"owner_action_key": owner, "owner_attempt": {"nonce": "one"}}
    publication._generation = None
    return publication


@pytest.mark.parametrize("owner,payload", [
    ("", b""), ("owner", b"owner"), ("é\x00💡", b"\xc3\xa9\x00\xf0\x9f\x92\xa1"),
    (17, b"17"), (True, b"True"), (None, b"None"),
])
def test_generation_preserves_text_and_sixteen_hex(owner, payload):
    publication = _identity_publisher(owner)
    expected = hashlib.sha256(payload).hexdigest()[:16]
    assert publication.generation == expected
    publication.instance["owner_action_key"] = "changed"
    publication.instance["owner_attempt"] = {"nonce": "two"}
    assert publication.generation == expected


def test_generation_native_missing_key_and_surrogate_refusals():
    publication = _identity_publisher()
    publication.instance = {}
    with pytest.raises(KeyError, match="owner_action_key"):
        _ = publication.generation
    publication.instance = {"owner_action_key": "\ud800"}
    with pytest.raises(UnicodeEncodeError) as refused:
        _ = publication.generation
    assert refused.value.__cause__ is None
    assert publication._generation is None


def test_generation_uses_existing_text_owner_once(monkeypatch):
    calls = []
    monkeypatch.setattr(produced, "text_sha256hex", lambda text: calls.append(text) or "d" * 64)
    publication = _identity_publisher("é\x00owner")
    assert publication.generation == "d" * 16
    assert publication.generation == "d" * 16
    assert calls == ["é\x00owner"]


@pytest.mark.parametrize("kind,boundary,group,probe,text,readable", [
    ("boundary", 0, 0, None, "boundary\x000\x00-\x000\x00cached", "b0"),
    ("cotangent", 12, 3, 0, "cotangent\x0012\x000\x003\x00cached", "b12p0"),
    ("é\x00💡", 2, 4, 9, "é\x00💡\x002\x009\x004\x00cached", "b2p9"),
])
def test_batch_identity_preserves_legacy_nul_frame_and_twelve_hex(
        kind, boundary, group, probe, text, readable):
    publication = _identity_publisher()
    publication._generation = "cached"
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    assert publication.batch_id_for(
        kind=kind, boundary_index=boundary, group_index=group, probe_index=probe,
    ) == f"stagea-{kind}-{readable}-g{group}-cached-{expected}"


@pytest.mark.parametrize("field,value,message", [
    ("boundary_index", -1, "boundary group coordinates must be nonnegative integers"),
    ("boundary_index", True, "boundary group coordinates must be nonnegative integers"),
    ("group_index", 1.0, "boundary group coordinates must be nonnegative integers"),
    ("group_index", False, "boundary group coordinates must be nonnegative integers"),
    ("probe_index", -1, "a boundary group probe index is nonnegative"),
    ("probe_index", True, "a boundary group probe index is nonnegative"),
    ("kind", "", "a boundary group kind is a bare name"),
    ("kind", "a/b", "a boundary group kind is a bare name"),
])
def test_batch_refuses_coordinates_and_kind_before_generation(field, value, message):
    publication = _identity_publisher()
    publication.instance = {}
    arguments = {"boundary_index": 0, "group_index": 0, field: value}
    with pytest.raises(ValueError) as refused:
        publication.batch_id_for(**arguments)
    assert str(refused.value) == message
    assert refused.value.__cause__ is None
    assert publication._generation is None


def test_batch_native_kind_and_utf8_refusals():
    publication = _identity_publisher()
    publication._generation = "cached"
    with pytest.raises(TypeError):
        publication.batch_id_for(kind=cast(str, 1), boundary_index=0, group_index=0)
    with pytest.raises(UnicodeEncodeError):
        publication.batch_id_for(kind="\ud800", boundary_index=0, group_index=0)


def test_batch_uses_existing_text_owner_after_local_framing(monkeypatch):
    calls = []
    monkeypatch.setattr(produced, "text_sha256hex", lambda text: calls.append(text) or "e" * 64)
    publication = _identity_publisher()
    publication._generation = "cached"
    assert publication.batch_id_for(
        kind="cotangent", boundary_index=2, probe_index=1, group_index=4,
    ) == "stagea-cotangent-b2p1-g4-cached-eeeeeeeeeeee"
    assert calls == ["cotangent\x002\x001\x004\x00cached"]


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


_TEMPLATE_BYTES = (
    b'{"durable_maxima": {"checkpoint_max_bytes": 0, "payload_max_bytes": 13, '
    b'"temp_max_bytes": 13}, "output_prefix": "/metadata", "permitted_tiers": ["tier"], '
    b'"schema": "prismaquant.prismabuild.produced_output_template.v1", '
    b'"slots": {"stage_b_metadata": {"class": "payload"}}, "version": 1, '
    b'"working_demands": {"tier": {"minimum_gib": 0, "window_gib": 0}}, "write_only": true}'
)


def test_template_preserves_handwritten_spaced_json_bytes_and_sixteen_hex():
    template = preparation.build_preparation_template(
        metadata_root="/ignored/../metadata", tier="tier", payload_max_bytes=13,
    )
    expected = json.loads(_TEMPLATE_BYTES)
    expected["template_id"] = "pq-stage-b-preparation-" + hashlib.sha256(
        _TEMPLATE_BYTES).hexdigest()[:16]
    assert template == expected


@pytest.mark.parametrize("maximum", [0, -1, True, 1.0, "13"])
def test_template_rejects_nonpositive_or_noninteger_maximum(maximum):
    with pytest.raises(ValueError) as refused:
        preparation.build_preparation_template(
            metadata_root="/metadata", tier="tier", payload_max_bytes=maximum,
        )
    assert str(refused.value) == "the preparation template needs a positive payload maximum"
    assert refused.value.__cause__ is None


def test_template_root_and_nonstring_tier_profiles():
    template = preparation.build_preparation_template(
        metadata_root="/méta/\ud800", tier=cast(str, 17), payload_max_bytes=13,
    )
    body = {key: value for key, value in template.items() if key != "template_id"}
    assert body["permitted_tiers"] == ["17"]
    assert body["working_demands"] == {"17": {"minimum_gib": 0, "window_gib": 0}}
    payload = json.dumps(body, sort_keys=True).encode("utf-8")
    assert b"\\u00e9" in payload and b"\\ud800" in payload
    assert template["template_id"] == "pq-stage-b-preparation-" + hashlib.sha256(payload).hexdigest()[:16]


def test_template_uses_existing_text_owner_after_local_serialization(monkeypatch):
    calls = []
    monkeypatch.setattr(preparation, "text_sha256hex", lambda text: calls.append(text) or "a" * 64)
    template = preparation.build_preparation_template(
        metadata_root="/metadata", tier="tier", payload_max_bytes=13,
    )
    assert template["template_id"] == "pq-stage-b-preparation-" + "a" * 16
    assert calls == [_TEMPLATE_BYTES.decode("utf-8")]
