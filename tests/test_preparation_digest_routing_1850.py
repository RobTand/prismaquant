"""Pin generic preparation byte identities and routing, without GPU work."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from prismaquant import staged_whole_file
from prismaquant import stage_b_prep_io as prep
from prismaquant.staged_tier_policy import TierPolicyRefused


RAW_VALUES = [b"", b"\x00\xff\xc3\xa9\r\n", bytearray(b"mutable"), memoryview(b"view")]


def _reader(monkeypatch, raw, *, failure=None):
    calls = []
    resolver = object()
    reader = prep.StagedPreparationReads(resolver, manifest_sha256="m" * 64)

    def staged(resolver_arg, path, entry, *, label, max_bytes=None):
        calls.append((resolver_arg, path, entry, label))
        if failure is not None:
            raise failure
        return raw

    monkeypatch.setattr(staged_whole_file, "read_staged_entry", staged)
    return reader, calls


class _ProducedOutput:
    """Native publication protocol control; byte/destination validation has its own suite."""
    write_only = True
    generation = "gen-fixture"
    binding = {"sha256": "a" * 64}

    def __init__(self, *, outcome=None, reject=False):
        self.events = []
        self.validated = []
        self.outcome = {"ok": True, "ref": "origin-ref"} if outcome is None else outcome
        self.reject = reject

    def bind_session(self, session):
        self.session = session

    def descriptor_for(self, reference, *, producer_generation, file_identity):
        self.events.append(("descriptor", reference, producer_generation, file_identity))
        if self.reject:
            raise ValueError("descriptor refused")
        descriptor = {"path": reference.path, "bytes": reference.file_bytes,
                      "sha256": reference.sha256, "producer_generation": producer_generation,
                      "file_identity": file_identity}
        self.validated.append(descriptor)
        return descriptor

    def publish(self, *, batch_id, descriptors):
        self.events.append(("publish", batch_id, descriptors))
        if not self.outcome["ok"]:
            raise prep.PreparationPublicationRefused("native publication refused")
        return self.outcome


def _publication(*, outcome=None, reject=False):
    po = _ProducedOutput(outcome=outcome, reject=reject)
    return prep.PreparationPublication(po), po


@pytest.mark.parametrize("raw", RAW_VALUES)
@pytest.mark.parametrize("pinned", [False, True])
def test_staged_bytes_keep_identity_and_both_digest_contracts(monkeypatch, raw, pinned):
    reader, calls = _reader(monkeypatch, raw)
    path = Path("declared-é.bin")
    digest = hashlib.sha256(raw).hexdigest()
    entry = {"sha256": digest}
    out = reader._read_entry(path, entry, where="header-é", sha256=digest if pinned else None)
    assert out is raw
    assert calls == [(reader.resolver, path, entry, "header-é")]


@pytest.mark.parametrize("map_digest,caller_digest", [("0" * 64, None), (None, "0" * 64)])
def test_staged_digest_refusals_remain_exact(monkeypatch, map_digest, caller_digest):
    raw = b"actual\x00bytes"
    reader, calls = _reader(monkeypatch, raw)
    digest = hashlib.sha256(raw).hexdigest()
    path = Path("declared.bin")
    entry = {"sha256": digest if map_digest is None else map_digest}
    with pytest.raises(prep.PreparationReadRefused) as refused:
        reader._read_entry(path, entry, where="source", sha256=caller_digest)
    assert str(refused.value) == (
        f"source at {path}: the staged bytes hash to {digest}, not the "
        "digest the manifest and the caller require"
    )
    assert refused.value.__cause__ is None
    assert len(calls) == 1


def test_stage_refusal_keeps_cause_and_does_not_hash(monkeypatch):
    cause = TierPolicyRefused("lease missing")
    reader, calls = _reader(monkeypatch, b"unused", failure=cause)
    with pytest.raises(prep.PreparationReadRefused) as refused:
        reader._read_entry(Path("declared.bin"), {}, where="header", sha256=None)
    assert str(refused.value) == "header at declared.bin was not served from the stage: lease missing"
    assert refused.value.__cause__ is cause
    assert len(calls) == 1


def test_missing_map_digest_keeps_native_key_refusal(monkeypatch):
    reader, calls = _reader(monkeypatch, b"raw")
    with pytest.raises(KeyError) as refused:
        reader._read_entry(Path("declared.bin"), {}, where="header", sha256=None)
    assert refused.value.args == ("sha256",)
    assert refused.value.__cause__ is None
    assert len(calls) == 1


def test_missing_stage_never_reads_the_pool(monkeypatch):
    reader = prep.StagedPreparationReads(
        SimpleNamespace(staged_read=lambda *args, **kwargs: None), manifest_sha256="m" * 64,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("pool read")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(prep.PreparationReadRefused) as refused:
        reader.whole("declared.bin", where="header")
    assert str(refused.value) == (
        "header at declared.bin is not staged for manifest mmmmmmmmmmmm: "
        "refusing rather than reading the pool"
    )
    assert refused.value.__cause__ is None


@pytest.mark.parametrize("created", [[], [("empty.bin", b"")], [
    ("é.bin", b"\xc3\xa9\x00"), ("binary.bin", b"\xff\r\n"),
]])
def test_publication_preserves_full_descriptors_order_and_batch_accounting(created):
    publication, po = _publication()
    identities = {path: (1, index + 1) for index, (path, _) in enumerate(created)}
    out = publication.commit("records", created, writer_identities=identities)
    assert out == {"ok": True, "ref": "origin-ref"}
    assert out is po.outcome
    assert po.session == {"generation": po.generation, "run_identity_sha256": po.binding["sha256"]}
    assert [event[0] for event in po.events] == ["descriptor"] * len(created) + ["publish"]
    for event, (path, raw) in zip(po.events[:-1], created, strict=True):
        reference = event[1]
        assert (reference.path, reference.name, reference.file_bytes, reference.sha256) == (
            path, Path(path).name, len(raw), hashlib.sha256(raw).hexdigest())
        assert event[2] == "stage-b-prep-records-gen-fixture"
        assert event[3] is identities[path]
    assert po.events[-1][1] == "stage-b-prep-records-gen-fixture"
    assert po.events[-1][2] == po.validated
    assert all(a is b for a, b in zip(po.events[-1][2], po.validated, strict=True))
    assert publication.batches == [{
        "kind": "records", "batch_id": "stage-b-prep-records-gen-fixture", "files": len(created),
        "bytes": sum(len(raw) for _, raw in created), "ref": "origin-ref",
    }]


def test_descriptor_refusal_precedes_commit_and_accounting():
    publication, po = _publication(reject=True)
    with pytest.raises(ValueError) as refused:
        publication.commit("records", [("first.bin", b"first"), ("second.bin", b"second")])
    assert str(refused.value) == "descriptor refused" and refused.value.__cause__ is None
    assert [event[0] for event in po.events] == ["descriptor"]
    assert publication.batches == []


def test_payload_type_refusal_precedes_descriptor_validation():
    publication, po = _publication()
    with pytest.raises(TypeError) as refused:
        publication.commit("records", [("str.bin", cast(bytes, "not bytes"))])
    assert str(refused.value) == "Strings must be encoded before hashing"
    assert refused.value.__cause__ is None
    assert po.events == [] and publication.batches == []


def test_commit_refusal_keeps_exact_message_and_does_not_account():
    outcome = {"ok": False, "refusal": "credit"}
    publication, po = _publication(outcome=outcome)
    with pytest.raises(prep.PreparationPublicationRefused) as refused:
        publication.commit("records", [("a.bin", b"A")])
    assert str(refused.value) == "native publication refused"
    assert refused.value.__cause__ is None
    assert [event[0] for event in po.events] == ["descriptor", "publish"]
    assert publication.batches == []


def test_bad_kind_refuses_before_validation_or_commit():
    publication, po = _publication()
    from prismaquant.stage_a_produced_output import BoundaryProducedBindingError
    with pytest.raises(BoundaryProducedBindingError, match="^bounded bare identifier required$"):
        publication.commit("bad/kind", [("a.bin", b"A")])
    assert po.events == [] and publication.batches == []


def test_staged_read_uses_the_existing_byte_owner(monkeypatch):
    raw = b"stage route\x00\xff"
    reader, _calls = _reader(monkeypatch, raw)
    seen = []
    digest = "e" * 64

    def owner(data):
        seen.append(data)
        return digest

    monkeypatch.setattr(prep, "bytes_sha256hex", owner)
    assert reader._read_entry(Path("a.bin"), {"sha256": digest}, where="header", sha256=digest) is raw
    assert len(seen) == 1 and seen[0] is raw


def test_publication_uses_the_existing_byte_owner(monkeypatch):
    publication, po = _publication()
    raw = b"publication route\x00\xff"
    seen = []

    def owner(data):
        seen.append(data)
        return "e" * 64

    monkeypatch.setattr(prep, "bytes_sha256hex", owner)
    publication.commit("records", [("a.bin", raw)])
    assert len(seen) == 1 and seen[0] is raw
    assert po.events[0][1].sha256 == "e" * 64
