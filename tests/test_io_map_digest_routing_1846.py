"""Pin generic IO/map byte identities separately from their acquisition policies."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from prismaquant import io_engine, residency_map, staged_lease, staged_tier_policy

BUFFERS = (b"", b"raw-\xc3\xa9\x00\r\n", bytes(range(256)))
MANIFEST = "a" * 64


@pytest.mark.parametrize("raw", BUFFERS)
@pytest.mark.parametrize("staged", (False, True))
def test_io_raw_receipt_keeps_its_one_read_and_declared_identity(
    tmp_path, monkeypatch, raw, staged
):
    declared = tmp_path / "declared.bin"
    declared.write_bytes(raw)
    source = declared
    entry = None
    if staged:
        source = tmp_path / "stage.bin"
        source.write_bytes(raw)
        entry = {"stage_path": str(source), "sha256": hashlib.sha256(raw).hexdigest()}
        monkeypatch.setattr(staged_tier_policy, "policy_is_active", lambda: False)
        monkeypatch.setattr(residency_map, "residency_resolver", lambda: None)
    opened = []
    original = Path.open

    def counted(path, *args, **kwargs):
        opened.append((path, args))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    timing = {}
    result, receipt, signature = io_engine.read_file(
        declared, len(raw), staged=entry, timing=timing
    )
    assert type(result) is bytes and result == raw
    expected = {
        "path": str(declared), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if staged:
        expected["serving_tier"] = "stage"
    assert receipt == expected
    assert len(receipt["sha256"]) == 64
    info = declared.lstat()
    assert signature == (
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
    )
    assert opened == [(source, ("rb",))]
    assert timing.keys() == {"read_s"} and timing["read_s"] >= 0


@pytest.mark.parametrize("refusal", ("bound", "symlink"))
def test_io_pre_read_refusals_keep_their_exact_messages(tmp_path, monkeypatch, refusal):
    path = tmp_path / "declared.bin"
    path.write_bytes(b"abc")
    limit = 2
    message = "PWC file exceeds the explicit read buffer bound"
    if refusal == "symlink":
        link = tmp_path / "link.bin"
        link.symlink_to(path)
        path, limit = link, 3
        message = "PWC file receipt requires a regular file, not a symlink"

    def forbidden(*args, **kwargs):
        pytest.fail("a pre-read refusal opened payload bytes")

    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(RuntimeError) as refused:
        io_engine.read_file(path, limit)
    assert str(refused.value) == message and refused.value.__cause__ is None


def test_io_native_pool_open_failure_is_not_wrapped(tmp_path, monkeypatch):
    path = tmp_path / "declared.bin"
    path.write_bytes(b"abc")
    error = PermissionError("fixture open refused")

    def denied(*args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(PermissionError) as refused:
        io_engine.read_file(path, 3)
    assert refused.value is error and refused.value.__cause__ is None


def test_io_stage_digest_refusal_still_precedes_return(tmp_path, monkeypatch):
    path, stage = tmp_path / "declared.bin", tmp_path / "stage.bin"
    path.write_bytes(b"abc")
    stage.write_bytes(b"xyz")
    monkeypatch.setattr(staged_tier_policy, "policy_is_active", lambda: False)
    monkeypatch.setattr(residency_map, "residency_resolver", lambda: None)
    with pytest.raises(io_engine.StagedReadRefused) as refused:
        io_engine.read_file(
            path, 3, staged={"stage_path": str(stage), "sha256": hashlib.sha256(b"abc").hexdigest()}
        )
    assert str(refused.value) == "staged bytes differ from the map digest"
    assert refused.value.__cause__ is None


def test_io_unsealed_buffer_routes_exactly_once_after_read(tmp_path, monkeypatch):
    path = tmp_path / "declared.bin"
    path.write_bytes(BUFFERS[1])
    seen = []
    opened = []
    original = Path.open

    def counted(p, *args, **kwargs):
        opened.append(p)
        return original(p, *args, **kwargs)

    def owner(raw):
        assert opened == [path]
        seen.append(raw)
        return hashlib.sha256(raw).hexdigest()

    monkeypatch.setattr(Path, "open", counted)
    monkeypatch.setattr(io_engine, "bytes_sha256hex", owner)
    raw, receipt, _signature = io_engine.read_file(path, len(BUFFERS[1]))
    assert seen == [BUFFERS[1]] and seen[0] is raw
    assert receipt["sha256"] == hashlib.sha256(BUFFERS[1]).hexdigest()


class _MapFixtureError(ValueError):
    pass


def _checked_map(with_ram=False):
    row = {
        "stage_path": "/stage/é.bin", "bytes": 3, "offset": 7,
        "sha256": "c" * 64,
    }
    result = {
        "manifest_sha256": MANIFEST,
        "entries": {"file:/pool/é:part.bin": row},
        "tier_id": "ssd-id", "generation": 9, "stage_root": "/stage",
        "leads": ["lead-b", "lead-a"],
    }
    if with_ram:
        row["ram_path"] = "/ram/é.bin"
        result.update(ram_tier_id="ram-id", ram_root="/ram", ram_epoch=11)
    return result


def _map_resolver(tmp_path):
    resolver = residency_map.ResidencyResolver(tmp_path / "map.json")
    resolver._manifest_sha256 = MANIFEST
    return resolver


@pytest.mark.parametrize("raw", BUFFERS)
@pytest.mark.parametrize("with_ram", (False, True))
def test_map_identity_hashes_original_raw_not_validated_projection(
    tmp_path, monkeypatch, raw, with_ram
):
    resolver = _map_resolver(tmp_path)
    payload, validated = object(), []
    checked = _checked_map(with_ram)

    def validate(value):
        validated.append(value)
        return checked

    client = SimpleNamespace(validate_residency_map=validate, ResidencyMapError=_MapFixtureError)
    monkeypatch.setattr(staged_lease, "client_sdk", lambda: client)
    resolver._adopt(payload, raw)
    assert validated == [payload]
    assert resolver._map_sha256 == hashlib.sha256(raw).hexdigest()
    assert resolver._entries == {
        "file:/pool/é:part.bin": dict(checked["entries"]["file:/pool/é:part.bin"],
                                    declared_path="/pool/é:part.bin")
    }
    assert resolver._leads == ("lead-b", "lead-a")
    assert (resolver._tier_id, resolver._stage_root, resolver._generation) == ("ssd-id", "/stage", 9)
    assert resolver._ram_tier_id == ("ram-id" if with_ram else None)
    assert resolver._ram_root == ("/ram" if with_ram else None)
    assert resolver._ram_epoch == (11 if with_ram else None)
    assert resolver._real_entries is None and resolver._refused is None


@pytest.mark.parametrize("failure", ("validator", "manifest"))
def test_map_validation_and_binding_refuse_before_adoption(tmp_path, monkeypatch, failure):
    resolver = _map_resolver(tmp_path)

    def validate(_payload):
        if failure == "validator":
            raise _MapFixtureError("fixture map invalid")
        return dict(_checked_map(), manifest_sha256="b" * 64)

    client = SimpleNamespace(validate_residency_map=validate, ResidencyMapError=_MapFixtureError)
    monkeypatch.setattr(staged_lease, "client_sdk", lambda: client)
    # A prematurely reordered hash would reject this object with TypeError.
    with pytest.raises(residency_map.ResidencyMapRefused) as refused:
        resolver._adopt(object(), cast(bytes, object()))
    expected = ("fixture map invalid" if failure == "validator" else
                "residency map names data manifest bbbbbbbbbbbb, this run reads aaaaaaaaaaaa")
    assert str(refused.value) == expected and refused.value.__cause__ is None
    assert resolver._entries == {} and resolver._map_sha256 is None


def test_map_raw_buffer_routes_after_validation_and_entry_adoption(tmp_path, monkeypatch):
    resolver = _map_resolver(tmp_path)
    events = []

    def validate(_payload):
        events.append("validated")
        return _checked_map()

    def owner(raw):
        assert events == ["validated"]
        assert resolver._entries["file:/pool/é:part.bin"]["declared_path"] == "/pool/é:part.bin"
        events.append(raw)
        return hashlib.sha256(raw).hexdigest()

    client = SimpleNamespace(validate_residency_map=validate, ResidencyMapError=_MapFixtureError)
    monkeypatch.setattr(staged_lease, "client_sdk", lambda: client)
    monkeypatch.setattr(residency_map, "bytes_sha256hex", owner)
    resolver._adopt(object(), BUFFERS[1])
    assert events == ["validated", BUFFERS[1]]
    assert resolver._map_sha256 == hashlib.sha256(BUFFERS[1]).hexdigest()
