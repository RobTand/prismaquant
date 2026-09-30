"""Preserve journal pickle bytes and reader policy while sharing raw hashes."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pytest

from prismaquant import cost_stage_checkpoint as checkpoint


STAGE = "cost"
QNAME = "layers.é.weight"
IDENTITY = "a" * 64


def _state(case: str) -> dict[str, object]:
    if case == "empty":
        return {}
    if case == "nested":
        return {"unicode": "雪", "binary": b"left\x00right", "nested": [1, {"x": None}]}
    shared = ["shared", b"\x00"]
    return {"z": shared, "a": shared, "last": 3}


def _legacy_envelope(state: object) -> tuple[bytes, dict[str, object]]:
    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    return payload, {
        "schema": "prismaquant.cost_stage_checkpoint.unit.v1",
        "stage": STAGE,
        "qname": QNAME,
        "identity_sha256": IDENTITY,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload": payload,
    }


def _write_envelope(root: Path, envelope: dict[str, object]) -> Path:
    path = root / "legacy.pkl"
    path.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
    return path


def _read(path: Path) -> dict[str, object]:
    return checkpoint._load_unit(
        path, stage=STAGE, qname=QNAME, identity_sha256=IDENTITY,
    )


def _forbid_loads(_payload: bytes) -> object:
    raise AssertionError("refused payload must not be deserialized")


@pytest.mark.parametrize("case", ["empty", "nested", "insertion-sharing"])
def test_writer_preserves_exact_inner_and_outer_pickle_bytes(tmp_path: Path, case: str):
    state = _state(case)
    payload, expected = _legacy_envelope(state)
    checkpoint.write_unit(
        tmp_path, stage=STAGE, qname=QNAME, identity_sha256=IDENTITY, state=state,
    )
    path = tmp_path / "units" / f"{hashlib.sha256(QNAME.encode('utf-8')).hexdigest()}.pkl"
    raw = path.read_bytes()
    assert raw == pickle.dumps(expected, protocol=pickle.HIGHEST_PROTOCOL)
    actual = pickle.loads(raw)
    assert actual["payload"] == payload
    assert actual["payload_sha256"] == hashlib.sha256(payload).hexdigest()
    if case == "insertion-sharing":
        decoded = pickle.loads(payload)
        assert list(decoded) == ["z", "a", "last"]
        assert decoded["z"] is decoded["a"]


@pytest.mark.parametrize("case", ["empty", "nested", "insertion-sharing"])
def test_reader_accepts_independent_legacy_envelopes(tmp_path: Path, case: str):
    state = _state(case)
    _payload, envelope = _legacy_envelope(state)
    actual = _read(_write_envelope(tmp_path, envelope))
    assert actual == state
    assert list(actual) == list(state)
    if case == "insertion-sharing":
        assert actual["z"] is actual["a"]


@pytest.mark.parametrize("field", ["schema", "stage", "qname", "identity_sha256"])
def test_metadata_mismatch_precedes_payload_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
):
    _payload, envelope = _legacy_envelope({})
    envelope[field] = "wrong"
    envelope["payload"] = None
    envelope["payload_sha256"] = "wrong"
    path = _write_envelope(tmp_path, envelope)
    monkeypatch.setattr(checkpoint.pickle, "loads", _forbid_loads)
    with pytest.raises(RuntimeError) as error:
        _read(path)
    assert str(error.value).startswith(
        f"{STAGE} checkpoint identity mismatch at unit[{QNAME}].{field}: "
    )
    assert str(error.value).endswith("; refusing reuse or recompute")
    assert error.value.__cause__ is None


@pytest.mark.parametrize("payload", [None, "not bytes", bytearray(b"bytes")])
def test_nonbytes_payload_refusal_precedes_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: object,
):
    _raw, envelope = _legacy_envelope({})
    envelope["payload"] = payload
    path = _write_envelope(tmp_path, envelope)
    monkeypatch.setattr(checkpoint.pickle, "loads", _forbid_loads)
    with pytest.raises(RuntimeError) as error:
        _read(path)
    assert str(error.value) == (
        f"{STAGE} unit checkpoint {path} has no byte payload for {QNAME}; "
        "refusing reuse or recompute"
    )
    assert error.value.__cause__ is None


def test_digest_refusal_precedes_state_unpickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _payload, envelope = _legacy_envelope({})
    envelope["payload"] = b"not a pickle"
    envelope["payload_sha256"] = "0" * 64
    path = _write_envelope(tmp_path, envelope)
    monkeypatch.setattr(checkpoint.pickle, "loads", _forbid_loads)
    with pytest.raises(RuntimeError) as error:
        _read(path)
    assert str(error.value) == (
        f"{STAGE} unit checkpoint {path} payload_sha256 differs for "
        f"{QNAME}; refusing reuse or recompute"
    )
    assert error.value.__cause__ is None


def test_corrupt_state_preserves_wrapped_pickle_error(tmp_path: Path):
    _payload, envelope = _legacy_envelope({})
    envelope["payload"] = b"not a pickle"
    envelope["payload_sha256"] = hashlib.sha256(b"not a pickle").hexdigest()
    path = _write_envelope(tmp_path, envelope)
    with pytest.raises(RuntimeError) as error:
        _read(path)
    assert str(error.value) == (
        f"{STAGE} unit checkpoint {path} state is corrupt for {QNAME}; "
        "refusing reuse or recompute"
    )
    assert isinstance(error.value.__cause__, pickle.UnpicklingError)


def test_nonmapping_state_refusal_stays_exact(tmp_path: Path):
    _payload, envelope = _legacy_envelope([1, 2])
    path = _write_envelope(tmp_path, envelope)
    with pytest.raises(RuntimeError) as error:
        _read(path)
    assert str(error.value) == (
        f"{STAGE} unit checkpoint {path} state is not an object for "
        f"{QNAME}; refusing reuse or recompute"
    )
    assert error.value.__cause__ is None


def test_writer_routes_only_the_exact_payload_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state = _state("insertion-sharing")
    expected, _envelope = _legacy_envelope(state)
    seen: list[bytes] = []

    def owner(payload: bytes) -> str:
        seen.append(payload)
        return hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(checkpoint, "bytes_sha256hex", owner)
    checkpoint.write_unit(
        tmp_path, stage=STAGE, qname=QNAME, identity_sha256=IDENTITY, state=state,
    )
    assert seen == [expected]


def test_reader_routes_only_the_stored_payload_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload, envelope = _legacy_envelope(_state("nested"))
    path = _write_envelope(tmp_path, envelope)
    seen: list[bytes] = []

    def owner(raw: bytes) -> str:
        seen.append(raw)
        return hashlib.sha256(raw).hexdigest()

    monkeypatch.setattr(checkpoint, "bytes_sha256hex", owner)
    assert _read(path) == _state("nested")
    assert seen == [payload]
