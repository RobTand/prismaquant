"""Byte-identical final manifest hashing, without changing producer/binder policy."""

from __future__ import annotations

import gzip
import hashlib
from copy import deepcopy

import pytest

from prismaquant import joint_layer_quanta as quanta

_VARIANTS = ("boundary", "executable")
_GOLDENS = (
    ({}, b"{}\n"),
    (
        {"z": "é", "a": [1, -0.0, "\x00\r\n"]},
        b'{"z":"\\u00e9","a":[1,-0.0,"\\u0000\\r\\n"]}\n',
    ),
    (
        {"x": [float("nan"), float("inf"), -float("inf")]},
        b'{"x":[NaN,Infinity,-Infinity]}\n',
    ),
    (
        {"a": True, "n": None, "i": 2**63 - 1},
        b'{"a":true,"n":null,"i":9223372036854775807}\n',
    ),
)


def _install(monkeypatch, variant, manifests):
    events = []
    bindings = []
    original_seal = quanta.seal_manifest_bytes

    def build(record, *_args, **_kwargs):
        events.append(("build", record["quantum_id"]))
        return manifests[record["quantum_id"]]

    def seal(manifest):
        events.append(("seal", manifest))
        return original_seal(manifest)

    def bind(record, *_args, **kwargs):
        events.append(("bind", record["quantum_id"]))
        bindings.append(kwargs)
        return {"source": record, "digest": kwargs["manifest_sha256"]}

    builder = (
        "build_quantum_boundary_readset"
        if variant == "boundary"
        else "build_quantum_executable_manifest"
    )
    binder = (
        "bind_quantum_boundary_readset"
        if variant == "boundary"
        else "bind_quantum_executable"
    )
    monkeypatch.setattr(quanta, builder, build)
    monkeypatch.setattr(quanta, binder, bind)
    monkeypatch.setattr(quanta, "seal_manifest_bytes", seal)
    return events, bindings


def _emit(variant, records, *, output_root="/run", metadata_root="/control"):
    fields = {
        "strided_boundaries": [8, 16],
        "n_probes": 2,
        "output_root": output_root,
        "metadata_root": metadata_root,
    }
    if variant == "boundary":
        return quanta.emit_quantum_boundary_readsets({}, records, **fields)
    return quanta.emit_quantum_executable_readsets(
        {}, records, {}, calib={}, render_prerequisite={}, **fields
    )


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize(("manifest", "decoded"), _GOLDENS)
def test_exact_manifest_bytes_hash_and_return_identity(monkeypatch, variant, manifest, decoded):
    record = {"quantum_id": "q-é"}
    events, bindings = _install(monkeypatch, variant, {"q-é": manifest})
    expected_wire = gzip.compress(decoded, mtime=0)
    expected_hash = hashlib.sha256(expected_wire).hexdigest()

    rows = _emit(variant, [record])

    assert quanta.seal_manifest_bytes(manifest) == expected_wire
    assert len(rows) == 1
    assert rows[0]["manifest"] is manifest
    assert rows[0]["manifest_sha256"] == expected_hash
    assert rows[0]["record"]["source"] is record
    assert rows[0]["record"]["digest"] == expected_hash
    suffix = "boundary-readset" if variant == "boundary" else "executable"
    assert rows[0]["manifest_path"] == (
        f"/control/adjoint/bound-readsets/q-é.{suffix}.json.gz"
    )
    assert bindings[0]["manifest"] is manifest
    assert bindings[0]["manifest_sha256"] == expected_hash
    assert [event[0] for event in events[:3]] == ["build", "seal", "bind"]


@pytest.mark.parametrize("variant", _VARIANTS)
def test_final_hash_routes_exact_sealed_bytes_before_binding(monkeypatch, variant):
    manifest = {"key": "é"}
    events, bindings = _install(monkeypatch, variant, {"q": manifest})
    expected_wire = gzip.compress(b'{"key":"\\u00e9"}\n', mtime=0)
    seen = []

    def owner(raw):
        events.append(("hash", raw))
        seen.append(raw)
        return "f" * 64

    monkeypatch.setattr(quanta, "bytes_sha256hex", owner)
    rows = _emit(variant, [{"quantum_id": "q"}])

    assert seen == [expected_wire]
    assert rows[0]["manifest_sha256"] == "f" * 64
    assert bindings[0]["manifest_sha256"] == "f" * 64
    assert [event[0] for event in events] == ["build", "seal", "hash", "bind"]


@pytest.mark.parametrize("variant", _VARIANTS)
def test_empty_records_refuse_before_building(monkeypatch, variant):
    events, bindings = _install(monkeypatch, variant, {})
    with pytest.raises(ValueError) as refused:
        _emit(variant, [])
    assert str(refused.value) == "no quantum records to bind: refusing"
    assert refused.value.__cause__ is None
    assert events == bindings == []


@pytest.mark.parametrize("variant", _VARIANTS)
def test_relative_output_refuses_before_building(monkeypatch, variant):
    events, bindings = _install(monkeypatch, variant, {"q": {}})
    with pytest.raises(ValueError) as refused:
        _emit(variant, [{"quantum_id": "q"}], output_root="relative")
    assert str(refused.value) == "an output root must be absolute: refusing"
    assert refused.value.__cause__ is None
    assert events == bindings == []


@pytest.mark.parametrize("variant", _VARIANTS)
def test_duplicate_refuses_before_second_seal_or_binding(monkeypatch, variant):
    events, bindings = _install(monkeypatch, variant, {"q": {}})
    with pytest.raises(ValueError) as refused:
        _emit(variant, [{"quantum_id": "q"}, {"quantum_id": "q"}])
    assert str(refused.value) == "duplicate quantum binding 'q': refusing"
    assert refused.value.__cause__ is None
    assert [event[0] for event in events] == ["build", "seal", "bind", "build"]
    assert len(bindings) == 1


@pytest.mark.parametrize("variant", _VARIANTS)
def test_iterator_order_and_source_objects_unchanged(monkeypatch, variant):
    records = [{"quantum_id": "z"}, {"quantum_id": "a"}]
    before = deepcopy(records)
    manifests = {"z": {"first": 1}, "a": {"second": 2}}
    events, bindings = _install(monkeypatch, variant, manifests)

    rows = _emit(variant, iter(records))

    assert records == before
    assert [row["record"]["source"] for row in rows] == records
    assert rows[0]["record"]["source"] is records[0]
    assert rows[1]["record"]["source"] is records[1]
    assert [event[1] for event in events if event[0] == "build"] == ["z", "a"]
    assert [binding["manifest"] for binding in bindings] == list(manifests.values())
    assert all(binding["metadata_root"] == "/control" for binding in bindings)


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("bad", ("nonobject", "unsupported", "cycle"))
def test_serializer_refusals_stay_before_binding(monkeypatch, variant, bad):
    if bad == "nonobject":
        manifest = []
        expected = "a data manifest must be a JSON object"
        cause = None
    elif bad == "unsupported":
        manifest = {"x": object()}
        expected = "a data manifest must be canonical JSON data"
        cause = TypeError
    else:
        manifest = {}
        manifest["x"] = manifest
        expected = "a data manifest must be canonical JSON data"
        cause = ValueError
    events, bindings = _install(monkeypatch, variant, {"q": manifest})

    with pytest.raises(ValueError) as refused:
        _emit(variant, [{"quantum_id": "q"}])

    assert str(refused.value) == expected
    if cause is None:
        assert refused.value.__cause__ is None
    else:
        assert type(refused.value.__cause__) is cause
    assert [event[0] for event in events] == ["build", "seal"]
    assert bindings == []
