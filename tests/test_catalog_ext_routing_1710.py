"""PQ #1710: catalog-extension fence hashes delegate to digests owners.

Routing: _rehash_drifted + _verify_overlay_payload must call
file_digest_sha256hex; _check_capture must call sorted_newline_utf8_sha256;
create_extension (driven with stubbed pair/header checks) must call
indent2_json_file_bytes + bytes_sha256hex. All _same comparisons stay with
identical refusal type/text. Values: byte-identical to the verbatim old
spellings.
"""
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant import joint_catalog_extension as cext
from prismaquant.digests import (
    bytes_sha256hex,
    file_digest_sha256hex,
    indent2_json_file_bytes,
)
from prismaquant.joint_catalog_extension import catalog_stat_fence


def _spy(monkeypatch, name):
    calls = []
    real = getattr(cext, name, None)

    def stand_in(value):
        calls.append(value)
        return real(value) if real is not None else None

    monkeypatch.setattr(cext, name, stand_in, raising=False)
    return calls


def _bound(path):
    raw = Path(path).read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def test_fence_hashes_route_to_file_owner(monkeypatch, tmp_path):
    file_calls = _spy(monkeypatch, "file_digest_sha256hex")
    wire = tmp_path / "wire.bin"
    wire.write_bytes(b"\x00\x01catalog-fence")
    blob_sha = hashlib.sha256(b"\x00\x01catalog-fence").hexdigest()
    before = catalog_stat_fence(wire.stat())
    assert cext._rehash_drifted(wire, before, blob_sha, "test") == blob_sha
    with pytest.raises(ValueError):
        cext._rehash_drifted(wire, before, "0" * 64, "test")
    render = tmp_path / "render.bin"
    render.write_bytes(b"\x02\x03catalog-render")
    render_sha = hashlib.sha256(b"\x02\x03catalog-render").hexdigest()
    observed = {"wire": (wire, before, before),
                "render": (render, None, catalog_stat_fence(render.stat()))}
    assert cext._verify_overlay_payload(observed, blob_sha, rehash=False,
                                        hash_render=True) == render_sha
    assert len(file_calls) == 4


def test_create_extension_envelope_routes_to_owners(monkeypatch, tmp_path):
    dumps_calls = _spy(monkeypatch, "indent2_json_file_bytes")
    hash_calls = _spy(monkeypatch, "bytes_sha256hex")
    monkeypatch.setattr(cext, "verify_catalog_pair", lambda inputs: {"stub": True})
    monkeypatch.setattr(cext, "_run_header",
                        lambda doc: {"run_identity": {"campaign_scope": {"x": 1}}})
    monkeypatch.setattr(cext, "_check_capture", lambda h, i, p: {"plan": True})
    capture = tmp_path / "capture.json"
    capture.write_bytes(json.dumps({"run_identity": {}}, sort_keys=True).encode())
    prepared = tmp_path / "prepared.json"
    prepared.write_bytes(json.dumps({"stub": True}, sort_keys=True).encode())
    output = tmp_path / "extension.json"
    bound = cext.create_extension(inputs={"a": 1, "original_prepared": _bound(prepared)}, adjoint_capture=_bound(capture),
                                  output=output)
    raw = output.read_bytes()
    assert bound == {"path": str(output.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
    assert len(dumps_calls) == 1
    assert len(hash_calls) == 1


def test_catalog_values_match_verbatim_spellings(tmp_path):
    raw = b"\x00\x01catalog-fence"
    blob = tmp_path / "blob.bin"
    blob.write_bytes(raw)
    with blob.open("rb") as handle:
        first = file_digest_sha256hex(handle)
        handle.seek(0)
        assert first == hashlib.file_digest(handle, "sha256").hexdigest()
    names = ["q.b", "q.a"]
    # NOTE: _check_capture's roster is a keep, not a delegation: it hashes one
    # terminating newline per qname ("q.a\nq.b\n"), while the newline owners
    # use LF separators with no trailing LF. No owner matches byte-for-byte.
    document = {"b": [1, 2], "a": "x"}
    assert indent2_json_file_bytes(document) == (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    assert bytes_sha256hex(raw) == hashlib.sha256(raw).hexdigest()
