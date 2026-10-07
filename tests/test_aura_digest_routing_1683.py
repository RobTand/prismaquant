"""Check actual checkpoint bytes against the retained manifest recipe."""
from __future__ import annotations

import hashlib
import json

import pytest

from prismaquant import aura_cost as aura
from prismaquant import production_weight_cache as cache


@pytest.mark.parametrize("identity,names", [
    ({"z": 2, "a": {"unicode": "雪", "values": [None, True, -0.0]}},
     ["unit-z", "unit-雪", "unit-a"]),
    ({"empty": [], "nested": {"b": 2, "a": 1}}, []),
])
def test_manifest_publication_preserves_exact_input_bytes(
    tmp_path, monkeypatch, identity, names
):
    source_digest = "d" * 64
    source_files = {"prismaquant/z.py": "b" * 64, "prismaquant/雪.py": "a" * 64}
    monkeypatch.setattr(cache, "_production_cache_source_profile",
                        lambda: (source_digest, source_files))
    identity_bytes = json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False, allow_nan=False).encode("utf-8")
    expected_digest = hashlib.sha256(identity_bytes).hexdigest()
    manifest = {
        "schema": aura.AURA_CHECKPOINT_MANIFEST_SCHEMA,
        "identity": identity,
        "identity_sha256": expected_digest,
        "units": [{"qname": name, "file": str(aura._aura_unit_checkpoint_path(
            tmp_path, name).relative_to(tmp_path))} for name in names],
        "producer_source_sha256": source_digest,
        "producer_source_files_sha256": source_files,
    }
    expected = json.dumps(manifest, indent=2, sort_keys=True,
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    assert aura._write_aura_checkpoint_manifest(tmp_path, identity, names) == expected_digest
    published = (tmp_path / "manifest.json").read_bytes()
    assert published == expected
    assert not published.endswith(b"\n")
    assert [unit["qname"] for unit in json.loads(published)["units"]] == names
    assert list(tmp_path.iterdir()) == [tmp_path / "manifest.json"]


@pytest.mark.parametrize("bad,error", [
    (float("nan"), ValueError),
    (float("inf"), ValueError),
    ("\ud800", UnicodeEncodeError),
    (object(), TypeError),
])
def test_manifest_encoding_error_preserves_published_bytes(tmp_path, monkeypatch, bad, error):
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"retained checkpoint")
    # Invalid diagnostic data reaches the manifest encoder after identity hashing.
    monkeypatch.setattr(cache, "_production_cache_source_profile", lambda: ("d" * 64, {"bad": bad}))
    with pytest.raises(error):
        aura._write_aura_checkpoint_manifest(tmp_path, {"valid": True}, [])
    assert manifest.read_bytes() == b"retained checkpoint"
    assert list(tmp_path.iterdir()) == [manifest]
