"""Auxiliary source identity hashes use declared staged metadata bytes."""
from __future__ import annotations

import hashlib

import pytest

from prismaquant import cost_streaming, staged_whole_file
from prismaquant.staged_tier_policy import TierPolicyRefused
from test_streamed_metadata_staged_reads import (
    _activate, _deny_pool_opens, checkpoint, _forget_state,  # noqa: F401
)

pytestmark = pytest.mark.own_process


@pytest.mark.parametrize("name", ["config.json", "model.safetensors.index.json"])
def test_auxiliary_digest_never_opens_pool(checkpoint, tmp_path, monkeypatch, name):
    root, _config, _index, manifest = checkpoint
    path = root / name
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    _activate(tmp_path, monkeypatch, manifest)
    _deny_pool_opens(monkeypatch, root)
    assert cost_streaming._source_checkpoint_metadata_sha256(path) == expected


def test_digest_cache_hit_does_not_bypass_staged_auxiliary_reads(checkpoint, tmp_path, monkeypatch):
    root, _config, _index, manifest = checkpoint
    cache = tmp_path / "source-digests.json"
    expected = cost_streaming.build_source_checkpoint_identity(root, digest_cache_path=cache)
    _activate(tmp_path, monkeypatch, manifest)
    _deny_pool_opens(monkeypatch, root)
    calls = []

    def no_shard_rehash(paths):
        calls.append(paths)
        assert paths == []
        return []

    monkeypatch.setattr(cost_streaming, "_hash_source_shards", no_shard_rehash)
    actual = cost_streaming.build_source_checkpoint_identity(root, digest_cache_path=cache)
    assert actual == expected
    assert calls == [[]]


def test_unbound_root_module_refuses_instead_of_hashing_pool(checkpoint, tmp_path, monkeypatch):
    root, _config, _index, manifest = checkpoint
    (root / "modeling_fixture.py").write_bytes(b"# auxiliary model source\n")
    cache = tmp_path / "source-digests.json"
    cost_streaming.build_source_checkpoint_identity(root, digest_cache_path=cache)
    _activate(tmp_path, monkeypatch, manifest)
    _deny_pool_opens(monkeypatch, root)
    with pytest.raises(TierPolicyRefused, match="metadata-readset-not-staged"):
        cost_streaming.build_source_checkpoint_identity(root, digest_cache_path=cache)


def test_inactive_digest_preserves_raw_binary_newlines(tmp_path):
    path = tmp_path / "source.py"
    raw = b"a\r\nb\r\n\xff"
    path.write_bytes(raw)
    assert cost_streaming._source_checkpoint_metadata_sha256(path) == hashlib.sha256(raw).hexdigest()


def test_staged_digest_does_not_decode_metadata(checkpoint, tmp_path, monkeypatch):
    root, _config, _index, manifest = checkpoint
    path = root / "config.json"
    raw = b"not JSON\r\n\xff"
    path.write_bytes(raw)
    manifest["entries"][0].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    _activate(tmp_path, monkeypatch, manifest)
    _deny_pool_opens(monkeypatch, root)
    assert cost_streaming._source_checkpoint_metadata_sha256(path) == hashlib.sha256(raw).hexdigest()
