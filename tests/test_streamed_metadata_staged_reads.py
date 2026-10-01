"""Streamed metadata is served from real PB material, never an undeclared pool."""
from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from prismaquant import cost_streaming, layer_streaming, staged_whole_file
from prismaquant.residency_map import bind_residency_manifest, residency_report
from prismaquant.staged_tier_policy import (
    TierPolicyRefused, activate_staged_tier_policy,
)
from test_stage_b_prep_staged_reads_1092 import _stage_manifest
from test_strict_reader_tier_enforcement import MANIFEST, _forget_state  # noqa: F401

pytestmark = pytest.mark.own_process


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "pool"
    root.mkdir()
    config = {"model_type": "pq_cpu_fixture", "architectures": []}
    index = {"weight_map": {"model.layers.0.proj.weight": "weights.safetensors"}}
    (root / "config.json").write_text(json.dumps(config))
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    (root / "weights.safetensors").write_bytes(b"payload is not read in this test")
    entries = []
    for name in ("config.json", "model.safetensors.index.json"):
        path = root / name
        raw = path.read_bytes()
        entries.append({"path": str(path), "offset": 0, "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()})
    return root, config, index, {"entries": entries}


def _deny_pool_opens(monkeypatch, root, *, names=(
        "config.json", "model.safetensors.index.json")):
    paths = {str(root / name) for name in names}

    def guard(original):
        def opened(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                value = os.fsdecode(os.fspath(path))
                if os.path.abspath(value) in paths:
                    raise AssertionError(f"undeclared pool metadata open: {value}")
            return original(path, *args, **kwargs)
        return opened

    monkeypatch.setattr(builtins, "open", guard(builtins.open))
    monkeypatch.setattr(io, "open", guard(io.open))
    monkeypatch.setattr(os, "open", guard(os.open))


def _activate(tmp_path, monkeypatch, manifest, *, skip=(), corrupt=()):
    _stage_manifest(tmp_path, monkeypatch, manifest, skip=skip, corrupt=corrupt)
    bind_residency_manifest(MANIFEST)
    activate_staged_tier_policy("ram,ssd")


@pytest.mark.parametrize("reader", ["json", "weight-map", "identity-index"])
def test_real_metadata_reader_never_opens_the_pool(checkpoint, tmp_path, monkeypatch, reader):
    root, config, index, manifest = checkpoint
    _activate(tmp_path, monkeypatch, manifest)
    _deny_pool_opens(monkeypatch, root)
    if reader == "json":
        assert layer_streaming._source_json(root / "config.json") == config
    elif reader == "weight-map":
        shards, names = layer_streaming._build_weight_map(str(root))
        assert shards == {"model.layers.0.proj.weight": str(root / "weights.safetensors")}
        assert names == {"model.layers.0.proj.weight": "model.layers.0.proj.weight"}
    else:
        names, paths = cost_streaming._local_checkpoint_shards(root)
        assert names == index["weight_map"]
        assert paths == [(root / "weights.safetensors").resolve()]
    report = residency_report()
    assert report is not None and report["fallbacks"] == []


@pytest.mark.parametrize("missing", ["config.json", "model.safetensors.index.json"])
def test_missing_metadata_refuses_without_pool_fallback(checkpoint, tmp_path, monkeypatch, missing):
    root, _config, _index, manifest = checkpoint
    _activate(tmp_path, monkeypatch, manifest, skip={str(root / missing)})
    _deny_pool_opens(monkeypatch, root)
    with pytest.raises(TierPolicyRefused):
        layer_streaming._source_json(root / missing)


def test_missing_declared_digest_refuses_before_material_read(checkpoint, tmp_path, monkeypatch):
    from prismaquant.residency_map import residency_resolver

    root, _config, _index, manifest = checkpoint
    _activate(tmp_path, monkeypatch, manifest)
    resolver = residency_resolver()
    assert resolver is not None
    original = resolver.staged_read

    def missing_digest(*args, **kwargs):
        staged = original(*args, **kwargs)
        assert staged is not None
        result = dict(staged)
        result.pop("sha256")
        return result

    def unexpected(*args, **kwargs):
        raise AssertionError("material read without a declared digest")

    monkeypatch.setattr(resolver, "staged_read", missing_digest)
    monkeypatch.setattr(staged_whole_file, "read_staged_entry", unexpected)
    _deny_pool_opens(monkeypatch, root)
    with pytest.raises(TierPolicyRefused, match="digest"):
        layer_streaming._source_json(root / "config.json")


def test_staged_bytes_must_match_the_declared_digest(checkpoint, tmp_path, monkeypatch):
    root, _config, _index, manifest = checkpoint
    _activate(tmp_path, monkeypatch, manifest)
    real = staged_whole_file.read_staged_entry

    def changed(*args, **kwargs):
        return real(*args, **kwargs) + b" "

    monkeypatch.setattr(staged_whole_file, "read_staged_entry", changed)
    _deny_pool_opens(monkeypatch, root)
    with pytest.raises(TierPolicyRefused, match="digest"):
        layer_streaming._source_json(root / "config.json")


def test_inactive_policy_keeps_legacy_metadata_text(checkpoint):
    root, config, index, _manifest = checkpoint
    assert layer_streaming._source_json(root / "config.json") == config
    assert cost_streaming._local_checkpoint_shards(root)[0] == index["weight_map"]


def test_capture_authenticator_still_owns_its_json_read(monkeypatch):
    class Owner:
        def read_json(self, path):
            assert path == "config.json"
            return {"authenticated": True}

    def unexpected(*args, **kwargs):
        raise AssertionError("capture owner was bypassed")

    monkeypatch.setattr(staged_whole_file, "read_source_metadata_text", unexpected)
    assert layer_streaming._source_json("config.json", Owner()) == {"authenticated": True}
