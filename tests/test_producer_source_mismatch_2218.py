"""A producer-source mismatch names the first file that moved (#2218, option c).

``_production_cache_source_sha256`` hashes the whole package tree, so a
checkpoint's ``producer_source_sha256`` refusal was two opaque tree digests.
The manifest now also records the per-file digests of the tree that wrote it,
and the mismatch refusal names the first differing relative path with both
digests -- a recurrence of the xdist flake is diagnosable from the refusal
alone. The digest itself is never patched here: the helpers run for real on a
synthetic tree through the established default-root seam.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import prismaquant.aura_cost as aura
from prismaquant import production_weight_cache as pwc


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "package"
    (root / "sub").mkdir(parents=True)
    (root / "__init__.py").write_bytes(b"# package\n")
    (root / "profiles.json").write_bytes(b'{"format": "BF16"}\n')
    (root / "sub" / "renderer.py").write_bytes(b"RENDER = 1\n")
    (root / "__pycache__").mkdir(parents=True)
    (root / "__pycache__" / "stash.pyc").write_bytes(b"interpreter-only bytes")
    return root


def _identity(digest: str) -> dict:
    return {"schema": aura.AURA_CHECKPOINT_IDENTITY_SCHEMA,
            "producer_source_sha256": digest}


def _point_default_root_at(monkeypatch, root: Path) -> None:
    # The established seam (tests/test_pwc_source_framing_1764.py): the digest
    # helpers resolve their roster from their own module's package directory.
    monkeypatch.setattr(pwc, "__file__", str(root / "__init__.py"))


def _write_manifest(checkpoint: Path, digest: str) -> None:
    aura._write_aura_checkpoint_manifest(checkpoint, _identity(digest), ["unit.a"])


def test_changed_file_is_named_with_both_per_file_digests(tmp_path, monkeypatch):
    root = _package(tmp_path)
    _point_default_root_at(monkeypatch, root)
    stored_digest = pwc._production_cache_source_sha256()
    checkpoint = tmp_path / "ckpt"
    _write_manifest(checkpoint, stored_digest)

    (root / "sub" / "renderer.py").write_bytes(b"RENDER = 2\n")
    current_digest = pwc._production_cache_source_sha256()
    assert current_digest != stored_digest

    with pytest.raises(RuntimeError) as refused:
        aura._load_aura_checkpoint_manifest(checkpoint, _identity(current_digest))
    message = str(refused.value)
    assert "producer_source_sha256" in message
    assert "sub/renderer.py" in message
    assert hashlib.sha256(b"RENDER = 1\n").hexdigest() in message
    assert hashlib.sha256(b"RENDER = 2\n").hexdigest() in message


@pytest.mark.parametrize("mutation,expected", [
    ("modified", ("profiles.json",
                  hashlib.sha256(b'{"format": "BF16"}\n').hexdigest(),
                  hashlib.sha256(b'{"format": "FP8"}\n').hexdigest())),
    ("added", ("sub/extra.py", "<missing>",
               hashlib.sha256(b"EXTRA = 1\n").hexdigest())),
    ("removed", ("sub/renderer.py",
                 hashlib.sha256(b"RENDER = 1\n").hexdigest(), "<missing>")),
])
def test_any_tree_mutation_names_the_first_differing_path(
        tmp_path, monkeypatch, mutation, expected):
    root = _package(tmp_path)
    _point_default_root_at(monkeypatch, root)
    checkpoint = tmp_path / "ckpt"
    _write_manifest(checkpoint, pwc._production_cache_source_sha256())

    if mutation == "modified":
        (root / "profiles.json").write_bytes(b'{"format": "FP8"}\n')
    elif mutation == "added":
        (root / "sub" / "extra.py").write_bytes(b"EXTRA = 1\n")
    else:
        (root / "sub" / "renderer.py").unlink()

    with pytest.raises(RuntimeError) as refused:
        aura._load_aura_checkpoint_manifest(
            checkpoint, _identity(pwc._production_cache_source_sha256()))
    message = str(refused.value)
    assert "producer_source_sha256" in message
    assert expected[0] in message
    assert f"stored={expected[1]}" in message
    assert f"current={expected[2]}" in message


def test_tree_changed_between_identity_and_manifest_write_is_named(
        tmp_path, monkeypatch):
    # The manifest write is one pass over the tree, later than the identity's
    # digest: when that pass's per-file map still matches the executing tree
    # but the identity aggregate differs, the refusal says when the tree moved.
    root = _package(tmp_path)
    _point_default_root_at(monkeypatch, root)
    identity = _identity(pwc._production_cache_source_sha256())
    (root / "profiles.json").write_bytes(b'{"format": "FP8"}\n')
    checkpoint = tmp_path / "ckpt"
    _write_manifest(checkpoint, identity["producer_source_sha256"])

    with pytest.raises(RuntimeError) as refused:
        aura._load_aura_checkpoint_manifest(
            checkpoint, _identity(pwc._production_cache_source_sha256()))
    message = str(refused.value)
    assert "producer_source_sha256" in message
    assert "the producer tree changed between identity and manifest write" in message
    assert "first differing source file" not in message


def test_manifest_without_the_listing_keeps_the_plain_refusal(tmp_path, monkeypatch):
    # A manifest written before the per-file listing existed still refuses with
    # exactly the old message: the diagnostic is additive, never a gate.
    root = _package(tmp_path)
    _point_default_root_at(monkeypatch, root)
    checkpoint = tmp_path / "ckpt"
    _write_manifest(checkpoint, pwc._production_cache_source_sha256())
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    manifest.pop("producer_source_files_sha256", None)
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))

    (root / "profiles.json").write_bytes(b'{"format": "FP8"}\n')
    with pytest.raises(RuntimeError) as refused:
        aura._load_aura_checkpoint_manifest(
            checkpoint, _identity(pwc._production_cache_source_sha256()))
    message = str(refused.value)
    assert "producer_source_sha256" in message
    assert "first differing source file" not in message
