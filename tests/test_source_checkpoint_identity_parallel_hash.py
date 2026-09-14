"""Uncached source shards are hashed concurrently, with serial semantics.

`build_source_checkpoint_identity` reads every shard it has no digest-cache
entry for. On a ~185 GB checkpoint the serial loop left all but one core idle,
so cache misses now go through a bounded thread pool. These tests hold the
pool to the serial contract: the same identity dict and digest-cache bytes,
the "changed while hashing" refusal, and the first failing shard in sorted
order named on every run.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("torch")

from prismaquant import cost_streaming  # noqa: E402
from prismaquant.cost_streaming import build_source_checkpoint_identity  # noqa: E402

_SHARDS = 6


def _write_checkpoint(root: Path, shards: int = _SHARDS) -> list[Path]:
    """A multi-shard checkpoint whose shards differ in size and content.

    The identity binds file bytes and never parses safetensors, so plain
    bytes are a faithful fixture and keep the test off the torch loader.
    """
    root.mkdir(parents=True)
    names = [f"model-{i + 1:05d}-of-{shards:05d}.safetensors" for i in range(shards)]
    paths = []
    for i, name in enumerate(names):
        path = root / name
        # Several 16 MiB read blocks for some shards, so the hash spans
        # multiple `update` calls, and distinct bytes per shard.
        size = (i + 1) * 5 * 1024 * 1024 + i
        path.write_bytes(bytes([i + 1]) * size)
        paths.append(path)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"t{i}": name for i, name in enumerate(names)}}),
        encoding="utf-8",
    )
    (root / "config.json").write_text('{"model_type": "fixture"}', encoding="utf-8")
    return paths


def _identity(root: Path, monkeypatch, threads: int, cache: Path | None = None):
    monkeypatch.setenv("PRISMAQUANT_SOURCE_HASH_THREADS", str(threads))
    return build_source_checkpoint_identity(root, digest_cache_path=cache)


def test_parallel_identity_equals_forced_serial(tmp_path, monkeypatch):
    root = tmp_path / "ckpt"
    _write_checkpoint(root)
    serial_cache = tmp_path / "serial_cache.json"
    parallel_cache = tmp_path / "parallel_cache.json"

    serial = _identity(root, monkeypatch, 1, serial_cache)
    parallel = _identity(root, monkeypatch, 4, parallel_cache)

    assert parallel == serial
    assert json.dumps(parallel, sort_keys=True) == json.dumps(serial, sort_keys=True)
    assert [row["name"] for row in parallel["shards"]] == sorted(
        row["name"] for row in serial["shards"]
    )
    assert len(parallel["shards"]) == _SHARDS
    assert parallel_cache.read_bytes() == serial_cache.read_bytes()


def test_partial_cache_hits_merge_in_order(tmp_path, monkeypatch):
    """Hits and misses interleave; misses fill their own slots."""
    root = tmp_path / "ckpt"
    paths = _write_checkpoint(root)
    cache = tmp_path / "cache.json"
    reference = _identity(root, monkeypatch, 1, cache)

    payload = json.loads(cache.read_text(encoding="utf-8"))
    payload["entries"] = payload["entries"][::2]  # keep every other shard
    cache.write_text(json.dumps(payload), encoding="utf-8")
    hashed: list[str] = []
    real = cost_streaming._file_sha256

    def recording(path):
        hashed.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(cost_streaming, "_file_sha256", recording)
    assert _identity(root, monkeypatch, 4, cache) == reference
    shard_reads = sorted(n for n in hashed if n.endswith(".safetensors"))
    assert shard_reads == [p.name for p in paths[1::2]]


def test_parallel_path_really_runs_concurrently(tmp_path, monkeypatch):
    """Two shard reads must be in flight at once; a serial loop would break
    the barrier instead of passing it."""
    root = tmp_path / "ckpt"
    _write_checkpoint(root)
    barrier = threading.Barrier(2, timeout=30)
    real = cost_streaming._file_sha256

    def gated(path):
        if Path(path).name.endswith(".safetensors"):
            barrier.wait()
        return real(path)

    monkeypatch.setattr(cost_streaming, "_file_sha256", gated)
    _identity(root, monkeypatch, 2)


def test_shard_mutated_mid_hash_is_refused_in_parallel(tmp_path, monkeypatch):
    root = tmp_path / "ckpt"
    paths = _write_checkpoint(root)
    target = paths[3]
    real = cost_streaming._file_sha256

    def mutating(path):
        digest = real(path)
        if Path(path) == target.resolve():
            # A same-size in-place rewrite: only ctime/mtime can tell.
            with target.open("r+b") as handle:
                handle.write(b"\xff")
        return digest

    monkeypatch.setattr(cost_streaming, "_file_sha256", mutating)
    with pytest.raises(RuntimeError, match="changed while hashing") as excinfo:
        _identity(root, monkeypatch, 4, tmp_path / "cache.json")
    assert target.name in str(excinfo.value)
    assert not (tmp_path / "cache.json").exists()


def test_first_failure_in_sorted_order_is_raised(tmp_path, monkeypatch):
    """The later shard finishes (and fails) first; the earlier one still wins."""
    root = tmp_path / "ckpt"
    paths = _write_checkpoint(root)
    early, late = paths[1].resolve(), paths[4].resolve()
    late_failed = threading.Event()
    real = cost_streaming._file_sha256

    def failing(path):
        path = Path(path)
        if path == early:
            assert late_failed.wait(30)
            raise OSError(f"early read failed: {path.name}")
        if path == late:
            try:
                raise OSError(f"late read failed: {path.name}")
            finally:
                late_failed.set()
        return real(path)

    monkeypatch.setattr(cost_streaming, "_file_sha256", failing)
    for _ in range(3):
        late_failed.clear()
        with pytest.raises(OSError, match="early read failed"):
            _identity(root, monkeypatch, _SHARDS)


def test_thread_override_and_default(monkeypatch):
    import os

    monkeypatch.setenv("PRISMAQUANT_SOURCE_HASH_THREADS", "3")
    assert cost_streaming.source_identity_hash_threads() == 3
    monkeypatch.setenv("PRISMAQUANT_SOURCE_HASH_THREADS", "0")
    assert cost_streaming.source_identity_hash_threads() == 1
    monkeypatch.delenv("PRISMAQUANT_SOURCE_HASH_THREADS")
    assert cost_streaming.source_identity_hash_threads() == len(os.sched_getaffinity(0))
