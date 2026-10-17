"""Tests for the v25 per-layer export cache (resume support).

These tests exercise cache-file conventions with local stand-ins and the
production activation-index fingerprint. They do not drive the streaming
layer loop.

The actual quantization math is covered by the per-Linear and batched
GPTQ equivalence tests; this file is purely about the resume
mechanics.
"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _write_layer_cache(cache_dir: Path, layer_idx: int, payload: dict) -> Path:
    """Helper: torch.save a fake per-layer dict the way the export
    streamer does (atomic via .tmp + rename)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"layer_{layer_idx:03d}.pt"
    tmp = target.with_suffix(".pt.tmp")
    torch.save(payload, str(tmp))
    tmp.rename(target)
    return target


def test_layer_cache_file_naming(tmp_path):
    """Cache files use 3-digit zero-padded layer indices so lexicographic
    sort matches numeric order — important for the scan-on-restart
    pattern."""
    expected = [
        f"layer_{i:03d}.pt"
        for i in (0, 1, 9, 10, 61, 100)
    ]
    for i in (0, 1, 9, 10, 61, 100):
        path = tmp_path / f"layer_{i:03d}.pt"
        torch.save({"dummy": torch.zeros(1)}, str(path))
        assert path.exists()
    actual = sorted(p.name for p in tmp_path.iterdir())
    # 0,1,9,10,61,100 should sort as 000,001,009,010,061,100 — same as
    # expected order.
    assert actual == sorted(expected)


def test_atomic_write_no_partial_visible(tmp_path):
    """Verify the .pt.tmp + rename pattern: a half-written cache file
    never appears as `layer_NNN.pt` while torch.save is in flight.
    We can't directly observe in-flight bytes from Python, but we can
    confirm the rename pattern is what the streamer uses."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    payload = {"weight_packed": torch.zeros(8, dtype=torch.int8)}
    final_path = _write_layer_cache(cache_dir, 5, payload)

    # After the helper returns, only the final file exists; no .tmp.
    files = sorted(p.name for p in cache_dir.iterdir())
    assert files == ["layer_005.pt"]
    assert not (cache_dir / "layer_005.pt.tmp").exists()

    # Reload to verify the contents survived.
    loaded = torch.load(str(final_path), weights_only=False)
    assert "weight_packed" in loaded
    assert torch.equal(loaded["weight_packed"], payload["weight_packed"])


def test_resume_skips_cached_layers(tmp_path):
    """Mimic the streamer's resume-detection logic. With cache files
    present for layers 0-4, the loop should consider those cached and
    only quantize 5-7."""
    cache_dir = tmp_path / "cache"
    for L in range(5):
        _write_layer_cache(cache_dir, L, {"k": torch.tensor([float(L)])})

    # Replicate the streamer's check pattern.
    num_layers = 8
    cached_layers = []
    quantize_layers = []
    for L in range(num_layers):
        cf = cache_dir / f"layer_{L:03d}.pt"
        if cf.exists():
            cached_layers.append(L)
        else:
            quantize_layers.append(L)

    assert cached_layers == [0, 1, 2, 3, 4]
    assert quantize_layers == [5, 6, 7]


def test_resume_replays_cached_payload_to_sink(tmp_path):
    """The streamer's hot path: when a layer's cache file exists,
    torch.load it and feed to tensor_sink. Verify shape/values
    survive the round-trip."""
    cache_dir = tmp_path / "cache"
    payload = {
        "model.layers.0.self_attn.q_proj.weight_packed": torch.randint(
            0, 16, (32, 16), dtype=torch.uint8),
        "model.layers.0.self_attn.q_proj.weight_scale": torch.randn(32, 1),
        "model.layers.0.self_attn.q_proj.weight_global_scale": torch.tensor([1.0]),
    }
    _write_layer_cache(cache_dir, 0, payload)

    # Replay
    sink_calls = []

    def sink(d: dict):
        sink_calls.append(dict(d))

    cf = cache_dir / "layer_000.pt"
    cached = torch.load(str(cf), weights_only=False)
    sink(cached)

    assert len(sink_calls) == 1
    got = sink_calls[0]
    assert set(got.keys()) == set(payload.keys())
    for k, v in payload.items():
        assert torch.equal(got[k], v), f"mismatch on {k}"


def test_orphan_tmp_files_ignored_on_resume(tmp_path):
    """If a kill happened mid-write, a stale .pt.tmp may sit beside a
    valid .pt for some other layer. The resume scan must only look at
    `layer_NNN.pt` files, not the .tmp variants."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Cache layer 0 cleanly.
    _write_layer_cache(cache_dir, 0, {"k": torch.zeros(1)})

    # Simulate a killed write on layer 1: only the .tmp exists.
    torch.save({"partial": torch.zeros(1)}, str(cache_dir / "layer_001.pt.tmp"))

    # Resume scan: only layer_000.pt is recognized as done.
    cached = sorted(
        int(p.stem.split("_")[1])
        for p in cache_dir.glob("layer_*.pt")
        if not p.name.endswith(".pt.tmp")
    )
    assert cached == [0], (
        f"orphan .tmp leaked into resume scan: {sorted(p.name for p in cache_dir.iterdir())}")


def test_export_cache_disabled_when_cache_dir_none(tmp_path):
    """When --export-cache-dir is not passed, the streamer's cache
    helper returns None and no per-layer files are written. This
    test just exercises the logic that the streamer uses to decide
    whether to engage the cache path."""
    cache_path = None  # equivalent to args.export_cache_dir being unset

    def _layer_cache_file(L: int):
        return None if cache_path is None else cache_path / f"layer_{L:03d}.pt"

    assert _layer_cache_file(0) is None
    assert _layer_cache_file(61) is None


def test_activation_cache_fingerprint_tracks_cache_identity(tmp_path):
    from prismaquant.export_native_compressed import (
        _activation_index_fingerprint,
    )

    cache_dir = tmp_path / "act"
    cache_dir.mkdir()
    fp = cache_dir / "layer__q_proj.pt"
    torch.save({"inputs": torch.zeros(2, 3)}, str(fp))

    class _Index:
        _paths = {"layer.q_proj": fp}

    first = _activation_index_fingerprint(_Index(), cache_dir)
    torch.save({"inputs": torch.zeros(5, 3)}, str(fp))
    second = _activation_index_fingerprint(_Index(), cache_dir)

    assert first["path"] == second["path"]
    assert first["n_files"] == 1
    assert second["n_files"] == 1
    assert first["hash"] != second["hash"]
