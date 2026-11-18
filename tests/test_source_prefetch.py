import json

import pytest

from prismaquant.source_prefetch import prefetch_safetensors_checkpoint


def test_source_prefetch_requires_local_safetensors(tmp_path):
    with pytest.raises(RuntimeError, match="no local safetensors"):
        prefetch_safetensors_checkpoint(tmp_path, mode="require", progress=False)


def test_source_prefetch_reads_unique_index_shards(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"a" * 17)
    (tmp_path / "b.safetensors").write_bytes(b"b" * 19)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({
            "weight_map": {
                "model.layers.0.weight": "a.safetensors",
                "model.layers.1.weight": "a.safetensors",
                "model.layers.2.weight": "b.safetensors",
            }
        })
    )

    stats = prefetch_safetensors_checkpoint(
        tmp_path,
        mode="require",
        max_resident_bytes=1024,
        workers=2,
        chunk_mb=1,
        progress=False,
    )

    assert stats["shards"] == 2
    assert stats["bytes"] == 36
    assert stats["prefetched_bytes"] == 36
    assert stats["skipped"] is False


def test_source_prefetch_uses_sysconf_memory_when_proc_is_unavailable(tmp_path, monkeypatch):
    from prismaquant import source_prefetch

    path = tmp_path / "model.safetensors"
    payload = b"checkpoint\x00bytes"
    path.write_bytes(payload)
    calls = []

    def unavailable():
        raise OSError

    def sysconf(name):
        calls.append(name)
        return {"SC_AVPHYS_PAGES": 8 * 1024**2, "SC_PAGE_SIZE": 4096}[name]

    monkeypatch.setattr(source_prefetch.io_spans, "mem_available_bytes", unavailable)
    monkeypatch.setattr(source_prefetch.os, "sysconf", sysconf)
    stats = prefetch_safetensors_checkpoint(tmp_path, workers=1, chunk_mb=1, progress=False)

    assert calls == ["SC_AVPHYS_PAGES", "SC_PAGE_SIZE"]
    assert stats["available_bytes"] == 32 * 1024**3
    assert stats["max_resident_bytes"] == 16 * 1024**3
    assert stats["prefetched_bytes"] == stats["bytes"] == 16
    assert stats["shards"] == 1
    assert stats["skipped"] is False
    assert path.read_bytes() == payload


def test_source_prefetch_budget_fail_fast(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x" * 32)

    with pytest.raises(RuntimeError, match="budget"):
        prefetch_safetensors_checkpoint(
            tmp_path,
            mode="require",
            max_resident_bytes=16,
            progress=False,
        )



@pytest.mark.parametrize("available", [None, 0, 4 * 1024**3, 16 * 1024**3])
@pytest.mark.parametrize("mode", ["auto", "require"])
def test_exhausted_or_unknown_auto_budget_never_starts_file_reads(tmp_path, monkeypatch, available, mode):
    from prismaquant import source_prefetch

    (tmp_path / "model.safetensors").write_bytes(b"checkpoint")
    monkeypatch.setattr(source_prefetch, "_available_memory_bytes", lambda: available)
    reads = []
    monkeypatch.setattr(source_prefetch, "_read_file_to_page_cache",
                        lambda path, **kwargs: reads.append(path) or path.stat().st_size)
    if mode == "require":
        with pytest.raises(RuntimeError, match="budget"):
            prefetch_safetensors_checkpoint(tmp_path, mode=mode, progress=False)
    else:
        stats = prefetch_safetensors_checkpoint(tmp_path, mode=mode, progress=False)
        assert stats["skipped"] and stats["max_resident_bytes"] == 0
        assert stats["prefetched_bytes"] == 0
    assert reads == []


def test_source_prefetch_refuses_invalid_weight_map_before_reads(tmp_path, monkeypatch):
    from prismaquant import source_prefetch

    reads = []
    monkeypatch.setattr(source_prefetch, "_read_file_to_page_cache",
                        lambda path, **kwargs: reads.append(path) or 0)
    index = tmp_path / "model.safetensors.index.json"
    for weight_map in (None, [], "a.safetensors"):
        index.write_text(json.dumps({"weight_map": weight_map}))
        index_bytes = index.read_bytes()
        paths_before = sorted(tmp_path.rglob("*"))
        with pytest.raises(ValueError) as excinfo:
            prefetch_safetensors_checkpoint(tmp_path, mode="require", progress=False)
        assert str(excinfo.value) == f"invalid safetensors index: {index}"
        assert reads == []
        assert index.read_bytes() == index_bytes
        assert sorted(tmp_path.rglob("*")) == paths_before


def test_read_ahead_keeps_about_eight_files_ahead_and_starts_each_once():
    from prismaquant import source_prefetch

    started = []
    submit = lambda fn, paths: started.extend(paths)  # noqa: E731
    shard = {f"model.layers.{L}.w{i}": f"s{2 * L + i}" for L in range(10) for i in range(2)}
    seen: set = set()
    source_prefetch.read_ahead_layer_files(shard, "model.layers.", range(1, 10), seen, submit)
    assert started == [f"s{n}" for n in range(2, 10)]
    source_prefetch.read_ahead_layer_files(shard, "model.layers.", range(2, 10), seen, submit)
    assert started[8:] == ["s10", "s11"]
