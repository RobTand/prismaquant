"""Offline structure diagnostics are not content or serving qualification."""
import json
from pathlib import Path
import struct

import pytest

from prismaquant.export_structure import main, verify_export_structure


def _shard(root, name, tensors):
    header = {}
    offset = 0
    for tensor, size in tensors.items():
        header[tensor] = {"dtype": "U8", "shape": [size],
                          "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    path = root / name
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))
    return path


def _index(root, weight_map, size=5):
    path = root / "model.safetensors.index.json"
    path.write_text(json.dumps({"metadata": {"total_size": size},
                                "weight_map": weight_map}))
    return path


def _artifact(root):
    a = _shard(root, "a.safetensors", {"a": 2})
    b = _shard(root, "b.safetensors", {"b": 3})
    _index(root, {"a": a.name, "b": b.name})
    return a, b


def test_indexed_data_size(tmp_path):
    a, b = _artifact(tmp_path)
    result = verify_export_structure(tmp_path, expect_shards=2, index_size_basis="data")
    assert result["status"] == "structure_ok"
    assert result["shards"] == 2
    assert result["tensors"] == 2
    assert result["tensor_data_bytes"] == 5
    assert result["shard_file_bytes"] == a.stat().st_size + b.stat().st_size
    assert result["header_bytes"] == result["shard_file_bytes"] - 5
    assert result["index_size_basis"] == "data"


def test_physical_size_is_explicit(tmp_path):
    a, b = _artifact(tmp_path)
    _index(tmp_path, {"a": a.name, "b": b.name}, a.stat().st_size + b.stat().st_size)
    assert verify_export_structure(tmp_path, index_size_basis="file")["shards"] == 2
    with pytest.raises(ValueError, match="total_size"):
        verify_export_structure(tmp_path, index_size_basis="data")
    assert verify_export_structure(tmp_path)["index_size_basis"] is None


def test_single_file_and_empty_tensor(tmp_path):
    _shard(tmp_path, "model.safetensors", {"empty": 0, "a": 2})
    result = verify_export_structure(tmp_path, expect_shards=1)
    assert result["tensors"] == 2
    with pytest.raises(ValueError, match="requires an index"):
        verify_export_structure(tmp_path, index_size_basis="file")


@pytest.mark.parametrize("change,error", [
    ("missing", "shard roster"), ("extra", "shard roster"),
    ("wrong_shard", "placed"), ("missing_tensor", "index tensor roster"),
    ("extra_tensor", "placed"), ("duplicate_tensor", "placed"),
    ("count", "expected 3 shards"), ("truncated", "data span"),
    ("trailing", "data area"),
])
def test_rejects_inconsistent_artifact(tmp_path, change, error):
    a, b = _artifact(tmp_path)
    kwargs = {}
    if change == "missing":
        b.unlink()
    elif change == "extra":
        _shard(tmp_path, "extra.safetensors", {"x": 1})
    elif change == "wrong_shard":
        _index(tmp_path, {"a": b.name, "b": a.name})
    elif change == "missing_tensor":
        _index(tmp_path, {"a": a.name, "b": b.name, "ghost": b.name})
    elif change == "extra_tensor":
        _shard(tmp_path, a.name, {"a": 2, "extra": 1})
    elif change == "duplicate_tensor":
        _shard(tmp_path, b.name, {"a": 2, "b": 3})
    elif change == "count":
        kwargs["expect_shards"] = 3
    elif change == "truncated":
        a.write_bytes(a.read_bytes()[:-1])
    elif change == "trailing":
        a.write_bytes(a.read_bytes() + b"x")
    with pytest.raises(ValueError, match=error):
        verify_export_structure(tmp_path, **kwargs)


@pytest.mark.parametrize("name", ["../a.safetensors", "/a.safetensors",
                                  "sub/a.safetensors", "sub\\a.safetensors", "a.bin", ""])
def test_rejects_unsafe_shard_names(tmp_path, name):
    _index(tmp_path, {"a": name})
    with pytest.raises(ValueError, match="shard name"):
        verify_export_structure(tmp_path)


@pytest.mark.parametrize("raw", [b"[]", b'{"weight_map":{}}',
    b'{"weight_map":{"a":"a.safetensors","a":"b.safetensors"}}',
    b'{"weight_map":{"a":"a.safetensors"},"metadata":{"total_size":NaN}}',
    b'{"weight_map":{"a":null}}', b'{"weight_map":[]}', b'{'])
def test_rejects_invalid_index(tmp_path, raw):
    (tmp_path / "model.safetensors.index.json").write_bytes(raw)
    with pytest.raises(ValueError):
        verify_export_structure(tmp_path)


@pytest.mark.parametrize("raw", [b"", struct.pack("<Q", 10) + b"{}",
    struct.pack("<Q", 2**63), struct.pack("<Q", 2) + b"[]",
    struct.pack("<Q", 2) + b"{}"])
def test_rejects_invalid_header(tmp_path, raw):
    (tmp_path / "model.safetensors").write_bytes(raw)
    with pytest.raises(ValueError):
        verify_export_structure(tmp_path)


def test_reuses_geometry_validator(tmp_path):
    raw = json.dumps({"a": {"dtype": "U8", "shape": [1],
                            "data_offsets": [1, 2]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"xx")
    with pytest.raises(ValueError, match="gap"):
        verify_export_structure(tmp_path)


@pytest.mark.parametrize("target", ["shard", "index"])
def test_rejects_symlinks(tmp_path, target):
    a, _ = _artifact(tmp_path)
    path = a if target == "shard" else tmp_path / "model.safetensors.index.json"
    backing = tmp_path / "backing"
    path.rename(backing)
    path.symlink_to(backing)
    with pytest.raises(ValueError, match="regular file"):
        verify_export_structure(tmp_path)


def test_does_not_read_tensor_payload(tmp_path, monkeypatch):
    a, b = _artifact(tmp_path)
    original = Path.open
    reads = []

    class HeaderOnly:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.handle.close()
        def fileno(self):
            return self.handle.fileno()
        def read(self, length=-1):
            pos = self.handle.tell()
            assert length >= 0
            if pos == 0:
                assert length == 8
            else:
                assert pos == 8  # exactly one bounded header read, no data read
            reads.append(length)
            return self.handle.read(length)

    def guarded(path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        if path.suffix == ".safetensors":
            assert kwargs["buffering"] == 0
            return HeaderOnly(handle)
        return handle

    monkeypatch.setattr(Path, "open", guarded)
    result = verify_export_structure(tmp_path)
    assert len(reads) == 4
    assert sum(reads) == result["header_bytes"]
    assert sum(reads) == a.stat().st_size + b.stat().st_size - 5


def test_cli_json_and_failure(tmp_path, capsys):
    _artifact(tmp_path)
    assert main([str(tmp_path), "--index-size-basis", "data"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "structure_ok"
    assert main([str(tmp_path), "--expect-shards", "3"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "structure_error"


@pytest.mark.parametrize("size", [None, True, -1, "5", 5.0])
def test_size_comparison_requires_nonnegative_integer(tmp_path, size):
    a, b = _artifact(tmp_path)
    _index(tmp_path, {"a": a.name, "b": b.name}, size)
    with pytest.raises(ValueError, match="total_size"):
        verify_export_structure(tmp_path, index_size_basis="data")


@pytest.mark.parametrize("kwargs", [{"expect_shards": 0}, {"expect_shards": True},
                                   {"index_size_basis": "guess"}])
def test_rejects_invalid_options(tmp_path, kwargs):
    with pytest.raises(ValueError):
        verify_export_structure(tmp_path, **kwargs)


def test_detects_changed_shard(tmp_path, monkeypatch):
    from prismaquant import export_structure
    a, _ = _artifact(tmp_path)
    original = export_structure.safetensors_header_spans

    def mutate(raw, **kwargs):
        spans = original(raw, **kwargs)
        with a.open("ab") as handle:
            handle.write(b"x")
        return spans

    monkeypatch.setattr(export_structure, "safetensors_header_spans", mutate)
    with pytest.raises(ValueError, match="changed during structure check"):
        verify_export_structure(tmp_path)


def test_malformed_dtype_is_clean_cli_failure(tmp_path, capsys):
    raw = json.dumps({"a": {"dtype": [], "shape": [1],
                            "data_offsets": [0, 1]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"x")
    assert main([str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "structure_error"
