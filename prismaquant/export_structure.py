"""Offline safetensors structure diagnostic; NOT content/serving qualification.

Run on an inactive export with ``python -m prismaquant.export_structure DIR``.
Only flat, regular shard files are supported. Tensor payloads are never read.
The existing PQ environment is required by package initialization; this module
itself uses stdlib and shipcard's shared header validation only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat

from .shipcard import (
    _MAX_SAFETENSORS_HEADER_BYTES,
    _content_stat,
    _strict_json_object,
    safetensors_header_spans,
)
from .digests import DIRECT_ASCII_SPACED_LAX
from .source_read_plan import safetensors_prefix_length


def _regular_stat(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{path}: expected a regular file (no symlinks)")
    return info


def _read_metadata(path: Path, *, shard: bool) -> tuple[bytes, int, dict]:
    """Bound metadata reads and detect ordinary mutation; not an attestation."""
    before = _content_stat(_regular_stat(path))
    with path.open("rb", buffering=0) as handle:
        if _content_stat(os.fstat(handle.fileno())) != before:
            raise ValueError(f"{path}: changed before metadata read")
        length = before["bytes"]
        prefix = 0
        if shard:
            raw_length = handle.read(8)
            length = safetensors_prefix_length(
                raw_length, before["bytes"], max_bytes=_MAX_SAFETENSORS_HEADER_BYTES,
                short_error=f"{path}: truncated header length",
                range_error=lambda length: f"{path}: invalid or truncated metadata length {length}")
            prefix = 8
        elif not 0 < length <= _MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(f"{path}: invalid or truncated metadata length {length}")
        raw = handle.read(length)
        if len(raw) != length:
            raise ValueError(f"{path}: truncated metadata")
        if _content_stat(os.fstat(handle.fileno())) != before:
            raise ValueError(f"{path}: changed during metadata read")
    if _content_stat(_regular_stat(path)) != before:
        raise ValueError(f"{path}: changed after metadata read")
    return raw, prefix + length, before


def verify_export_structure(
    artifact_dir: str | Path,
    *,
    expect_shards: int | None = None,
    index_size_basis: str | None = None,
) -> dict:
    """Check physical structure without reading or authenticating tensor data.

    ``index_size_basis`` is explicit: ``data`` compares the index total to tensor
    payload bytes; ``file`` compares it to complete shard file bytes. ``None``
    leaves that producer-dependent field unchecked. No release gates are filled.
    """
    if expect_shards is not None and (type(expect_shards) is not int or expect_shards < 1):
        raise ValueError("expect_shards must be a positive integer")
    if index_size_basis not in (None, "data", "file"):
        raise ValueError("index_size_basis must be data, file or None")
    root = Path(artifact_dir)
    index = root / "model.safetensors.index.json"
    weight_map = None
    snapshots = {}
    # lexists also detects a dangling index symlink, which must not fall back.
    if os.path.lexists(index):
        raw, _, snapshots[index] = _read_metadata(index, shard=False)
        payload = _strict_json_object(raw, where=str(index))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index}: expected a nonempty weight_map")
        for tensor, name in weight_map.items():
            if not tensor or not isinstance(name, str) or not name.endswith(".safetensors") or (
                Path(name).name != name or "/" in name or "\\" in name
            ):
                raise ValueError(f"{index}: invalid tensor or shard name {tensor!r}: {name!r}")
        expected = set(weight_map.values())
    else:
        if index_size_basis is not None:
            raise ValueError("index size comparison requires an index")
        expected = {"model.safetensors"}

    actual = {p.name for p in root.iterdir() if p.name.endswith(".safetensors")}
    if actual != expected:
        raise ValueError(f"shard roster differs: missing={sorted(expected - actual)[:5]}, "
                         f"extra={sorted(actual - expected)[:5]}")
    if expect_shards is not None and len(actual) != expect_shards:
        raise ValueError(f"expected {expect_shards} shards, found {len(actual)}")

    tensors = set()
    file_bytes = header_bytes = 0
    for name in sorted(actual):
        path = root / name
        raw, metadata_bytes, snapshots[path] = _read_metadata(path, shard=True)
        size = snapshots[path]["bytes"]
        try:
            spans = safetensors_header_spans(raw, data_bytes=size - metadata_bytes, where=str(path))
        except TypeError as exc:
            raise ValueError(f"{path}: malformed tensor header: {exc}") from exc
        for _, _, tensor in spans:
            if tensor in tensors or (weight_map is not None and weight_map.get(tensor) != name):
                raise ValueError(f"{path}: tensor {tensor!r} is duplicate or not placed as indexed")
            tensors.add(tensor)
        file_bytes += size
        header_bytes += metadata_bytes
    if weight_map is not None and tensors != set(weight_map):
        raise ValueError(f"index tensor roster differs: missing={sorted(set(weight_map) - tensors)[:5]}")
    if index_size_basis is not None:
        metadata = payload.get("metadata")
        total = metadata.get("total_size") if isinstance(metadata, dict) else None
        observed = file_bytes if index_size_basis == "file" else file_bytes - header_bytes
        if type(total) is not int or total < 0 or total != observed:
            raise ValueError(f"index metadata.total_size={total!r}, expected {observed} "
                             f"({index_size_basis} bytes)")
    for path, before in snapshots.items():
        if _content_stat(_regular_stat(path)) != before:
            raise ValueError(f"{path}: changed during structure check")
    if {p.name for p in root.iterdir() if p.name.endswith(".safetensors")} != actual:
        raise ValueError("shard roster changed during structure check")
    if weight_map is None and os.path.lexists(index):
        raise ValueError("index appeared during structure check")
    return {
        "status": "structure_ok",
        "scope": "headers_and_index_only",
        "shards": len(actual),
        "tensors": len(tensors),
        "shard_file_bytes": file_bytes,
        "header_bytes": header_bytes,
        "tensor_data_bytes": file_bytes - header_bytes,
        "index_size_basis": index_size_basis,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--expect-shards", type=int)
    parser.add_argument("--index-size-basis", choices=("data", "file"),
                        help="Explicitly compare index total_size; omitted means unchecked")
    args = parser.parse_args(argv)
    try:
        result = verify_export_structure(args.artifact_dir, expect_shards=args.expect_shards,
                                         index_size_basis=args.index_size_basis)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "structure_error", "error": str(exc)}))
        return 1
    print(DIRECT_ASCII_SPACED_LAX.text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
