"""Independent selected original functions; never import production helpers.
Source commit 30b6231df000726beda55fb6706e641a5cb2ccb0; shipcard blob acf1b46ab0802eada0512a22b78828bd9420bd8a.
Only the two consumers and their transitive stdlib-only validation helpers.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any, Callable, Mapping

def safetensors_prefix_length(raw: bytes, size: int, *, max_bytes: int,
                              short_error: str, range_error: str | Callable[[int], str]) -> int:
    """Decode a bounded u64 prefix; acquisition and refusal text stay local.

    Callers supply exactly the eight prefix bytes, not the header body. The
    explicit bound and messages preserve each reader's existing contract.
    A callable range message receives the decoded unsigned length only on
    refusal; existing literal messages remain unchanged.
    """
    if len(raw) != 8:
        raise ValueError(short_error)
    (length,) = struct.unpack("<Q", raw)
    if not 0 < length <= min(max_bytes, size - 8):
        message = (range_error(length)
                   if callable(range_error) and not isinstance(range_error, str)
                   else range_error)
        raise ValueError(message)
    return length

_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024

_SAFETENSORS_DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E4M3FN": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E8M0": 8,
    "F4": 4,
    "F4_E2M1": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "U16": 16,
    "I16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "I32": 32,
    "F32": 32,
    "U64": 64,
    "I64": 64,
    "F64": 64,
}

def _content_stat(st: os.stat_result) -> dict[str, int]:
    return {
        "device": int(st.st_dev),
        "inode": int(st.st_ino),
        "bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "ctime_ns": int(st.st_ctime_ns),
    }

def _strict_json_object(raw: bytes, *, where: str) -> Mapping[str, Any]:
    def _pairs(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"{where}: duplicate JSON key {key!r}")
            out[key] = value
        return out

    def _constant(value: str) -> None:
        raise ValueError(f"{where}: nonfinite JSON constant {value!r}")

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{where}: invalid safetensors header JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{where}: safetensors header must be an object")
    return payload

def safetensors_header_spans(
    raw_header: bytes,
    *,
    data_bytes: int,
    where: str,
) -> tuple[tuple[int, int, str], ...]:
    """Validate one header and return its exact contiguous payload geometry."""

    header = _strict_json_object(raw_header, where=where)
    spans: list[tuple[int, int, str]] = []
    for tensor_name, descriptor in header.items():
        if tensor_name == "__metadata__":
            if not isinstance(descriptor, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in descriptor.items()
            ):
                raise ValueError(
                    f"{where}: __metadata__ must be a string-to-string object"
                )
            continue
        offsets = descriptor.get("data_offsets") if isinstance(
            descriptor, Mapping
        ) else None
        dtype = descriptor.get("dtype") if isinstance(
            descriptor, Mapping
        ) else None
        shape = descriptor.get("shape") if isinstance(
            descriptor, Mapping
        ) else None
        if (
            not isinstance(tensor_name, str)
            or not tensor_name
            or not isinstance(descriptor, Mapping)
            or set(descriptor) != {"dtype", "shape", "data_offsets"}
            or dtype not in _SAFETENSORS_DTYPE_BITS
            or not isinstance(shape, list)
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
        ):
            raise ValueError(f"{where}: invalid tensor header {tensor_name!r}")
        start, end = offsets
        if not 0 <= start <= end <= data_bytes:
            raise ValueError(f"{where}: invalid data span for {tensor_name!r}")
        expected_bytes = (
            int(math.prod(shape)) * _SAFETENSORS_DTYPE_BITS[str(dtype)] + 7
        ) // 8
        if end - start != expected_bytes:
            raise ValueError(
                f"{where}: tensor {tensor_name!r} span is {end - start}B but "
                f"{dtype}{tuple(shape)} requires {expected_bytes}B"
            )
        spans.append((start, end, tensor_name))
    spans.sort()
    if not spans:
        raise ValueError(f"{where}: safetensors container has no tensors")
    cursor = 0
    for start, end, tensor_name in spans:
        if start != cursor:
            relation = "overlaps" if start < cursor else "leaves a gap after"
            raise ValueError(
                f"{where}: tensor {tensor_name!r} {relation} the preceding "
                f"data span (expected offset {cursor}, got {start})"
            )
        cursor = end
    if cursor != data_bytes:
        raise ValueError(
            f"{where}: tensor spans end at {cursor}B but data area is "
            f"{data_bytes}B"
        )
    return tuple(spans)

def _read_exact_fd(
    fd: int,
    length: int,
    *,
    digest: Any,
    tensor_digest: Any | None = None,
    capture: bool = False,
) -> tuple[bytes, int]:
    chunks: list[bytes] = []
    remaining = int(length)
    calls = 0
    while remaining:
        chunk = os.read(fd, min(1024 * 1024, remaining))
        calls += 1
        if not chunk:
            raise ValueError("truncated safetensors content during verification")
        digest.update(chunk)
        if tensor_digest is not None:
            tensor_digest.update(chunk)
        if capture:
            chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks), calls

def _verify_open_safetensors_fd(
    fd: int,
    *,
    name: str,
    initial_stat: os.stat_result,
) -> tuple[dict[str, object], int, int]:
    """Hash one container and all tensor spans in one forward traversal."""

    full_digest = hashlib.sha256()
    raw_length, read_calls = _read_exact_fd(
        fd, 8, digest=full_digest, capture=True
    )
    header_length = safetensors_prefix_length(
        raw_length, int(initial_stat.st_size), max_bytes=_MAX_SAFETENSORS_HEADER_BYTES,
        short_error="truncated safetensors content during verification",
        range_error=lambda length: f"{name}: invalid safetensors header length {length}")
    raw_header, calls = _read_exact_fd(
        fd, header_length, digest=full_digest, capture=True
    )
    read_calls += calls
    data_bytes = int(initial_stat.st_size) - 8 - header_length
    spans = safetensors_header_spans(
        raw_header,
        data_bytes=data_bytes,
        where=name,
    )
    tensor_digests: dict[str, str] = {}
    bytes_read = 8 + header_length
    for start, end, tensor_name in spans:
        tensor_digest = hashlib.sha256()
        _ignored, calls = _read_exact_fd(
            fd,
            end - start,
            digest=full_digest,
            tensor_digest=tensor_digest,
        )
        read_calls += calls
        bytes_read += end - start
        tensor_digests[tensor_name] = tensor_digest.hexdigest()
    final_stat = os.fstat(fd)
    if _content_stat(final_stat) != _content_stat(initial_stat):
        raise ValueError(f"{name}: file stat changed during content verification")
    return {
        "stat": _content_stat(final_stat),
        "sha256": full_digest.hexdigest(),
        "tensor_sha256": dict(sorted(tensor_digests.items())),
    }, bytes_read, read_calls

def write_shipcard(path: str | os.PathLike, card: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(card, indent=2, default=str) + "\n").encode("utf-8")
    reserved = card.get("reserved_file_bytes")
    if reserved is not None:
        if isinstance(reserved, bool) or not isinstance(reserved, int) or reserved <= 0:
            raise ValueError("shipcard reserved_file_bytes must be a positive integer")
        if len(encoded) > reserved:
            # A full card outgrows pretty-printing long before it outgrows its
            # reservation: indent=2 inflates this schema ~1.5x, and a six-slot
            # DSv4 card crossed the line on the LAST slot fill -- dropping a
            # record every gate had already passed.  The reservation is a byte
            # contract (the file's size is part of the frozen artifact
            # inventory), not a formatting contract; fall back to compact JSON
            # before refusing.
            encoded = (
                json.dumps(card, separators=(",", ":"), default=str) + "\n"
            ).encode("utf-8")
        if len(encoded) > reserved:
            raise ValueError(
                f"shipcard needs {len(encoded)} bytes but its fixed reservation "
                f"is {reserved} bytes; refusing to invalidate the artifact inventory"
            )
        encoded += b" " * (reserved - len(encoded))
    out.write_bytes(encoded)
    return out
