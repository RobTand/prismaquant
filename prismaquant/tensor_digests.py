"""Tensor digests: one recipe for a tensor's host bytes (PQ #1384).

A tensor digest is an identity stored in receipts, cache manifests and
checkpoint identities, so every site must hash the same bytes. The bytes are
the tensor's host copy: ``tensor.detach().to("cpu").contiguous()`` viewed as
``uint8``. Viewing a zero-dimensional tensor whose element is wider than a
byte raises torch's ``RuntimeError``; that refusal is kept, not papered over.

- ``tensor_hash_update``: feeds ``str(tuple(shape))``, ``str(dtype)`` and the
  bytes into a running hash object (the calibration content hash).
- ``tensor_host_bytes``: the bytes alone, for a site that frames them itself.
- ``token_ids_int32_sha256``: caller-ordered int32 token batches, each cast
  to int32 then folded with no delimiter, shape, or dtype header
  (cohort ``int32-token-stream``).
- ``tensor_chunked_payload_sha256``: contiguous payload bytes fed in
  ``chunk_bytes`` element windows, dtype and shape aside, one chunk beside
  the device tensor at a time (cohort ``device-tensor-chunks``).
- ``tensor_view_stream_identity``: ``{"shape", "dtype", "logical_bytes",
  "content_sha256"}``, in that key order, over flattened uint8 memoryview
  windows with no second copy (cohort ``tensor-view-stream``).
- ``fp32_tensor_stream_identity``: ``(shape, hex)`` over C-order chunks of
  at most 4194304 elements, each converted to contiguous CPU FP32 and fed
  as little-endian f4 bytes (cohort ``fp32-tensor-stream``).

Each chunked recipe keeps its exact feed: widths, conversions, fold order,
empty-tensor behavior, and refusals. A consumer keeps its own name and
result wrapper and delegates the bytes here.

This module imports only torch, hashlib, math, and ``digests``, so any
module can own a call to it without widening its import surface.
``digests`` stays stdlib-only.
"""
from collections.abc import Sequence
from typing import Any

import hashlib
import math

import torch

from .digests import bytes_sha256hex


def _host(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").contiguous()


def _raw(host: torch.Tensor) -> bytes:
    return host.view(torch.uint8).numpy().tobytes()


def tensor_host_bytes(tensor: torch.Tensor) -> bytes:
    """The tensor's host bytes, C-contiguous."""
    return _raw(_host(tensor))


def tensor_sha256(tensor: torch.Tensor) -> str:
    """SHA-256 of the tensor's exact host bits, C-contiguous (PQ #1531)."""
    return bytes_sha256hex(tensor_host_bytes(tensor))


def tensor_identity(tensor: torch.Tensor) -> dict[str, object]:
    """``{"dtype", "shape", "sha256"}`` of the tensor's host bytes."""
    host = _host(tensor)
    return {
        "dtype": str(host.dtype),
        "shape": [int(dim) for dim in host.shape],
        "sha256": bytes_sha256hex(_raw(host)),
    }


def tensor_value_stamp(value: Any) -> dict[str, object]:
    """``{"shape", "dtype", "sha256"}`` of ``torch.as_tensor(value)``."""
    host = _host(torch.as_tensor(value))
    return {
        "shape": [int(dim) for dim in host.shape],
        "dtype": str(host.dtype),
        "sha256": bytes_sha256hex(_raw(host)),
    }


def tensor_hash_update(digest: Any, tensor: torch.Tensor) -> None:
    """Feed shape text, dtype text and the host bytes into ``digest``."""
    host = _host(tensor)
    digest.update(str(tuple(host.shape)).encode())
    digest.update(str(host.dtype).encode())
    digest.update(_raw(host))

def token_ids_int32_sha256(batches: Sequence[torch.Tensor]) -> str:
    """SHA-256 over caller-ordered batches cast to int32 (PQ #2638)."""
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(batch.to(dtype=torch.int32).cpu().numpy().tobytes())
    return digest.hexdigest()


def tensor_chunked_payload_sha256(
    tensor: torch.Tensor, *, chunk_bytes: int = 1 << 26
) -> str:
    """SHA-256 of contiguous payload bytes, dtype and shape aside (PQ #2638).

    The bytes reach the host ``chunk_bytes`` at a time, so hashing a device
    tensor holds one chunk beside it, never a second copy of it.
    """
    data = tensor.detach().contiguous().reshape(-1)
    digest = hashlib.sha256()
    step = max(1, int(chunk_bytes) // max(data.element_size(), 1))
    for start in range(0, data.numel(), step):
        digest.update(data[start:start + step].to("cpu").view(torch.uint8).numpy())
    return digest.hexdigest()


#: Feed width for streamed host-tensor digests. ``hashlib`` releases the GIL
#: for buffers this size, so one memoryview fed in wide slices hashes at the
#: same single-core rate as one contiguous feed, without the ``tobytes()``
#: copy that would double resident bytes per tensored identity (PQ #725).
_TENSOR_VIEW_STREAM_CHUNK_BYTES = 8 << 20


def tensor_view_stream_identity(tensor: torch.Tensor) -> dict[str, object]:
    """``{"shape", "dtype", "logical_bytes", "content_sha256"}`` (PQ #2638).

    ``cast("B")`` flattens the dimensions without copying, so slices below
    are byte windows, never copies. An empty tensor takes the empty feed
    directly -- the same sha256 of zero bytes ``tobytes()`` produced.
    """
    stored = tensor.detach().to(device="cpu").contiguous()
    nbytes = stored.nbytes
    digest = hashlib.sha256()
    if nbytes:
        view = memoryview(stored.view(torch.uint8).numpy()).cast("B")
        for offset in range(0, len(view), _TENSOR_VIEW_STREAM_CHUNK_BYTES):
            digest.update(view[offset:offset + _TENSOR_VIEW_STREAM_CHUNK_BYTES])
    return {
        "shape": [int(dim) for dim in stored.shape],
        "dtype": str(stored.dtype),
        "logical_bytes": nbytes,
        "content_sha256": digest.hexdigest(),
    }


def fp32_tensor_stream_identity(weight: Any) -> tuple[list[int], str]:
    """``(shape, hex)`` over C-order FP32 chunks of any weight (PQ #2638).

    Chunks hold at most 4194304 elements (16 MiB after FP32 conversion).
    Concatenating them is exactly C-order traversal.
    """
    tensor = torch.as_tensor(weight).detach()
    shape = [int(dim) for dim in tensor.shape]
    digest = hashlib.sha256()
    max_chunk_elements = 4 * 1024 * 1024  # <=16 MiB after fp32 conversion

    def _chunks_c_order(value: torch.Tensor):
        if value.numel() <= max_chunk_elements or value.ndim == 0:
            yield value
            return
        trailing = math.prod(int(dim) for dim in value.shape[1:])
        if trailing <= max_chunk_elements:
            step = max(1, max_chunk_elements // max(trailing, 1))
            for start in range(0, int(value.shape[0]), step):
                yield value[start:start + step]
            return
        # A single leading slice is still too large. Recurse dimension by
        # dimension; concatenating these chunks is exactly C-order traversal.
        for index in range(int(value.shape[0])):
            yield from _chunks_c_order(value[index])

    for chunk in _chunks_c_order(tensor):
        cpu = chunk.to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(
            cpu.numpy().astype("<f4", copy=False).tobytes(order="C")
        )
        del cpu
    return shape, digest.hexdigest()
