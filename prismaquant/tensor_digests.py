"""Tensor digests: exact named recipes for tensor identities (PQ #1384, #1301).

A tensor digest is an identity stored in receipts, cache manifests and
checkpoint identities, so every site using a profile must hash the same bytes.
The original host-byte profiles below copy with
``tensor.detach().to("cpu").contiguous()`` and view as ``uint8``. Viewing a
zero-dimensional tensor whose element is wider than a byte raises torch's
``RuntimeError``; that refusal is kept, not papered over.

Original host-byte profiles:

- ``tensor_identity``: ``{"dtype", "shape", "sha256"}``, in that key order.
- ``tensor_value_stamp``: ``{"shape", "dtype", "sha256"}``, in that key order,
  after ``torch.as_tensor``, so a list or a scalar is stamped too.
- ``tensor_hash_update``: feeds ``str(tuple(shape))``, ``str(dtype)`` and the
  bytes into a running hash object (the calibration content hash).
- ``tensor_host_bytes``: the bytes alone, for a site that frames them itself.

Bounded profiles retain distinct original recipes: ``tensor_payload_sha256``
copies flattened payload chunks to CPU; ``source_weight_value_identity`` walks
C-order chunks and hashes little-endian fp32 values; ``cb_cache_tensor_identity``
hashes a full contiguous host copy through 8 MiB memoryview windows. Their
copy order, scalar behavior and errors are intentionally not unified.
``digests`` stays stdlib-only.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

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


def tensor_payload_sha256(tensor, *, chunk_bytes: int = 1 << 26) -> str:
    """sha256 of a tensor's contiguous payload bytes, dtype and shape aside.

    The bytes are copied to the host ``chunk_bytes`` at a time, so hashing a
    device tensor holds one chunk beside it, not a second copy of it.
    """
    import torch

    data = tensor.detach().contiguous().reshape(-1)
    digest = hashlib.sha256()
    step = max(1, int(chunk_bytes) // max(data.element_size(), 1))
    for start in range(0, data.numel(), step):
        digest.update(data[start:start + step].to("cpu").view(torch.uint8).numpy())
    return digest.hexdigest()


def source_weight_value_identity(
    weight: torch.Tensor,
) -> tuple[list[int], str]:
    """C-order value identity in <=4M-element little-endian fp32 chunks."""
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


_TENSOR_DIGEST_CHUNK_BYTES = 8 << 20


def cb_cache_tensor_identity(tensor: torch.Tensor) -> dict[str, object]:
    """Cache identity of host bits, fed as zero-copy 8 MiB byte windows."""
    stored = tensor.detach().to(device="cpu").contiguous()
    # ``cast("B")`` flattens the dimensions without copying, so slices below
    # are byte windows, never copies. (A multi-dimensional memoryview's bare
    # ``len`` would be the first axis, not the byte count.) An empty tensor
    # has zeros in its shape, which ``cast`` refuses, so it takes the empty
    # feed directly -- the same sha256 of zero bytes ``tobytes()`` produced.
    nbytes = stored.nbytes
    digest = hashlib.sha256()
    if nbytes:
        view = memoryview(stored.view(torch.uint8).numpy()).cast("B")
        for offset in range(0, len(view), _TENSOR_DIGEST_CHUNK_BYTES):
            digest.update(view[offset:offset + _TENSOR_DIGEST_CHUNK_BYTES])
    return {
        "shape": [int(dim) for dim in stored.shape],
        "dtype": str(stored.dtype),
        "logical_bytes": nbytes,
        "content_sha256": digest.hexdigest(),
    }
