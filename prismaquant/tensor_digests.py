"""Tensor digests: one recipe for a tensor's host bytes (PQ #1384).

A tensor digest is an identity stored in receipts, cache manifests and
checkpoint identities, so every site must hash the same bytes. The bytes are
the tensor's host copy: ``tensor.detach().to("cpu").contiguous()`` viewed as
``uint8``. Viewing a zero-dimensional tensor whose element is wider than a
byte raises torch's ``RuntimeError``; that refusal is kept, not papered over.

Profiles:

- ``tensor_identity``: ``{"dtype", "shape", "sha256"}``, in that key order.
- ``tensor_value_stamp``: ``{"shape", "dtype", "sha256"}``, in that key order,
  after ``torch.as_tensor``, so a list or a scalar is stamped too.
- ``tensor_hash_update``: feeds ``str(tuple(shape))``, ``str(dtype)`` and the
  bytes into a running hash object (the calibration content hash).
- ``tensor_bytes``: the bytes alone, for a site that frames them itself.

This module imports only torch and ``digests``, so any module can own a call
to it without widening its import surface. ``digests`` stays stdlib-only.
"""
from __future__ import annotations

from typing import Any

import torch

from .digests import bytes_sha256hex


def _host(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").contiguous()


def _raw(host: torch.Tensor) -> bytes:
    return host.view(torch.uint8).numpy().tobytes()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    """The tensor's host bytes, C-contiguous."""
    return _raw(_host(tensor))


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
