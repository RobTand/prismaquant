"""PQ #725 fix 4: ``_cb_cache_tensor_identity`` hashes through a chunked
byte-view instead of a ``tobytes()`` copy, stamping the exact legacy
digest (same shape/dtype/logical_bytes/content_sha256 receipt)."""
from __future__ import annotations

import hashlib

import torch


def _legacy_tensor_digest(tensor):
    stored = tensor.detach().to(device="cpu").contiguous()
    raw = stored.view(torch.uint8).numpy().tobytes()
    return {
        "shape": [int(dim) for dim in stored.shape],
        "dtype": str(stored.dtype),
        "logical_bytes": len(raw),
        "content_sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_cb_tensor_identity_matches_legacy_digest():
    from prismaquant.production_weight_cache import _cb_cache_tensor_identity

    for shape, dtype in [((16, 16), torch.bfloat16),
                         ((1024, 4096), torch.bfloat16),
                         ((7, 130), torch.float32)]:
        tensor = torch.randn(*shape, dtype=torch.float32).to(dtype)
        assert _cb_cache_tensor_identity(tensor) == _legacy_tensor_digest(tensor)


def test_cb_tensor_identity_accepts_noncontiguous_input():
    from prismaquant.production_weight_cache import _cb_cache_tensor_identity

    base = torch.randn(64, 64, dtype=torch.bfloat16)
    assert (_cb_cache_tensor_identity(base.t())
            == _legacy_tensor_digest(base.t()))


def test_cb_tensor_identity_hashes_past_the_chunk_width():
    from prismaquant.production_weight_cache import (
        _TENSOR_DIGEST_CHUNK_BYTES, _cb_cache_tensor_identity)

    tensor = torch.randn(2048, 4096, dtype=torch.bfloat16)  # 16 MiB
    assert tensor.nbytes > _TENSOR_DIGEST_CHUNK_BYTES
    assert _cb_cache_tensor_identity(tensor) == _legacy_tensor_digest(tensor)


def test_cb_tensor_identity_covers_empty_tensors():
    from prismaquant.production_weight_cache import _cb_cache_tensor_identity

    tensor = torch.empty(0, 8, dtype=torch.bfloat16)
    assert _cb_cache_tensor_identity(tensor) == _legacy_tensor_digest(tensor)
