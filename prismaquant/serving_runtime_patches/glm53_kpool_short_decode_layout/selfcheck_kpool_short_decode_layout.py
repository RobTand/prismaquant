#!/usr/bin/env python3
"""Check the patched module, carried source hashes, and all short rows."""

import hashlib
from pathlib import Path

import torch
from vllm.models.glm5next.nvidia.ops import kpool_compress
from vllm.model_executor.layers import sparse_attn_indexer_kpool as indexer

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
EXPECTED = {
    "model_executor/layers/sparse_attn_indexer_kpool.py": "5f4b6c672dafb9e9f4ab0d9e18e4fab7b98de6c11d41ba4e7546de9628738dca",
    "v1/worker/gpu/model_runner.py": "1c30b8c0d3ffc96172cba57965cb7c3648156ce693ea6b3232edd2f36a9d1781",
    "models/glm5next/nvidia/mtp.py": "45124573a928ecd76e6bd4f595aec3c3d98d81a71fc7a41fb32564ace560b7d1",
    "models/glm5next/nvidia/ops/kpool_compress.py": "bd6f1d7b5e88a10d9756799a744d3e201421b6caabd32f7b3319d31904a91a0a",
    "v1/worker/gpu/block_table.py": "61c004315d5af7e7eae4e2a9e6be92ea82c520327690a7f55a73bb9ce95f520a",
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py": "a0023f72125cb0d5599b5bf940c86be1f0c9985bd62b0919f243a8fda76f4449",
    "v1/attention/backends/mla/flashinfer_mla_sparse.py": "093181e4e0198b34075a3713fa62267c3f1fe74b1c96bb08298d553727d44478",
    "v1/attention/backends/mla/sparse_utils.py": "20372237899fb0a0c9152e12eab40396119b76c1ac5f1c9bdfdbed53f8146559",
    "v1/attention/backend.py": "8ddf8dd73c2b953a79f99e8de74db616b135590ac1f74e21e20b9ddc366b9994",
}
for relative, digest in EXPECTED.items():
    got = hashlib.sha256((SITE / relative).read_bytes()).hexdigest()
    assert got == digest, (relative, got)

lengths = torch.arange(2049, dtype=torch.int32)
rows = torch.full((lengths.numel(), 2051), -7, dtype=torch.int32)
assert indexer._fill_short_decode_causal_indices(
    rows, lengths - 1, lengths.numel(), 2048, 2048, 4
)
expected = torch.full_like(rows, -1)
for row, length in zip(expected, lengths.tolist()):
    tail_start = (length // 4) * 4
    row[:tail_start] = torch.arange(tail_start, dtype=torch.int32)
    row[2048 : 2048 + length % 4] = torch.arange(tail_start, length, dtype=torch.int32)
torch.testing.assert_close(rows, expected, rtol=0, atol=0)
print("Docker self-check passes for every length 0..2048 and all carried source hashes")
print("patched_indexer_sha256=" + hashlib.sha256(Path(indexer.__file__).read_bytes()).hexdigest())
