#!/usr/bin/env python3
"""Assert the glm53_kpool_short_decode_layout image carries exactly its edit.

Run by the Dockerfile after the patch (a file, not a Dockerfile heredoc, so the
classic builder runs it too: that builder drops a heredoc body and would pass an
empty check). Every hash below is a file this image must carry byte for byte.
"""

import hashlib
from pathlib import Path

import vllm.model_executor.layers.sparse_attn_indexer_kpool as indexer

assert indexer._fill_short_decode_general_layout is not None
assert indexer._canonicalize_short_row_pool_ids is not None
assert "index_kpool" in indexer._fill_short_decode_causal_indices.__code__.co_varnames

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
EXPECTED = {
    # The edited file, after this patch set.
    "model_executor/layers/sparse_attn_indexer_kpool.py": "b689bcc1bc4d762eee687dce7444ca399e86e39e024d78ca367f0d07108cf29e",
    # The base image's own edits, carried through unchanged.
    "v1/worker/gpu/model_runner.py": "1c30b8c0d3ffc96172cba57965cb7c3648156ce693ea6b3232edd2f36a9d1781",
    "models/glm5next/nvidia/mtp.py": "45124573a928ecd76e6bd4f595aec3c3d98d81a71fc7a41fb32564ace560b7d1",
    # The slot-mapping kernel stays stock; only the indexer's row layout moved.
    # The expand kernel is covered by the base-image digest: the Dockerfile
    # builds FROM the pinned digest and the patch touches one file, so every
    # other byte is the base's by construction (PQ #1631 manifest notes).
    "v1/worker/gpu/block_table.py": "61c004315d5af7e7eae4e2a9e6be92ea82c520327690a7f55a73bb9ce95f520a",
}
for relative, digest in EXPECTED.items():
    got = hashlib.sha256((SITE / relative).read_bytes()).hexdigest()
    print(f"{got}  {relative}")
    assert got == digest, (relative, got)
print("glm53 kpool short-decode layout patch verify OK; only the indexer moved")
