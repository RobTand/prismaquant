#!/usr/bin/env python3
"""Assert the glm53_kpool_tail_slot_mapping image carries exactly its edit.

Run by the Dockerfile after the patch (a file, not a Dockerfile heredoc, so the
classic builder runs it too: that builder drops a heredoc body and would pass an
empty check). Every hash below is a file this image must carry byte for byte.
"""

import hashlib
from pathlib import Path

import vllm.v1.worker.gpu.model_runner as model_runner
from vllm.v1.kv_cache_interface import CircularBufferSpec, KpoolTailSpec

assert model_runner.KpoolTailSpec is KpoolTailSpec
assert model_runner.CircularBufferSpec is CircularBufferSpec

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
EXPECTED = {
    # The edited file, after this patch set.
    "v1/worker/gpu/model_runner.py": "1c30b8c0d3ffc96172cba57965cb7c3648156ce693ea6b3232edd2f36a9d1781",
    # The base image's own edit (glm53_mtp_mapper), carried through unchanged.
    "models/glm5next/nvidia/mtp.py": "45124573a928ecd76e6bd4f595aec3c3d98d81a71fc7a41fb32564ace560b7d1",
    # The slot-mapping kernel itself stays stock; only the runner's rule moved.
    "v1/worker/gpu/block_table.py": "61c004315d5af7e7eae4e2a9e6be92ea82c520327690a7f55a73bb9ce95f520a",
    # Stock MLA sources that Tessera's GLM53 NoPE backend pins.
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py": "a0023f72125cb0d5599b5bf940c86be1f0c9985bd62b0919f243a8fda76f4449",
    "v1/attention/backends/mla/flashinfer_mla_sparse.py": "093181e4e0198b34075a3713fa62267c3f1fe74b1c96bb08298d553727d44478",
    "v1/attention/backends/mla/sparse_utils.py": "20372237899fb0a0c9152e12eab40396119b76c1ac5f1c9bdfdbed53f8146559",
    "v1/attention/backend.py": "8ddf8dd73c2b953a79f99e8de74db616b135590ac1f74e21e20b9ddc366b9994",
}
for relative, digest in EXPECTED.items():
    got = hashlib.sha256((SITE / relative).read_bytes()).hexdigest()
    print(f"{got}  {relative}")
    assert got == digest, (relative, got)
print("glm53 kpool-tail slot-mapping patch verify OK; MLA sources stock; MTP mapper carried")
