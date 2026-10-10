#!/usr/bin/env python3
"""Assert the glm53_dflash2_drafter image carries exactly its edits.

Run by the Dockerfile after the patches (a file, not a Dockerfile
heredoc, so the classic builder runs it too: that builder drops a
heredoc body and would pass an empty check). Every hash below is a file
this image must carry byte for byte.
"""

import hashlib
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
EXPECTED = {
    # The edited files, after this patch set.
    "models/glm5next/nvidia/model.py": (
        "f6a27dfd2306056f51335eed22cc4ffc666e35bad0aa9a457d27c33c6490e444"
    ),
    "v1/core/kv_cache_utils.py": (
        "93769c6b6c43b880cb29ad7cfe6562f85142a8eb53bc89a86f00d39abe35828c"
    ),
    # The base image's own edits, carried through unchanged.
    "v1/worker/gpu/model_runner.py": (
        "1c30b8c0d3ffc96172cba57965cb7c3648156ce693ea6b3232edd2f36a9d1781"
    ),
    "models/glm5next/nvidia/mtp.py": (
        "45124573a928ecd76e6bd4f595aec3c3d98d81a71fc7a41fb32564ace560b7d1"
    ),
    # The slot-mapping kernel stays stock; only the grouping rule moved.
    "v1/worker/gpu/block_table.py": (
        "61c004315d5af7e7eae4e2a9e6be92ea82c520327690a7f55a73bb9ce95f520a"
    ),
    # Stock MLA sources that Tessera's GLM53 NoPE backend pins.
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py": (
        "a0023f72125cb0d5599b5bf940c86be1f0c9985bd62b0919f243a8fda76f4449"
    ),
    "v1/attention/backends/mla/flashinfer_mla_sparse.py": (
        "093181e4e0198b34075a3713fa62267c3f1fe74b1c96bb08298d553727d44478"
    ),
    "v1/attention/backends/mla/sparse_utils.py": (
        "20372237899fb0a0c9152e12eab40396119b76c1ac5f1c9bdfdbed53f8146559"
    ),
    "v1/attention/backend.py": (
        "8ddf8dd73c2b953a79f99e8de74db616b135590ac1f74e21e20b9ddc366b9994"
    ),
    # The base image already carries the DFlash2 drafter and its dispatch;
    # this set changes neither, and pins that fact.
    "v1/worker/gpu/spec_decode/__init__.py": (
        "1f1f84bfe6f4f5af3a7e21a7a8716ecf7b9013da1d03327668a3ee36afc24c32"
    ),
    "model_executor/models/qwen3_dflash2.py": (
        "c141daa4b2059c0098224ac36471c2197b7052c100bef0a4dbc2ca79b627053f"
    ),
    "v1/worker/gpu/spec_decode/dflash2/speculator.py": (
        "9ae6a9e27e8777d9590914cbc925d9cb3b66a3031e830abb468c7c4cb2295382"
    ),
}
for relative, digest in EXPECTED.items():
    got = hashlib.sha256((SITE / relative).read_bytes()).hexdigest()
    print(f"{got}  {relative}")
    assert got == digest, (relative, got)

import vllm.v1.core.kv_cache_utils as kv_utils
import vllm.v1.worker.gpu.model_runner as model_runner

assert "draft_specs" in kv_utils._get_kv_cache_groups_glm5_next.__code__.co_names
assert "draft_names" in kv_utils._glm5_next_tensor_layout.__code__.co_names
assert model_runner.set_eagle3_aux_hidden_state_layers is not None
print("glm53 dflash2 drafter patch verify OK")
