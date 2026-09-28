#!/usr/bin/env python3
"""Keep the kpool-tail KV group out of the V2 runner's generic slot mapping.

Backport of vLLM #57317 (upstream commit 70df48dc3d01, 2026-09-18) to the
GLM serving image. Upstream gave every KV-cache spec a ``uses_slot_mapping``
property and made ``KpoolTailSpec`` answer ``False``. This image predates that
refactor, and its runner enables the generic mapping for every spec that is not
a ``CircularBufferSpec``, so the kpool-tail group is mapped too.

The generic mapping (``v1/worker/gpu/block_table.py``,
``_compute_slot_mappings_kernel``) loads ``block_table[req, position //
block_size]`` without a mask. The kpool tail has ONE 4-token circular block per
request, in a row 32 entries wide, so every position at or past 128 reads past
its row, and every position at or past 1024 (row 0) reads past the whole
1 KiB table. The CUDA memcheck tool reports the read at ``block_table.py:344``
for group 1 (tessera#508). The values it produces are never used: the
kpool-tail metadata builder overwrites the tail row with
``block_table[req, 0] * 4 + position % 4`` whenever it has positions. The read
is the defect, and when it leaves the allocation it is an illegal memory
access.

After this patch the tail row stays at the pad slot until the builder writes
it, exactly as upstream: the runner no longer reads the tail's table at all.

Two edits, each asserted to land exactly once, and the file is refused unless
it is byte-identical to the base image's.
"""

import hashlib
from pathlib import Path

#: v1/worker/gpu/model_runner.py as the base image ships it. The edit refuses
#: any other bytes, so a moved base fails the build instead of receiving an edit
#: written for another file.
BASE_MODEL_RUNNER_SHA256 = "39a5edd7b1e76b13c039be4b22a72dc512a17c3b78b9fa9aa34158ce9a7d78c3"

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
target = SITE / "v1/worker/gpu/model_runner.py"
raw = target.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
if digest != BASE_MODEL_RUNNER_SHA256:
    raise SystemExit(f"model_runner.py is not the base image's (sha256 {digest})")
text = raw.decode()

old_import = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    CircularBufferSpec,\n"
    "    KVCacheConfig,\n"
)
if text.count(old_import) != 1:
    raise SystemExit("expected one kv_cache_interface import block in model_runner.py")
text = text.replace(
    old_import,
    "from vllm.v1.kv_cache_interface import (\n"
    "    CircularBufferSpec,\n"
    "    KpoolTailSpec,\n"
    "    KVCacheConfig,\n",
)

old_rule = (
    "            slot_mapping_enabled.append(not isinstance(layer_spec, CircularBufferSpec))\n"
)
if text.count(old_rule) != 1:
    raise SystemExit("expected one slot_mapping_enabled rule in model_runner.py")
text = text.replace(
    old_rule,
    "            # GLM53_KPOOL_TAIL_SLOT_MAPPING (vLLM #57317, 70df48dc3d01): the\n"
    "            # kpool tail addresses its one circular block itself; the generic\n"
    "            # mapping would read its 32-entry row by absolute position.\n"
    "            slot_mapping_enabled.append(\n"
    "                not isinstance(layer_spec, (CircularBufferSpec, KpoolTailSpec))\n"
    "            )\n",
)
target.write_text(text)
print("applied glm53_kpool_tail_slot_mapping")
