#!/usr/bin/env python3
"""Preserve automatic block-size provenance in the pinned draft dtype copy.

PQ #1514: dataclasses.replace resets CacheConfig's init=False fields. Its
validator then mistakes the already resolved block size for a user choice.
Restore that flag before nesting the validated copy into VllmConfig. Keep the
resolved block size and the cache-dtype validation; do not broaden a backend's
supported block sizes. This is a candidate image edit, not serving evidence.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

BASE_SHA256 = "65d882b8fb476eb0dd6161247346cd7b0392b22832abf36fbb11e795979f9e28"
TARGET = "v1/worker/gpu/spec_decode/eagle/utils.py"
OLD_COPY = """            cache_config=replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            ),"""
NEW_COPY = """            cache_config=_draft_cache_config(
                vllm_config.cache_config, speculative_config.kv_cache_dtype
            ),"""
HELPER = '''def _draft_cache_config(cache_config, cache_dtype):
    """Validate a draft dtype while preserving the target's block-size origin."""
    draft = replace(cache_config, cache_dtype=cache_dtype)
    # PQ1514: replace resets init=False flags and re-runs the validator on
    # an already resolved int. This copy changes only dtype, not block size.
    draft.user_specified_block_size = cache_config.user_specified_block_size
    return draft


'''
ANCHOR = "def load_eagle_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:\n"


def patched_source(raw: bytes) -> bytes:
    digest = hashlib.sha256(raw).hexdigest()
    if digest != BASE_SHA256:
        raise ValueError(f"{TARGET} is not the base image's (sha256 {digest})")
    text = raw.decode("utf-8")
    if text.count(OLD_COPY) != 1 or text.count(ANCHOR) != 1:
        raise ValueError("expected exactly one draft dtype copy and load_eagle_model")
    return text.replace(OLD_COPY, NEW_COPY).replace(ANCHOR, HELPER + ANCHOR).encode("utf-8")


def main() -> None:
    target = Path("/usr/local/lib/python3.12/dist-packages/vllm") / TARGET
    target.write_bytes(patched_source(target.read_bytes()))
    print("applied glm53_draft_cache_copy")


if __name__ == "__main__":
    main()
