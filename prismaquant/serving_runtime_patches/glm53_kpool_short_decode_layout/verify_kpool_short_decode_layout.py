#!/usr/bin/env python3
"""Compare actual short-decode rows with persistent top-k and expansion."""

import json

import torch
from vllm.models.glm5next.nvidia.ops import kpool_compress
from vllm.model_executor.layers import sparse_attn_indexer_kpool as indexer


def main() -> None:
    topk_tokens = 2048
    pool_size = 4
    select_k = topk_tokens // pool_size
    lengths = torch.arange(topk_tokens + 1, dtype=torch.int32, device="cuda")
    pool_lengths = (lengths // pool_size).reshape(-1, 1)
    generator = torch.Generator(device="cuda").manual_seed(1631)
    logits = torch.randn(
        (lengths.numel(), 1024), generator=generator, device="cuda"
    )
    pool_ids = torch.empty(
        (lengths.numel(), select_k), dtype=torch.int32, device="cuda"
    )
    workspace = torch.empty(
        (indexer.RADIX_TOPK_WORKSPACE_SIZE,), dtype=torch.uint8, device="cuda"
    )
    torch.ops._C.persistent_topk(
        logits, pool_lengths, pool_ids, workspace, select_k, 1024
    )
    expected_pool_ids = torch.arange(select_k, device="cuda", dtype=torch.int32)
    expected_pool_ids = torch.where(
        expected_pool_ids[None, :] < pool_lengths,
        expected_pool_ids[None, :],
        -1,
    )
    torch.testing.assert_close(pool_ids, expected_pool_ids, rtol=0, atol=0)
    general = kpool_compress.expand_pools_and_append_tail(
        pool_ids, lengths, pool_size
    )
    fast = torch.full_like(general, -7)
    args = (fast, lengths - 1, lengths.numel(), topk_tokens, topk_tokens)
    if "index_kpool" in indexer._fill_short_decode_causal_indices.__code__.co_varnames:
        assert indexer._fill_short_decode_causal_indices(*args, pool_size)
    else:
        assert indexer._fill_short_decode_causal_indices(*args)
    mismatched_lengths = lengths[(fast != general).any(dim=1)].cpu().tolist()
    print(json.dumps({
        "mismatched_length_count": len(mismatched_lengths),
        "first_mismatched_lengths": mismatched_lengths[:12],
        "device": torch.cuda.get_device_name(),
        "pool_order": "ascending_with_fill_last",
        "short_lengths_checked": lengths.numel(),
    }), flush=True)
    torch.testing.assert_close(fast, general, rtol=0, atol=0)

    # Capture the actual general path and replay with different pool lengths.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.persistent_topk(
            logits, pool_lengths, pool_ids, workspace, select_k, 1024
        )
        replay_general = kpool_compress.expand_pools_and_append_tail(
            pool_ids, lengths, pool_size
        )
    lengths.copy_(torch.arange(topk_tokens, -1, -1, device="cuda", dtype=torch.int32))
    pool_lengths.copy_((lengths // pool_size).reshape(-1, 1))
    graph.replay()
    indexer._fill_short_decode_causal_indices(
        fast, lengths - 1, lengths.numel(), topk_tokens, topk_tokens, pool_size
    )
    torch.testing.assert_close(fast, replay_general, rtol=0, atol=0)
    print("short-decode rows equal the captured general rows for all lengths 0..2048")


if __name__ == "__main__":
    main()
