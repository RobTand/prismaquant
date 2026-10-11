#!/usr/bin/env python3
"""Match short-decode rows to the general path in the pinned GLM image.

The base fast path places incomplete-pool tokens beside the history.
The general path places those tokens after the fixed top-k region.
PrismaBuild action 800a707a proves ascending pool order for all short rows.
Thus only the fast-path layout must change. The general path stays stock.
The patch refuses different base bytes and requires each anchor once.
"""

import hashlib
from pathlib import Path

#: model_executor/layers/sparse_attn_indexer_kpool.py as the base image
#: ships it. The edit refuses any other bytes, so a moved base fails the
#: build instead of receiving an edit written for another file.
BASE_INDEXER_SHA256 = "631ae1cce792c18f2922af7763264c486317cb6f4318823b1482e55bec5d7f59"

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
target = SITE / "model_executor/layers/sparse_attn_indexer_kpool.py"

#: Pure-torch helpers, inserted before the fast path. The CPU gate
#: (tests/test_kpool_short_decode_layout.py) executes exactly this block.
NEW_LAYOUT_HELPERS = '''def _fill_short_decode_general_layout(
    buffer_rows: torch.Tensor,
    positions: torch.Tensor,
    topk_tokens: int,
    pool_size: int,
) -> None:
    """Write the general-path sparse-index row for short contexts.

    Pools expanded in order into columns ``[0 .. tail_start)``, ``-1``
    through column ``topk_tokens - 1``, and the incomplete pool's newest
    tokens at ``[topk_tokens .. topk_tokens + tail_count)``. Matches
    ``_expand_pools_and_append_tail_kernel`` for an all-selected row in
    ascending pool order (PQ #1631).
    """
    seq_lens = positions.to(torch.int32) + 1
    tail_starts = (seq_lens // pool_size) * pool_size
    tail_counts = seq_lens - tail_starts
    cols = torch.arange(
        buffer_rows.shape[1], device=buffer_rows.device, dtype=torch.int32
    )[None, :]
    history = cols < tail_starts[:, None]
    tail_off = cols - topk_tokens
    tail = (tail_off >= 0) & (tail_off < tail_counts[:, None])
    out = torch.where(
        history, cols.expand(buffer_rows.shape[0], buffer_rows.shape[1]), -1
    )
    out = torch.where(tail, tail_starts[:, None] + tail_off, out)
    buffer_rows[:] = out


'''

OLD_FAST_PATH_FUNC = '''def _fill_short_decode_causal_indices(
    topk_indices_buffer: torch.Tensor,
    positions: torch.Tensor | None,
    num_decode_tokens: int,
    max_seq_len: int,
    topk_tokens: int,
) -> bool:
    """Fill exact causal rows when sparse decode would select every token."""
    if positions is None or positions.numel() == 0 or max_seq_len > topk_tokens:
        return False
    _fill_causal_indices(
        topk_indices_buffer[:num_decode_tokens], positions[:num_decode_tokens]
    )
    return True
'''

NEW_FAST_PATH_FUNC = '''def _fill_short_decode_causal_indices(
    topk_indices_buffer: torch.Tensor,
    positions: torch.Tensor | None,
    num_decode_tokens: int,
    max_seq_len: int,
    topk_tokens: int,
    index_kpool: int,
) -> bool:
    """Fill short-decode rows in the general-path layout (PQ #1631).

    When sparse decode would select every token (the batch's longest
    sequence fits the top-k budget), write exactly the row the general
    path builds: pools expanded in order, ``-1`` through the unused
    slots, and the incomplete pool at ``topk_tokens`` and up. Capture
    freezes the general path into every FULL graph, so eager must write
    the same bytes. Without kpool the general row is the causal row.
    """
    if positions is None or positions.numel() == 0 or max_seq_len > topk_tokens:
        return False
    if index_kpool > 1:
        _fill_short_decode_general_layout(
            topk_indices_buffer[:num_decode_tokens],
            positions[:num_decode_tokens],
            topk_tokens,
            index_kpool,
        )
    else:
        _fill_causal_indices(
            topk_indices_buffer[:num_decode_tokens], positions[:num_decode_tokens]
        )
    return True
'''

OLD_DECODE_FAST_CALL = '''        if current_platform.is_cuda_alike() and _fill_short_decode_causal_indices(
            topk_indices_buffer,
            positions,
            num_decode_tokens,
            attn_metadata_narrowed.max_seq_len,
            topk_tokens,
        ):'''

NEW_DECODE_FAST_CALL = '''        if current_platform.is_cuda_alike() and _fill_short_decode_causal_indices(
            topk_indices_buffer,
            positions,
            num_decode_tokens,
            attn_metadata_narrowed.max_seq_len,
            topk_tokens,
            index_kpool,
        ):'''


EDITS = (
    ("short_decode_general_layout", OLD_FAST_PATH_FUNC,
     NEW_LAYOUT_HELPERS + NEW_FAST_PATH_FUNC),
    ("short_decode_pool_size", OLD_DECODE_FAST_CALL, NEW_DECODE_FAST_CALL),
)


def patch_indexer(raw: bytes) -> bytes:
    digest = hashlib.sha256(raw).hexdigest()
    if digest != BASE_INDEXER_SHA256:
        raise SystemExit(f"indexer is not the base image's (sha256 {digest})")
    text = raw.decode()
    for tag, old, new in EDITS:
        found = text.count(old)
        if found != 1:
            raise SystemExit(f"expected one {tag} anchor, found {found}")
        text = text.replace(old, new)
    return text.encode()


def main() -> None:
    target.write_bytes(patch_indexer(target.read_bytes()))
    print("applied glm53_kpool_short_decode_layout")


if __name__ == "__main__":
    main()
