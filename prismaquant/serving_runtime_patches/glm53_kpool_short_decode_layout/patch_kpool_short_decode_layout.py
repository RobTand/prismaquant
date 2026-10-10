#!/usr/bin/env python3
"""Write the general-path sparse-index row on the short-decode fast path.

PQ #1631: in this image eager and FULL CUDA-graph serves build the decode
index differently. The fast path (``_fill_short_decode_causal_indices``)
writes the causal row ``[0 .. pos, -1 ..]`` when the batch's longest sequence
fits the top-k budget. The general path expands the selected pools into
columns ``[0 .. 2047]`` with ``-1`` in unused slots and writes the
incomplete pool's newest tokens at columns ``[2048 .. 2050]``
(``_expand_pools_and_append_tail_kernel``). The fast-path test reads
``attn_metadata.max_seq_len`` on the host, so capture freezes the general
path into every FULL graph and each replay runs it even for short contexts.

The fix has two halves, both in
``vllm/model_executor/layers/sparse_attn_indexer_kpool.py``:

* The fast path writes the general layout directly: pools expanded in
  order into columns ``[0 .. 4n)``, ``-1`` through column 2047, and the
  incomplete pool at 2048 and up. Eager keeps its speed (no logits, no
  top-k) and both modes build the same row.
* The general path sorts each short row's selected pool ids ascending
  before expansion, with fill (``-1``) kept last. ``persistent_topk``
  orders a fully selected row by score, not by pool, so without this the
  rows still differ by construction. Long rows are untouched.

Five edits, each asserted to land exactly once, and the file is refused
unless it is byte-identical to the base image's.
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


def _canonicalize_short_row_pool_ids(
    pool_topk: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_tokens: int,
) -> torch.Tensor:
    """Sort each short row's selected pools ascending, fill last.

    A short row selects every pool, but ``persistent_topk`` orders by
    score, so expansion would group the same tokens differently from the
    fast path. Sorting keeps the selected SET and canonicalizes the
    ORDER; rows above the budget pass through untouched (PQ #1631).
    """
    short_rows = seq_lens.to(torch.int32) <= topk_tokens
    if not bool(short_rows.any()):
        return pool_topk
    sentinel = torch.full((), torch.iinfo(pool_topk.dtype).max,
                          dtype=pool_topk.dtype, device=pool_topk.device)
    ranked, _ = torch.sort(torch.where(pool_topk >= 0, pool_topk, sentinel), dim=1)
    fixed = torch.where(ranked == sentinel, -1, ranked)
    return torch.where(short_rows[:, None], fixed, pool_topk)

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
    index_kpool: int = 1,
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
    if index_kpool is not None and index_kpool > 1:
        _fill_short_decode_general_layout(
            topk_indices_buffer[:num_decode_tokens],
            positions[:num_decode_tokens].to(torch.int32),
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

OLD_DECODE_POOL_IDS = '''        # Resolve to token-level indices in the output buffer.
        if index_kpool > 1:
            pool_ids = pool_topk.to(torch.int64)
            n = pool_topk.shape[0]'''

NEW_DECODE_POOL_IDS = '''        # Resolve to token-level indices in the output buffer.
        if index_kpool > 1:
            n = pool_topk.shape[0]'''

OLD_DECODE_EXPAND = '''            out = kpool_ops.expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)'''

NEW_DECODE_EXPAND = '''            # GLM53_KPOOL_SHORT_DECODE_LAYOUT (PQ #1631): a short row selects
            # every pool, so sort the ids before expansion; the fast path
            # writes that same ascending row.
            pool_ids = _canonicalize_short_row_pool_ids(
                pool_topk, dec_seq, topk_tokens
            ).to(torch.int64)
            out = kpool_ops.expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)'''

EDITS = [
    ("insert_layout_helpers", "def _fill_short_decode_causal_indices(\n",
     lambda text: text.replace(
         "def _fill_short_decode_causal_indices(\n",
         NEW_LAYOUT_HELPERS + "def _fill_short_decode_causal_indices(\n")),
    ("fast_path_writes_general_layout", OLD_FAST_PATH_FUNC,
     lambda text: text.replace(OLD_FAST_PATH_FUNC, NEW_FAST_PATH_FUNC)),
    ("fast_path_takes_index_kpool", OLD_DECODE_FAST_CALL,
     lambda text: text.replace(OLD_DECODE_FAST_CALL, NEW_DECODE_FAST_CALL)),
    ("decode_pool_ids_canonicalized_below", OLD_DECODE_POOL_IDS,
     lambda text: text.replace(OLD_DECODE_POOL_IDS, NEW_DECODE_POOL_IDS)),
    ("decode_expand_reads_canonical_ids", OLD_DECODE_EXPAND,
     lambda text: text.replace(OLD_DECODE_EXPAND, NEW_DECODE_EXPAND)),
]


def main() -> None:
    raw = target.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != BASE_INDEXER_SHA256:
        raise SystemExit(f"indexer is not the base image's (sha256 {digest})")
    text = raw.decode()
    for tag, old, apply in EDITS:
        found = text.count(old)
        if found != 1:
            raise SystemExit(f"expected one {tag} anchor, found {found}")
        text = apply(text)
    target.write_text(text)
    print("applied glm53_kpool_short_decode_layout")


main()
