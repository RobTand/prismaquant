"""The short-decode fast path must write the general-path sparse-index row.

Reproduces prismaquant#1631. The serving image builds the decode index two
ways. The eager fast path writes the causal row ``[0 .. pos, -1 ..]``. The
general path expands the selected pools into columns ``[0 .. 2047]`` and
writes the incomplete pool at columns ``[2048 .. 2050]``. CUDA-graph capture
freezes the general path, so every FULL decode graph replays it, even for
short contexts. The rows hold the same tokens in a different grouping.

The fix lives in the serving image, not here. The patch script carries two
pure-torch helpers. This test executes those shipped bytes and checks their
rows against the kernel layout. Before the patch set exists the loader
fails. After it the rows match.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

PATCH = (
    Path(__file__).resolve().parents[1]
    / "prismaquant"
    / "serving_runtime_patches"
    / "glm53_kpool_short_decode_layout"
    / "patch_kpool_short_decode_layout.py"
)

TOPK = 2048
POOL = 4
WIDTH = TOPK + POOL - 1


def _load_helpers() -> dict:
    """Execute the helper blocks the patch script ships."""
    src = PATCH.read_text()
    tree = ast.parse(src)
    blocks = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in ("NEW_LAYOUT_HELPERS", "NEW_FAST_PATH_FUNC")
            for t in node.targets
        )
    ]
    assert blocks, "patch script carries no helper block"
    namespace: dict = {"torch": torch, "_fill_causal_indices": _base_causal_fill}
    for block in blocks:
        exec(block, namespace)  # noqa: S102 - the test executes the shipped patch bytes
    return namespace


def _base_causal_fill(rows: torch.Tensor, positions: torch.Tensor) -> None:
    """Mirror the base file's unchanged `_fill_causal_indices` helper."""
    causal_range = torch.arange(rows.shape[1], dtype=torch.int32)
    positions = positions.to(torch.int32)
    rows[:] = causal_range[None, :]
    rows[causal_range[None, :] > positions[:, None]] = -1


def _general_layout_row(seq_len: int) -> torch.Tensor:
    """Reference row: expanded pools in order, -1 fill, tail at 2048."""
    pool_len = seq_len // POOL
    tail_start = pool_len * POOL
    tail_count = seq_len - tail_start
    row = torch.full((WIDTH,), -1, dtype=torch.int32)
    row[:tail_start] = torch.arange(tail_start, dtype=torch.int32)
    row[TOPK : TOPK + tail_count] = torch.arange(tail_start, seq_len, dtype=torch.int32)
    return row


def _old_fast_path_row(seq_len: int) -> torch.Tensor:
    """The pre-fix fast path row: causal indices, then -1."""
    row = torch.full((WIDTH,), -1, dtype=torch.int32)
    row[:seq_len] = torch.arange(seq_len, dtype=torch.int32)
    return row


SHORT_LENGTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 100, 101, 1023, 1024, 1025, 2044, 2045, 2046, 2047, 2048]


def test_old_fast_path_differs_from_general_layout_only_past_pool_edge() -> None:
    """Characterize the defect: causal and general rows differ exactly then."""
    for seq_len in SHORT_LENGTHS:
        same = bool(((_old_fast_path_row(seq_len) == _general_layout_row(seq_len)).all()))
        assert same == (seq_len % POOL == 0), seq_len


def test_patched_fast_path_matches_general_layout() -> None:
    """The shipped fast path writes the general row for every short length."""
    helpers = _load_helpers()
    fill = helpers["_fill_short_decode_general_layout"]
    for seq_len in SHORT_LENGTHS:
        buf = torch.full((1, WIDTH), -7, dtype=torch.int32)
        fill(buf, torch.tensor([seq_len - 1]), TOPK, POOL)
        assert bool(((buf[0] == _general_layout_row(seq_len)).all())), seq_len


def test_patched_fast_path_matches_general_layout_batched() -> None:
    """Mixed short rows in one batch each match the reference row."""
    helpers = _load_helpers()
    fill = helpers["_fill_short_decode_general_layout"]
    seq_lens = [3, 4, 7, 8, 1025, 2048]
    buf = torch.full((len(seq_lens), WIDTH), -7, dtype=torch.int32)
    fill(buf, torch.tensor([s - 1 for s in seq_lens]), TOPK, POOL)
    for i, seq_len in enumerate(seq_lens):
        assert bool(((buf[i] == _general_layout_row(seq_len)).all())), seq_len


def test_patched_fast_path_dispatches_on_kpool() -> None:
    """The shipped predicate keeps causal rows without kpool, general with it."""
    helpers = _load_helpers()
    dispatch = helpers["_fill_short_decode_causal_indices"]
    buf = torch.full((2, WIDTH), -7, dtype=torch.int32)
    assert dispatch(buf, torch.tensor([6, 9]), 2, 10, TOPK, 1) is True
    assert bool(((buf[0] == _old_fast_path_row(7)).all()))
    assert bool(((buf[1] == _old_fast_path_row(10)).all()))
    assert dispatch(buf, torch.tensor([6, 9]), 2, 10, TOPK, 4) is True
    assert bool(((buf[0] == _general_layout_row(7)).all()))
    assert bool(((buf[1] == _general_layout_row(10)).all()))
    assert dispatch(buf, torch.tensor([6]), 1, TOPK + 1, TOPK, 4) is False


def test_canonicalize_sorts_short_rows_and_keeps_fill_last() -> None:
    """Shuffled pool ids sort ascending; fill stays last; long rows stay."""
    helpers = _load_helpers()
    canonicalize = helpers["_canonicalize_short_row_pool_ids"]
    short = torch.tensor([1, 0] + [-1] * 510, dtype=torch.int32)
    full = torch.arange(511, -1, -1, dtype=torch.int32)
    long = torch.arange(511, -1, -1, dtype=torch.int32)
    pool_topk = torch.stack([short, full, long])
    seq_lens = torch.tensor([7, 2048, 3000], dtype=torch.int32)
    out = canonicalize(pool_topk, seq_lens, TOPK)
    assert out[0, :2].tolist() == [0, 1]
    assert (out[0, 2:] == -1).all()
    assert out[1].tolist() == list(range(512))
    assert out[2].tolist() == long.tolist()


def test_canonicalize_without_short_rows_returns_input() -> None:
    """No short row means no work and no copy."""
    helpers = _load_helpers()
    canonicalize = helpers["_canonicalize_short_row_pool_ids"]
    pool_topk = torch.arange(511, -1, -1, dtype=torch.int32).unsqueeze(0)
    out = canonicalize(pool_topk, torch.tensor([3000], dtype=torch.int32), TOPK)
    assert out is pool_topk
