"""Check short-decode layout, row bounds, and the base-byte refusal."""

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


def _general_layout_row(seq_len: int, pool_size: int = POOL) -> torch.Tensor:
    """Place complete pools first and incomplete-pool tokens after top-k."""
    tail_start = (seq_len // pool_size) * pool_size
    tail_count = seq_len - tail_start
    row = torch.full((TOPK + pool_size - 1,), -1, dtype=torch.int32)
    row[:tail_start] = torch.arange(tail_start, dtype=torch.int32)
    row[TOPK : TOPK + tail_count] = torch.arange(tail_start, seq_len, dtype=torch.int32)
    return row


def _old_fast_path_row(seq_len: int) -> torch.Tensor:
    """The pre-fix fast path row: causal indices, then -1."""
    row = torch.full((WIDTH,), -1, dtype=torch.int32)
    row[:seq_len] = torch.arange(seq_len, dtype=torch.int32)
    return row


@pytest.mark.parametrize("pool_size", [2, 4, 8])
def test_patched_fast_path_matches_general_layout(pool_size: int) -> None:
    """Every short length matches the general row, including padded rows."""
    fill = _load_helpers()["_fill_short_decode_general_layout"]
    seq_lens = list(range(TOPK + 1))
    buf = torch.full((len(seq_lens), TOPK + pool_size - 1), -7, dtype=torch.int32)
    fill(buf, torch.tensor(seq_lens) - 1, TOPK, pool_size)
    expected = torch.stack([_general_layout_row(length, pool_size) for length in seq_lens])
    torch.testing.assert_close(buf, expected, rtol=0, atol=0)


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


def test_short_dispatch_leaves_other_buffer_rows_untouched() -> None:
    dispatch = _load_helpers()["_fill_short_decode_causal_indices"]
    buf = torch.full((3, WIDTH), -7, dtype=torch.int32)
    assert dispatch(buf, torch.tensor([6, 9, 99]), 2, 10, TOPK, POOL)
    torch.testing.assert_close(buf[0], _general_layout_row(7), rtol=0, atol=0)
    torch.testing.assert_close(buf[1], _general_layout_row(10), rtol=0, atol=0)
    assert (buf[2] == -7).all()


@pytest.mark.parametrize("positions", [None, torch.empty(0, dtype=torch.int64)])
def test_missing_positions_do_not_use_the_fast_path(positions) -> None:
    dispatch = _load_helpers()["_fill_short_decode_causal_indices"]
    buf = torch.full((1, WIDTH), -7, dtype=torch.int32)
    assert not dispatch(buf, positions, 1, 1, TOPK, POOL)
    assert (buf == -7).all()


def test_unknown_base_bytes_fail_before_a_write(tmp_path: Path) -> None:
    import runpy

    namespace = runpy.run_path(str(PATCH))
    target = tmp_path / "indexer.py"
    raw = b"not the pinned serving image"
    target.write_bytes(raw)
    namespace["main"].__globals__["target"] = target
    with pytest.raises(SystemExit, match="indexer is not the base image"):
        namespace["main"]()
    assert target.read_bytes() == raw
