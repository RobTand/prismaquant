"""Row-weight clip of the render scorer (issue #159).

The mean-1 normalise-then-clip rule lives in ``prismaquant.render_score``
(``normalize_row_weights`` and ``resolve_fisher_row_weight_clip``). The
native exporter reader (#3480) and the quant-cost Fisher path (#3448) are
retired, so the scorer is its only consumer. These tests pin the env
precedence and the exact arithmetic.
"""

from __future__ import annotations

import pytest
import torch

CANONICAL_ENV = "PRISMAQUANT_FISHER_OUTPUT_MSE_ROW_WEIGHT_CLIP"
ALIAS_ENV = "PRISMAQUANT_FISHER_GPTQ_ROW_WEIGHT_CLIP"
_CPU = torch.device("cpu")


def _clear_clip_env(monkeypatch) -> None:
    monkeypatch.delenv(CANONICAL_ENV, raising=False)
    monkeypatch.delenv(ALIAS_ENV, raising=False)


def _render_path(row_weights: torch.Tensor, n_rows: int) -> torch.Tensor | None:
    from prismaquant.render_score import normalize_row_weights

    return normalize_row_weights(row_weights, n_rows, _CPU)


def _dense_vector() -> torch.Tensor:
    # Mean-1 normalises to [2.4, 0.8 * 7]: a clip of 2.0 binds on the hot
    # element without the post-clip renormalisation hiding it.
    return torch.tensor([3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def test_canonical_env_drives_all_paths(monkeypatch) -> None:
    """The canonical env var sets the scorer clip."""
    _clear_clip_env(monkeypatch)
    monkeypatch.setenv(CANONICAL_ENV, "2.0")
    render_out = _render_path(_dense_vector(), 8)
    assert render_out is not None
    # The clip binds: without it the hot element would be 2.4.
    assert float(render_out[0]) < 2.4


def test_alias_selects_same_value_as_canonical(monkeypatch) -> None:
    """The old env var name keeps working as a documented alias."""
    vec = _dense_vector()
    _clear_clip_env(monkeypatch)
    monkeypatch.setenv(ALIAS_ENV, "2.0")
    via_alias = _render_path(vec, 8)
    _clear_clip_env(monkeypatch)
    monkeypatch.setenv(CANONICAL_ENV, "2.0")
    via_canonical = _render_path(vec, 8)
    assert via_alias is not None and via_canonical is not None
    assert torch.equal(via_alias, via_canonical)
    # And the alias on its own also clips (not silently ignored).
    assert float(via_alias[0]) < 2.4


def test_canonical_wins_when_both_set(monkeypatch) -> None:
    """Precedence matches the historical cost-probe rule: canonical first."""
    _clear_clip_env(monkeypatch)
    monkeypatch.setenv(CANONICAL_ENV, "2.0")
    monkeypatch.setenv(ALIAS_ENV, "33.0")
    render_out = _render_path(_dense_vector(), 8)
    assert render_out is not None
    assert float(render_out[0]) < 2.4


@pytest.mark.parametrize("rows", [[0.0, 0.0], [-1.0, float("nan")]])
def test_degenerate_rows_preserve_each_consumer_policy(monkeypatch, rows) -> None:
    """Scoring retains degenerate rows as zero weights."""
    _clear_clip_env(monkeypatch)
    render_out = _render_path(torch.tensor(rows), 2)
    assert render_out is not None
    assert torch.equal(render_out, torch.zeros(2))


def test_default_weights_are_bit_identical(monkeypatch) -> None:
    """Numeric identity: hand-computed exact outputs, default clip.

    Both vectors use exact binary arithmetic, so torch.equal is exact, not
    approximate. These pass identically before and after the fix; they prove
    the single-sourcing changed nothing numerically.
    """
    _clear_clip_env(monkeypatch)
    # Fully concentrated row over 128 rows: mean-1 value is exactly n_rows,
    # the clip binds, and the post-clip renormalisation restores it exactly.
    hot = torch.zeros(128)
    hot[0] = 128.0
    expected_hot = torch.zeros(128)
    expected_hot[0] = 128.0
    out = _render_path(hot, 128)
    assert out is not None
    assert torch.equal(out, expected_hot)
    # Dense vector below the clip: pure mean-1 normalisation.
    dense = torch.tensor([4.0, 2.0, 2.0, 0.0])
    expected_dense = torch.tensor([2.0, 1.0, 1.0, 0.0])
    out = _render_path(dense, 4)
    assert out is not None
    assert torch.equal(out, expected_dense)


def test_binding_clip_is_exact(monkeypatch) -> None:
    """Pin the clip-then-renormalise arithmetic with exact binary values."""
    _clear_clip_env(monkeypatch)
    monkeypatch.setenv(CANONICAL_ENV, "1.0")
    # Mean-1 gives [2, 0]; the clip binds to [1, 0]; renormalisation by the
    # post-clip mean 0.5 restores [2, 0]. Every step is exact in float32.
    vec = torch.tensor([2.0, 0.0])
    expected = torch.tensor([2.0, 0.0])
    out = _render_path(vec, 2)
    assert out is not None
    assert torch.equal(out, expected)


def test_resolver_defaults_and_fallbacks(monkeypatch) -> None:
    from prismaquant.render_score import resolve_fisher_row_weight_clip

    _clear_clip_env(monkeypatch)
    assert resolve_fisher_row_weight_clip() == pytest.approx(64.0)
    monkeypatch.setenv(ALIAS_ENV, "7.5")
    assert resolve_fisher_row_weight_clip() == pytest.approx(7.5)
    monkeypatch.setenv(CANONICAL_ENV, "3.25")
    assert resolve_fisher_row_weight_clip() == pytest.approx(3.25)
    monkeypatch.setenv(CANONICAL_ENV, "not-a-number")
    assert resolve_fisher_row_weight_clip() == pytest.approx(64.0)
