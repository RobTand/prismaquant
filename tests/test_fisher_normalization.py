"""Fisher h_trace must be normalized by the GLOBAL calib token count.

Synthetic reproducer: two identical Linears with identical gradient
statistics over the same calibration run. The "dense" one sees every token
(most carrying zero gradient); the "expert" one only sees the tokens routed
to it — the very tokens that carry the gradient. Tokens never routed to an
expert contribute zero gradient, so both rows have the SAME empirical
Fisher. The old per-row `h_trace_raw / n_tokens_seen` divided the expert by
its routed count only, inflating it by (global/routed).
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from prismaquant.probe_statistics import finalize_fisher_stats


def _h_trace_raw(x: torch.Tensor, gy: torch.Tensor) -> float:
    """The probe's per-token trace accumulator: sum_t ||gy_t||^2 * ||x_t||^2."""
    return float((gy.pow(2).sum(dim=1) * x.pow(2).sum(dim=1)).sum().item())


def test_h_trace_equal_for_equal_gradient_mass_after_global_normalization():
    torch.manual_seed(0)
    global_tokens, routed_tokens = 32, 4  # the expert is routed 1/8 of tokens
    dense = nn.Linear(16, 8, bias=False)
    expert = nn.Linear(16, 8, bias=False)
    expert.load_state_dict(dense.state_dict())

    x = torch.randn(global_tokens, 16)
    # Identical per-token gradients on the routed tokens; zero elsewhere
    # (a token never routed to the expert contributes zero gradient).
    gy = torch.zeros(global_tokens, 8)
    gy[:routed_tokens] = torch.randn(routed_tokens, 8)

    y_dense = dense(x)                      # dense row: all 32 tokens
    y_dense.backward(gy)
    y_expert = expert(x[:routed_tokens])    # expert row: its 4 routed tokens
    y_expert.backward(gy[:routed_tokens])

    stats = {
        "dense": {"h_trace_raw": _h_trace_raw(x, gy),
                  "h_w2_sum_raw": 0.0, "n_tokens_seen": global_tokens},
        "expert": {"h_trace_raw": _h_trace_raw(x[:routed_tokens],
                                               gy[:routed_tokens]),
                   "h_w2_sum_raw": 0.0, "n_tokens_seen": routed_tokens},
    }
    # Same gradient mass -> same raw accumulator (and same weight grads).
    assert stats["expert"]["h_trace_raw"] == pytest.approx(
        stats["dense"]["h_trace_raw"])
    assert torch.allclose(dense.weight.grad, expert.weight.grad)

    # OLD normalization (per-row n_tokens_seen), computed inline for
    # contrast: the expert looks (global/routed) = 8x more sensitive.
    old = {n: s["h_trace_raw"] / s["n_tokens_seen"] for n, s in stats.items()}
    assert old["expert"] == pytest.approx(
        old["dense"] * global_tokens / routed_tokens)

    finalize_fisher_stats(stats, global_tokens)
    assert stats["expert"]["h_trace"] == pytest.approx(stats["dense"]["h_trace"])
    assert stats["dense"]["h_trace"] == pytest.approx(
        stats["dense"]["h_trace_raw"] / global_tokens)
    # n_tokens_seen stays raw (the routed count).
    assert stats["expert"]["n_tokens_seen"] == routed_tokens
    assert stats["expert"]["h_trace_norm_tokens"] == global_tokens


def test_rows_without_token_counter_use_global_count_not_one():
    """Stat rows filled by the batched MoE-block flush path never increment
    n_tokens_seen; the old finalize divided them by max(0, 1) == 1."""
    stats = {"expert": {"h_trace_raw": 64.0, "h_w2_sum_raw": 0.0,
                        "n_tokens_seen": 0}}
    finalize_fisher_stats(stats, 32)
    assert stats["expert"]["h_trace"] == pytest.approx(2.0)
