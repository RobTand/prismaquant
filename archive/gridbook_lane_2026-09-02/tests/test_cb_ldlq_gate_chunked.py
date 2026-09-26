"""Chunked per-expert holdout-gate scoring == monolithic full-stack scoring.

The 2026-08-08 GB10 production-shape canary OOM'd because the holdout gate
reconstructed the FULL packed stack twice (raw + candidate, ~16 GiB fp32 each
on the DSv4 fused gate_up 256x4096x4096 stack).  The gate now scores
expert-slice-by-expert-slice.  Reconstruction is elementwise per row (codebook
gather + per-row scale multiply, no cross-expert reduction) and the gate MSE
is a within-expert reduction, so the chunked scoring must be BITWISE-identical
to the monolithic path — these tests assert that property on synthetic packed
fields across shapes, fp4/fp8 grids, and both scale codings.
"""

from __future__ import annotations

import pytest
import torch

from prismaquant import nvfp4_cb_formats as cb


def _make_case(grid: str, k: int, scale_coding: str, *, experts: int = 16,
               rows: int = 4, columns: int = 256, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(experts, rows, columns, generator=generator) * 0.25
    col_weights = (
        torch.rand(experts, 1, columns, generator=generator) + 0.05
    )
    fields = cb.nvfp4_cb_fields(
        weight, k, grid=grid, mode="product", col_weights=col_weights,
        scale_coding=scale_coding,
    )
    activations = tuple(
        torch.randn(32, columns, generator=generator)
        for _ in range(experts)
    )
    return weight, fields, col_weights, activations


_CASES = (
    ("fp4", 12, cb.SCALE_CODING_V1),
    ("fp4", 12, cb.SCALE_CODING_TWO_TIER),
    ("fp8", 28, cb.SCALE_CODING_V1),
)


@pytest.mark.parametrize(("grid", "k", "scale_coding"), _CASES)
def test_expert_slice_reconstruction_bitwise_equals_full_stack(
    grid: str, k: int, scale_coding: str,
):
    weight, fields, _cw, _acts = _make_case(grid, k, scale_coding)
    experts, rows, columns = map(int, weight.shape)
    full = cb.nvfp4_cb_reconstruct(fields, k, grid=grid, mode="product")
    assert tuple(full.shape) == (experts, rows, columns)
    for expert in range(experts):
        sliced = cb.reconstruct_packed_cb_expert(
            fields, expert, rows, columns, k=k, grid=grid, mode="product",
        )
        assert torch.equal(sliced, full[expert]), (
            f"expert {expert} slice reconstruction diverged from the "
            f"monolithic stack ({grid}/{scale_coding})"
        )


@pytest.mark.parametrize(
    ("grid", "k", "scale_coding"),
    _CASES + (("fp4", 12, cb.SCALE_CODING_V1),),
)
@pytest.mark.parametrize(("rows", "columns"), ((4, 256), (2, 512)))
def test_chunked_gate_scoring_bitwise_equals_monolithic(
    grid: str, k: int, scale_coding: str, rows: int, columns: int,
):
    weight, fields, col_weights, activations = _make_case(
        grid, k, scale_coding, rows=rows, columns=columns,
        seed=rows * 1000 + columns,
    )
    candidate = cb.ldlq_reassign_cb_fields(
        weight, fields, col_weights, activations,
        grid=grid, mode="product",
    )
    # Monolithic reference: full-stack reconstruction of both arms, scored
    # with the shared per-expert activation-MSE reduction (the exact
    # pre-chunking gate math).
    raw_full = cb.nvfp4_cb_reconstruct(
        fields, k, grid=grid, mode="product").to(weight.dtype)
    cand_full = cb.nvfp4_cb_reconstruct(
        candidate, k, grid=grid, mode="product").to(weight.dtype)
    raw_ref = cb._ldlq_per_expert_activation_mse(
        weight, raw_full, activations)
    cand_ref = cb._ldlq_per_expert_activation_mse(
        weight, cand_full, activations)

    raw_chunked, cand_chunked = cb._ldlq_packed_gate_activation_mses(
        weight, fields, candidate, activations,
        k=k, grid=grid, mode="product",
    )
    # Bitwise float equality, not approximate: the chunked path performs the
    # identical elementwise ops on identical per-expert operands.
    assert raw_chunked == raw_ref
    assert cand_chunked == cand_ref


def test_chunked_scorer_candidate_is_raw_reuses_raw_values():
    weight, fields, _cw, activations = _make_case(
        "fp4", 12, cb.SCALE_CODING_V1)
    raw_vals, ldlq_vals = cb._ldlq_packed_gate_activation_mses(
        weight, fields, fields, activations,
        k=12, grid="fp4", mode="product", candidate_is_raw=True,
    )
    assert raw_vals == ldlq_vals
    # And the reuse is itself bitwise-identical to scoring the raw arm twice.
    raw_again, ldlq_again = cb._ldlq_packed_gate_activation_mses(
        weight, fields, fields, activations,
        k=12, grid="fp4", mode="product", candidate_is_raw=False,
    )
    assert raw_vals == raw_again
    assert ldlq_vals == ldlq_again


def test_chunked_scorer_empty_rows_sentinel_matches_monolithic():
    weight, fields, _cw, activations = _make_case(
        "fp4", 12, cb.SCALE_CODING_V1)
    activations = (torch.empty(0, weight.shape[-1]),) + activations[1:]
    full = cb.nvfp4_cb_reconstruct(
        fields, 12, grid="fp4", mode="product").to(weight.dtype)
    ref = cb._ldlq_per_expert_activation_mse(weight, full, activations)
    chunked_raw, chunked_ldlq = cb._ldlq_packed_gate_activation_mses(
        weight, fields, fields, activations,
        k=12, grid="fp4", mode="product", candidate_is_raw=True,
    )
    assert chunked_raw == ref
    assert chunked_raw[0] == float("inf")
    assert chunked_ldlq[0] == float("inf")


@pytest.mark.parametrize(("grid", "k"), (("fp4", 12), ("fp8", 28)))
def test_gated_packed_fields_and_telemetry_match_monolithic_reference(
    monkeypatch: pytest.MonkeyPatch, grid: str, k: int,
):
    """End-to-end: the holdout gate ships the exact splice its (now chunked)
    scores select, and its telemetry lists equal the monolithic reference."""
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_GATE", "holdout")
    weight, fields, col_weights, activations = _make_case(
        grid, k, cb.SCALE_CODING_V1, seed=7)
    gated_fields, info = cb.ldlq_reassign_cb_fields_gated(
        weight, fields, col_weights, activations,
        grid=grid, mode="product", k=k,
    )
    assert info["gate"] in {"ldlq_kept_all", "raw_kept_all", "mixed_per_expert"}

    # Reference: replicate the holdout split + half-fit candidate, then score
    # both arms monolithically (full-stack reconstructions).
    splits = cb._ldlq_holdout_splits(activations)
    assert all(split is not None for split in splits)
    fit_rows = tuple(split[0] for split in splits)
    score_rows = tuple(split[1] for split in splits)
    candidate = cb.ldlq_reassign_cb_fields(
        weight, fields, col_weights, fit_rows, grid=grid, mode="product",
    )
    raw_full = cb.nvfp4_cb_reconstruct(
        fields, k, grid=grid, mode="product").to(weight.dtype)
    cand_full = cb.nvfp4_cb_reconstruct(
        candidate, k, grid=grid, mode="product").to(weight.dtype)
    raw_ref = cb._ldlq_per_expert_activation_mse(weight, raw_full, score_rows)
    cand_ref = cb._ldlq_per_expert_activation_mse(
        weight, cand_full, score_rows)

    assert info["raw_mse_per_expert"] == raw_ref
    assert info["ldlq_mse_per_expert"] == cand_ref

    keep = [c < r for r, c in zip(raw_ref, cand_ref)]
    rows = int(weight.shape[1])
    for expert, kept in enumerate(keep):
        source = candidate if kept else fields
        assert torch.equal(
            gated_fields["indices"][expert * rows:(expert + 1) * rows],
            source["indices"][expert * rows:(expert + 1) * rows],
        ), f"expert {expert} bytes did not come from the winning arm"
