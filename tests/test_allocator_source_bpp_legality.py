"""Exact source-payload legality at the allocator candidate gate.

These tests use shapes and source kinds only; no tensors or GPU kernels are
constructed.  The boundary assertions deliberately use integer payload bytes
as well as bpp so a future floating-point tolerance cannot widen the menu.
"""
from __future__ import annotations

import math

import pytest

from prismaquant import format_registry as fr
from prismaquant.activation_fair_pricing import BRANCH_SOURCE_PASSTHROUGH
from prismaquant.allocator_candidates import (
    SOURCE_BPP_EXCEEDED_REASON,
    SOURCE_BPP_UNKNOWN_REASON,
    build_candidates,
    check_format_applicability,
    source_footprint_owner_for_kind,
    source_format_for_kind,
    summarize_applicability_masks,
)


def _stats(shape: tuple[int, ...]) -> dict[str, object]:
    row: dict[str, object] = {
        "h_trace": 1.0,
        "n_params": math.prod(shape),
        "out_features": shape[-2],
        "in_features": shape[-1],
    }
    if len(shape) == 3:
        row["num_experts"] = shape[0]
    return row


def _measured_cost() -> dict[str, object]:
    return {
        "weight_mse": 1.0e-4,
        "output_mse": 2.0e-4,
        "output_mse_measured": True,
    }


def test_mxfp4_packed_expert_boundary_and_audit_are_byte_exact():
    # MXFP4 re-encodes an MXFP4 source at exactly its payload (4.25 bpp) and
    # stays legal: equality is the boundary. NVFP4 (4.5 bpp) exceeds it by
    # params/32 bytes and is eliminated. (Until 2026-09-25 this pinned the
    # same boundary on two retired codebook rungs, #1304.)
    qname = "model.layers.0.mlp.experts.gate_up_proj"
    shape = (256, 4096, 2048)
    stats = {qname: _stats(shape)}
    costs = {
        qname: {
            "MXFP4": _measured_cost(),
            "NVFP4": _measured_cost(),
        },
    }
    records: list[dict] = []

    candidates = build_candidates(
        stats,
        costs,
        [fr.get_format("MXFP4"), fr.get_format("NVFP4")],
        source_manifest={qname: "mxfp4"},
        mask_records=records,
    )

    assert source_format_for_kind("mxfp4").name == "MXFP4_SOURCE"
    assert [candidate.fmt for candidate in candidates[qname]] == ["MXFP4"]
    mxfp4 = candidates[qname][0]
    assert mxfp4.bits_per_param == 4.25
    # 4.25 == 17/4 bits/parameter, pinned without float arithmetic.
    params = math.prod(shape)
    assert 8 * mxfp4.memory_bytes * 4 == 17 * params

    assert len(records) == 1
    eliminated = records[0]
    assert eliminated["format"] == "NVFP4"
    assert eliminated["reason"] == SOURCE_BPP_EXCEEDED_REASON
    assert eliminated["shape"] == [256, 4096, 2048]
    assert eliminated["source_bpp"] == 4.25
    assert eliminated["candidate_bpp"] == 4.5
    assert eliminated["comparison"] == (
        "candidate_payload_bytes <= source_payload_bytes"
    )
    assert "exact integer bytes with no tolerance" in eliminated["detail"]

    source_bits = eliminated["source_bpp_numerator_bits"]
    candidate_bits = eliminated["candidate_bpp_numerator_bits"]
    assert source_bits * 4 == 17 * params  # 4.25 == 17/4.
    assert candidate_bits * 2 == 9 * params  # 4.5 == 9/2.
    assert eliminated["candidate_payload_bytes"] > eliminated[
        "source_payload_bytes"
    ]
    assert (
        eliminated["candidate_payload_bytes"]
        - eliminated["source_payload_bytes"]
        == params // 32
    )

    audit = summarize_applicability_masks(
        records,
        source_census_present=True,
        source_census_units=1,
    )["source_bpp_legality"]
    assert audit["comparison_arithmetic"] == (
        "exact_integer_bytes_no_float_tolerance"
    )
    assert audit["eliminated_count"] == 1
    assert audit["evaluated"] is True
    assert audit["source_census_units"] == 1
    assert audit["eliminated_candidates"] == [eliminated]
    assert audit["eliminated_candidates"][0]["source_bpp"] == 4.25
    assert audit["eliminated_candidates"][0]["candidate_bpp"] == 4.5


def test_dense_fp8_source_allows_a_lower_rung_and_its_measured_equal_source_format():
    source_kind = "fp8"
    source_format = "FP8_SOURCE"
    qname = "model.layers.0.self_attn.o_proj.fp8"
    shape = (8192, 4096)
    costs = {
        "NVFP4": _measured_cost(),
        source_format: {
            "weight_mse": 0.0,
            "output_mse": 0.0,
            "output_mse_measured": True,
        },
    }
    records: list[dict] = []

    candidates = build_candidates(
        {qname: _stats(shape)},
        {qname: costs},
        [fr.get_format("NVFP4"), fr.get_format(source_format)],
        source_manifest={qname: source_kind},
        mask_records=records,
    )

    resolved_source = source_format_for_kind(source_kind)
    assert resolved_source is not None
    assert resolved_source.name == source_format
    by_format = {candidate.fmt: candidate for candidate in candidates[qname]}
    assert set(by_format) == {"NVFP4", source_format}
    assert records == []
    assert by_format[source_format].memory_bytes == (
        resolved_source.memory_bytes_for_shape(shape)
    )
    assert by_format["NVFP4"].memory_bytes < (
        by_format[source_format].memory_bytes
    )


def test_dense_ue8m0_w8a16_source_is_an_exact_identity_terminal():
    """The raw-resident W8A16 route preserves both source W and BF16 A."""
    qname = "model.layers.0.self_attn.o_proj.fp8_ue8m0"
    shape = (8192, 4096)
    source_format = "FP8_BLOCK_UE8M0_SOURCE"
    records: list[dict] = []

    candidates = build_candidates(
        {qname: _stats(shape)},
        {qname: {"NVFP4": _measured_cost()}},
        [fr.get_format("NVFP4"), fr.get_format(source_format)],
        source_manifest={qname: "fp8_ue8m0"},
        mask_records=records,
    )

    resolved_source = source_format_for_kind("fp8_ue8m0")
    assert resolved_source is not None
    assert resolved_source.name == source_format
    assert not resolved_source.act_quant_changes_input
    by_format = {candidate.fmt: candidate for candidate in candidates[qname]}
    assert set(by_format) == {"NVFP4", source_format}
    terminal = by_format[source_format]
    assert terminal.memory_bytes == resolved_source.memory_bytes_for_shape(shape)
    assert terminal.predicted_dloss == 0.0
    assert terminal.activation_pricing == BRANCH_SOURCE_PASSTHROUGH
    assert by_format["NVFP4"].memory_bytes < terminal.memory_bytes
    assert records == []


def test_unknown_source_kind_fails_closed_before_candidate_construction():
    qname = "model.layers.0.self_attn.q_proj"
    shape = (4096, 2048)
    assert source_format_for_kind("future_fp6") is None
    verdict = check_format_applicability(
        shape,
        "NVFP4",
        qname=qname,
        source_kind="future_fp6",
    )
    assert not verdict.legal
    assert verdict.reason == SOURCE_BPP_UNKNOWN_REASON

    with pytest.raises(ValueError, match="source-bpp legality cannot be established"):
        build_candidates(
            {qname: _stats(shape)},
            {qname: {"NVFP4": _measured_cost()}},
            [fr.get_format("NVFP4")],
            source_manifest={qname: "future_fp6"},
            )


@pytest.mark.parametrize(("source_kind", "source_bpp"), [
    ("f16", 16.0),
    ("f32", 32.0),
])
def test_plain_safetensors_dtypes_derive_their_source_rate(
    source_kind: str,
    source_bpp: float,
):
    shape = (128, 256)
    owner = source_footprint_owner_for_kind(source_kind)
    assert owner is not None
    assert owner.format_name is None
    assert owner.safetensors_dtype == source_kind.upper()

    verdict = check_format_applicability(
        shape,
        "NVFP4",
        qname=f"model.layers.0.source_{source_kind}",
        source_kind=source_kind,
    )
    assert verdict.legal
    assert verdict.provenance is not None
    assert verdict.provenance["source_footprint_owner"] == "safetensors_dtype"
    assert verdict.provenance["source_dtype"] == source_kind.upper()
    assert verdict.provenance["source_bpp"] == source_bpp


def test_audit_distinguishes_no_census_from_zero_eliminations():
    absent = summarize_applicability_masks(
        [], source_census_present=False
    )["source_bpp_legality"]
    evaluated = summarize_applicability_masks(
        [], source_census_present=True, source_census_units=7
    )["source_bpp_legality"]
    assert absent["evaluated"] is False
    assert absent["evaluation_status"] == "not_evaluated_no_source_census"
    assert evaluated["evaluated"] is True
    assert evaluated["evaluation_status"] == "evaluated"
    assert evaluated["source_census_units"] == 7
    assert evaluated["eliminated_count"] == 0


def test_source_census_without_exact_linear_shape_fails_closed():
    qname = "model.layers.0.legacy_rank1"
    with pytest.raises(ValueError, match="exact rank>=2 Linear shape"):
        build_candidates(
            {qname: {"h_trace": 1.0, "n_params": 4096}},
            {qname: {"NVFP4": _measured_cost()}},
            [fr.get_format("NVFP4")],
            source_manifest={qname: "bf16"},
        )
