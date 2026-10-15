"""Exactness pins for the MXFP4 -> MXFP8 widening transcode.

The claim under test is narrow and total: widening changes the width of the
element plane and NOTHING about the numbers on it, and the scale plane is
carried byte-identically. Both halves are pinned exhaustively where the input
space allows it (all 256 byte codes, all 256 scale codes) rather than sampled,
because "exact" is not a statistical property.

Real-checkpoint arms run only when the DSv4-Flash source is present; they read
a handful of small tensor slices from the safetensors headers and never load a
layer. Skipped otherwise so the suite stays hermetic on CI.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from prismaquant.mxfp4_widen import (
    E2M1_VALUES,
    GROUP_SIZE,
    dequantize_mxfp4_source,
    dequantize_mxfp8,
    e2m1_to_e4m3_table,
    mxfp4_source_to_mxfp8,
)


DSV4_SOURCE = Path(
    os.environ.get("DSV4_FLASH_SOURCE",
                   "/home/rob/dq-runs/dsv4-flash-0731/source"))


def _nan_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise-meaningful equality: NaNs must coincide, finites must match."""

    if a.shape != b.shape:
        return False
    a32, b32 = a.to(torch.float32), b.to(torch.float32)
    both_nan = torch.isnan(a32) & torch.isnan(b32)
    return bool(torch.all(both_nan | (a32 == b32)))


# --- the exactness claim itself ---------------------------------------------


def test_every_e2m1_value_is_exact_in_e4m3():
    """The premise the whole module rests on, stated as a test."""

    values = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    roundtrip = values.to(torch.float8_e4m3fn).to(torch.float32)
    assert torch.equal(roundtrip, values), (
        "E2M1 -> E4M3 is only a transcode if every code point survives the "
        "cast; a mismatch here means widening would be a requantization")


def test_widening_table_covers_all_256_byte_codes_in_logical_order():
    table = e2m1_to_e4m3_table()
    assert table.shape == (256, 2)
    assert table.dtype == torch.float8_e4m3fn
    as_f32 = table.to(torch.float32)
    for code in range(256):
        # low nibble is the EVEN logical element, high nibble the odd one
        assert as_f32[code, 0].item() == E2M1_VALUES[code & 0x0F]
        assert as_f32[code, 1].item() == E2M1_VALUES[code >> 4]


def test_dequant_is_bit_equal_over_every_byte_code_and_scale_code():
    """Exhaustive over the full cross product of element and scale codes.

    256 byte codes x 256 E8M0 codes, including 0xFF (NaN) and the subnormal
    ends of the exponent range. This is the property in its strongest form:
    there is no input to the transcode that this misses.
    """

    # One row per scale code; each row holds all 256 byte codes, padded to a
    # whole number of groups so every element in the row shares that scale.
    packed = torch.arange(256, dtype=torch.uint8).repeat(256, 1)
    assert packed.shape[-1] * 2 % GROUP_SIZE == 0
    n_groups = packed.shape[-1] * 2 // GROUP_SIZE
    scale = (torch.arange(256, dtype=torch.uint8)
             .unsqueeze(1).expand(256, n_groups).contiguous()
             .view(torch.float8_e8m0fnu))

    widened = mxfp4_source_to_mxfp8(packed, scale)
    assert _nan_equal(dequantize_mxfp8(widened.weight, widened.weight_scale),
                      dequantize_mxfp4_source(packed, scale))


def test_scale_plane_is_carried_byte_identically():
    packed = torch.randint(0, 256, (8, 64), dtype=torch.uint8)
    scale = torch.randint(0, 256, (8, 4), dtype=torch.uint8).view(
        torch.float8_e8m0fnu)

    widened = mxfp4_source_to_mxfp8(packed, scale)

    assert widened.weight_scale.dtype == torch.float8_e8m0fnu
    assert torch.equal(widened.weight_scale.view(torch.uint8),
                       scale.view(torch.uint8)), (
        "the MXFP8 wire uses the SAME group-of-32 E8M0 grid as the source, so "
        "the scale bytes must survive untouched — recomputing them could "
        "disagree exactly where the source is unusual (0xFF = NaN)")


def test_carried_scale_does_not_alias_the_caller_storage():
    scale = torch.zeros((2, 2), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    widened = mxfp4_source_to_mxfp8(
        torch.zeros((2, 32), dtype=torch.uint8), scale)
    scale.view(torch.uint8).fill_(0x7F)
    assert int(widened.weight_scale.view(torch.uint8).max()) == 0


def test_widened_plane_doubles_the_element_bytes_and_keeps_scale_bytes():
    packed = torch.randint(0, 256, (16, 128), dtype=torch.uint8)
    scale = torch.randint(0, 256, (16, 8), dtype=torch.uint8).view(
        torch.float8_e8m0fnu)

    widened = mxfp4_source_to_mxfp8(packed, scale)

    assert widened.weight.shape == (16, 256)
    assert widened.weight.dtype == torch.float8_e4m3fn
    assert widened.weight.numel() == packed.numel() * 2
    assert widened.weight_scale.numel() == scale.numel()


def test_negative_zero_code_is_normalized_to_positive_zero():
    """The one normalization, pinned so it cannot drift silently."""

    packed = torch.tensor([[0x08] * 16], dtype=torch.uint8)   # both nibbles -0
    scale = torch.tensor([[127]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    widened = mxfp4_source_to_mxfp8(packed, scale)
    signbits = widened.weight.view(torch.uint8) & 0x80
    assert int(signbits.max()) == 0
    assert torch.equal(dequantize_mxfp8(widened.weight, widened.weight_scale),
                       torch.zeros(1, 32))


# --- geometry refusals -------------------------------------------------------


@pytest.mark.parametrize("packed_shape,scale_shape", [
    ((8, 64), (8, 3)),      # too few groups
    ((8, 64), (8, 5)),      # too many groups
    ((8, 64), (4, 8)),      # transposed-ish, numel-incompatible leading dim
    ((8, 64), (64, 8)),     # transposed scale grid
])
def test_mismatched_scale_grid_is_refused(packed_shape, scale_shape):
    packed = torch.zeros(packed_shape, dtype=torch.uint8)
    scale = torch.zeros(scale_shape, dtype=torch.uint8).view(
        torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="scale plane shape"):
        mxfp4_source_to_mxfp8(packed, scale)


def test_ragged_reduce_dim_is_refused():
    packed = torch.zeros((4, 5), dtype=torch.uint8)     # logical K = 10
    scale = torch.zeros((4, 1), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="not a multiple of"):
        mxfp4_source_to_mxfp8(packed, scale)


def test_already_widened_input_is_refused():
    """A transcode of the SOURCE wire must not accept its own output."""

    weight = torch.zeros((4, 64), dtype=torch.float8_e4m3fn)
    scale = torch.zeros((4, 2), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="nibble-pack"):
        mxfp4_source_to_mxfp8(weight, scale)


# --- real checkpoint slices --------------------------------------------------


def _load_pairs(limit: int):
    """(name, packed, scale) for a few real MXFP4 expert tensors."""

    from safetensors import safe_open

    index = DSV4_SOURCE / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    picked: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    # Deliberately mixed: MTP experts (the unit this capability exists for),
    # a body expert, and both projection shapes (w1/w3 square-ish, w2 wide).
    wanted = [
        "mtp.0.ffn.experts.0.w1",
        "mtp.0.ffn.experts.0.w2",
        "mtp.1.ffn.experts.7.w3",
        "mtp.2.ffn.experts.255.w2",
        "layers.5.ffn.experts.0.w1",
    ][:limit]
    handles: dict[str, object] = {}
    for base in wanted:
        wkey, skey = base + ".weight", base + ".scale"
        if wkey not in weight_map or skey not in weight_map:
            continue
        for key in (wkey, skey):
            shard = weight_map[key]
            if shard not in handles:
                handles[shard] = safe_open(
                    str(DSV4_SOURCE / shard), framework="pt", device="cpu")
        picked.append((base,
                       handles[weight_map[wkey]].get_tensor(wkey),
                       handles[weight_map[skey]].get_tensor(skey)))
    return picked


@pytest.mark.skipif(
    not (DSV4_SOURCE / "model.safetensors.index.json").exists(),
    reason="DSv4-Flash source checkpoint not present")
def test_real_checkpoint_slices_widen_exactly():
    pairs = _load_pairs(limit=5)
    assert pairs, "index present but no MXFP4 expert pair resolved"
    for name, packed, scale in pairs:
        # Row slice keeps the reduce dim (and therefore the group grid) whole
        # while keeping the test cheap.
        packed_s, scale_s = packed[:16], scale[:16]
        widened = mxfp4_source_to_mxfp8(packed_s, scale_s)

        assert widened.weight.shape[-1] == packed_s.shape[-1] * 2, name
        assert torch.equal(widened.weight_scale.view(torch.uint8),
                           scale_s.view(torch.uint8)), name
        assert _nan_equal(
            dequantize_mxfp8(widened.weight, widened.weight_scale),
            dequantize_mxfp4_source(packed_s, scale_s)), name


@pytest.mark.skipif(
    not (DSV4_SOURCE / "model.safetensors.index.json").exists(),
    reason="DSv4-Flash source checkpoint not present")
def test_real_checkpoint_scale_grid_matches_the_shared_group_of_32():
    """The geometry assumption, checked against the checkpoint not the docs."""

    for name, packed, scale in _load_pairs(limit=5):
        logical_k = packed.shape[-1] * 2
        assert scale.shape == (packed.shape[0], logical_k // GROUP_SIZE), (
            f"{name}: source scale grid is not one E8M0 byte per "
            f"{GROUP_SIZE} logical elements, so the MXFP8 wire's grid is NOT "
            f"the same grid and a byte-identical carry would be wrong")
        assert packed.dtype in (torch.int8, torch.uint8), name
        assert scale.dtype is torch.float8_e8m0fnu, name


# --- serving honesty ---------------------------------------------------------


def test_grouped_mxfp8_route_is_recorded_as_not_backed():
    """The capability must not be able to drift into a serving claim.

    Widening only applies to units that are MXFP4 in the source, and in
    DSv4-Flash those are all routed-expert stacks. Gridbook's MXFP8 lane is
    dense-only, so a grouped MXFP8 route is UNAUDITED on sm_121. If someone
    audits one and flips this constant, they must also update the evidence
    string — which is the point of pinning both.
    """

    from prismaquant.mxfp4_widen import (
        MXFP8_GROUPED_ROUTE_EVIDENCE,
        MXFP8_GROUPED_ROUTE_STATUS,
    )
    from prismaquant.source_format_contracts import (
        ROUTE_STATUS_BACKED,
        ROUTE_STATUS_PENDING,
    )

    assert MXFP8_GROUPED_ROUTE_STATUS != ROUTE_STATUS_BACKED
    assert MXFP8_GROUPED_ROUTE_STATUS == ROUTE_STATUS_PENDING, (
        "the local spelling must stay inside the allocator's route_status "
        "vocabulary; it is duplicated only to keep this module torch-only")
    assert "dense" in MXFP8_GROUPED_ROUTE_EVIDENCE.lower()


def test_mxfp8_wire_id_is_the_one_the_consumer_routes():
    """Widened bytes are only declarable under an id the consumer knows."""

    from prismaquant.source_format_contracts import REQUANT_WIRE_FORMAT_IDS

    assert REQUANT_WIRE_FORMAT_IDS["MXFP8_UE8M0_G32"] == "mxfp8_e4m3_e8m0_g32"
