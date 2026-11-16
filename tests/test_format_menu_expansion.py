from __future__ import annotations

from prismaquant import allocator
from prismaquant import format_registry as fr
from prismaquant.allocator import (
    filter_candidates_for_profile,
    resolve_target_profile,
)
from prismaquant.allocator_candidates import check_format_applicability
from prismaquant.allocator_solver import Candidate
from prismaquant.export_native_compressed import (
    FORMAT_SCHEME,
    canonicalize_format,
)

VLLM_PROFILE = "vllm_packed_moe"


class _ProfileWithServingDefault:
    def serving_profile_id(self) -> str:
        return VLLM_PROFILE


def _non_bf16_format_menu(requested: list[str], floor_format: str) -> list[str]:
    """The format-menu expansion rule: canonicalize every requested name plus
    the floor, drop BF16, sort.

    Was `kl_sensitivity_probe._production_cache_formats` until 2026-07-30, when
    the probe was walled with the L3 cascade
    (`archive/l3_propagated_2026-07-30/`, re-vet R4). The rule itself is still
    the live contract — `run-pipeline.sh` computes exactly this for
    `CACHE_FORMATS` / `AURA_CACHE_FORMATS` — so the registry canonicalization it
    depends on stays pinned here.
    """
    formats = {
        fr.canonical_format_name(str(fmt).strip().upper())
        for fmt in [*requested, floor_format]
    }
    return sorted(fmt for fmt in formats if fmt != "BF16")


def test_production_cache_formats_include_all_non_bf16_registry_picks():
    assert _non_bf16_format_menu(
        ["NVFP4", "MXFP8_E4M3", "FP8_E4M3", "BF16"],
        "NVFP4",
    ) == ["FP8_E4M3", "MXFP8_E4M3", "NVFP4"]

    assert _non_bf16_format_menu(
        ["MXFP8_E5M2", "FP8_E5M2", "BF16"],
        "NVFP4",
    ) == ["FP8_E5M2", "MXFP8_E5M2", "NVFP4"]


def test_allocator_target_profile_defaults_to_model_profile_config():
    assert resolve_target_profile(_ProfileWithServingDefault()) == VLLM_PROFILE
    assert (
        resolve_target_profile(_ProfileWithServingDefault(), "research")
        == "research"
    )


def test_vllm_profile_allows_dense_fp8_e4m3_but_not_e5m2():
    dense = "model.layers.0.self_attn.q_proj"
    shape = (5120, 17408)

    assert check_format_applicability(
        shape,
        "FP8_E4M3",
        qname=dense,
        source_kind="bf16",
        target_profile=VLLM_PROFILE,
    ).legal

    for fmt in ("MXFP8_E5M2", "FP8_E5M2"):
        verdict = check_format_applicability(
            shape,
            fmt,
            qname=dense,
            source_kind="bf16",
            target_profile=VLLM_PROFILE,
        )
        assert not verdict.legal
        assert verdict.reason == "profile_mismatch"


def test_vllm_profile_keeps_packed_moe_menu_vllm_backed():
    expert = "model.layers.0.mlp.experts.gate_up_proj"
    root_expert = "model.layers.0.experts.gate_up_proj"
    shape = (5120, 17408)

    assert check_format_applicability(
        shape,
        "MXFP8_E4M3",
        qname=expert,
        source_kind="bf16",
        target_profile=VLLM_PROFILE,
    ).legal
    assert check_format_applicability(
        shape,
        "MXFP4",
        qname=root_expert,
        source_kind="bf16",
        target_profile=VLLM_PROFILE,
    ).legal
    assert check_format_applicability(
        shape,
        "FP8_E4M3",
        qname=expert,
        source_kind="bf16",
        target_profile=VLLM_PROFILE,
    ).legal
    for fmt in ("MXFP8_E5M2", "FP8_E5M2"):
        verdict = check_format_applicability(
            shape,
            fmt,
            qname=expert,
            source_kind="bf16",
            target_profile=VLLM_PROFILE,
        )
        assert not verdict.legal
        assert verdict.reason == "profile_mismatch"


def test_allocator_profile_filter_keeps_only_vllm_backed_fp8_choices():
    dense = "model.layers.0.self_attn.q_proj"
    expert = "model.layers.0.mlp.experts.gate_up_proj"
    candidates = {
        dense: [
            Candidate("NVFP4", 4.5, 100, 1.0),
            Candidate("FP8_E4M3", 8.5, 190, 0.1),
            Candidate("FP8_E5M2", 8.5, 190, 0.1),
            Candidate("MXFP8_E5M2", 8.25, 180, 0.2),
            Candidate("BF16", 16.0, 512, 0.0),
        ],
        expert: [
            Candidate("NVFP4", 4.5, 100, 1.0),
            Candidate("MXFP4", 4.25, 96, 0.9),
            Candidate("FP8_E4M3", 8.5, 190, 0.1),
            Candidate("MXFP8_E4M3", 8.25, 180, 0.2),
            Candidate("BF16", 16.0, 512, 0.0),
        ],
    }

    filtered = filter_candidates_for_profile(candidates, VLLM_PROFILE)

    assert [c.fmt for c in filtered[dense]] == [
        "NVFP4",
        "FP8_E4M3",
        "BF16",
    ]
    assert [c.fmt for c in filtered[expert]] == [
        "NVFP4",
        "MXFP4",
        "FP8_E4M3",
        "MXFP8_E4M3",
        "BF16",
    ]


def test_export_canonicalizes_and_configures_dense_fp8_e4m3():
    assert canonicalize_format("FP8_E4M3") == "FP8_E4M3"
    assert canonicalize_format(
        fr.get_format("FP8_E4M3").autoround_config()
    ) == "FP8_E4M3"
    assert FORMAT_SCHEME["FP8_E4M3"]["format"] == "float-quantized"
    assert FORMAT_SCHEME["FP8_E4M3"]["weights"]["strategy"] == "channel"
    assert FORMAT_SCHEME["FP8_E4M3"]["input_activations"]["strategy"] == "token"


def test_e5m2_is_parsed_then_gated_by_serving_profile():
    assert canonicalize_format("FP8_E5M2") == "FP8_E5M2"
    assert (
        canonicalize_format(fr.get_format("MXFP8_E5M2").autoround_config())
        == "MXFP8_E5M2"
    )
    assert "FP8_E5M2" not in FORMAT_SCHEME
    verdict = check_format_applicability(
        (128, 128),
        "FP8_E5M2",
        qname="model.layers.0.self_attn.o_proj",
        target_profile=VLLM_PROFILE,
    )
    assert not verdict.legal
    assert verdict.reason == "profile_mismatch"
    assert "not enabled for dense Linears" in verdict.detail


def test_assignment_override_helpers_stay_deleted():
    assert not hasattr(allocator, "apply_visual_format_override")
    assert not hasattr(allocator, "apply_mtp_format_override")
