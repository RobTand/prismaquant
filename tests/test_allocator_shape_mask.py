"""Tests for the allocator's kernel-shape-aware format masking.

Runtime GEMM kernels for NVFP4 / MXFP8 have alignment constraints that
the knapsack DP must respect — otherwise the solver picks an
unservable format and the artifact fails at vLLM load. The
`_format_kernel_supports_shape` helper gates each format candidate on
its kernel's shape rules before the DP sees it.

These tests pin the known rules so silent kernel changes don't
regress the shape-mask:

  - CUTLASS mm_mxfp8: N ≥ 128, K ≥ 128, K % 32 == 0
  - CUTLASS mm_nvfp4: K % 16 == 0 (group_size)
  - BF16: no shape constraint

Plus an end-to-end check that build_candidates drops the MXFP8
candidate for a Linear whose out_features < 128 (the DeltaNet
in_proj_a shape on Qwen3.6-27B that triggered the Pareto detour).
"""
from __future__ import annotations

from prismaquant import format_registry as fr
from prismaquant.allocator import (
    _format_kernel_supports_shape,
    build_candidates,
)


# ---------------------------------------------------------------------------
# Unit: per-format shape rules
# ---------------------------------------------------------------------------

def test_mxfp8_rejects_small_n():
    # DeltaNet in_proj_a: (5120, 48) — 48 < 128
    assert _format_kernel_supports_shape("MXFP8_E4M3", 5120, 48) is False


def test_mxfp8_rejects_small_k():
    assert _format_kernel_supports_shape("MXFP8_E4M3", 64, 5120) is False


def test_mxfp8_rejects_k_not_divisible_by_32():
    # K = 130 is ≥ 128 but not divisible by 32
    assert _format_kernel_supports_shape("MXFP8_E4M3", 130, 5120) is False


def test_mxfp8_accepts_standard_attention():
    # (5120, 10240) — DeltaNet in_proj_qkv
    assert _format_kernel_supports_shape("MXFP8_E4M3", 5120, 10240) is True


def test_mxfp8_accepts_mlp_shapes():
    # Qwen3.6-27B MLP: (5120, 17408) gate/up, (17408, 5120) down
    assert _format_kernel_supports_shape("MXFP8_E4M3", 5120, 17408) is True
    assert _format_kernel_supports_shape("MXFP8_E4M3", 17408, 5120) is True


def test_nvfp4_rejects_k_not_divisible_by_16():
    # NVFP4 group_size = 16
    assert _format_kernel_supports_shape("NVFP4", 17, 128) is False


def test_nvfp4_accepts_standard_transformer_shapes():
    # Every hidden dim in Qwen3.5/3.6 family is divisible by 16
    for (in_f, out_f) in [(5120, 17408), (5120, 10240), (6144, 5120),
                          (5120, 48), (5120, 6144), (2048, 512)]:
        assert _format_kernel_supports_shape("NVFP4", in_f, out_f) is True, \
            f"NVFP4 should support ({in_f}, {out_f})"


def test_bf16_accepts_everything():
    for (in_f, out_f) in [(5120, 48), (48, 48), (1, 1), (5120, 10240)]:
        assert _format_kernel_supports_shape("BF16", in_f, out_f) is True


def test_unknown_format_defaults_to_true():
    """Unknown format names must default to supported — we don't want
    to silently drop new formats that haven't been shape-profiled yet."""
    assert _format_kernel_supports_shape("EXPERIMENTAL_FP2", 5120, 48) is True


# ---------------------------------------------------------------------------
# build_candidates: mxfp8 dropped for small-N Linear
# ---------------------------------------------------------------------------

def test_build_candidates_drops_mxfp8_on_small_n_shape():
    """A Linear with out_features=48 (DeltaNet in_proj_a) should have
    MXFP8 removed from its candidate list, leaving NVFP4 + BF16."""
    stats = {
        "model.layers.0.linear_attn.in_proj_a": {
            "h_trace": 0.5,
            "n_params": 5120 * 48,
            "in_features": 5120,
            "out_features": 48,
        },
        "model.layers.0.mlp.gate_proj": {
            "h_trace": 0.8,
            "n_params": 5120 * 17408,
            "in_features": 5120,
            "out_features": 17408,
        },
    }
    costs = {
        "model.layers.0.linear_attn.in_proj_a": {
            "NVFP4":      {"weight_mse": 0.02, "predicted_dloss": 0.005},
            "MXFP8_E4M3": {"weight_mse": 0.005, "predicted_dloss": 0.001},
            "BF16":       {"weight_mse": 0.0, "predicted_dloss": 0.0},
        },
        "model.layers.0.mlp.gate_proj": {
            "NVFP4":      {"weight_mse": 0.02, "predicted_dloss": 0.008},
            "MXFP8_E4M3": {"weight_mse": 0.005, "predicted_dloss": 0.002},
            "BF16":       {"weight_mse": 0.0, "predicted_dloss": 0.0},
        },
    }
    specs = [fr.REGISTRY["NVFP4"], fr.REGISTRY["MXFP8_E4M3"], fr.REGISTRY["BF16"]]

    cands = build_candidates(stats, costs, specs)

    # in_proj_a (out=48): MXFP8 should be dropped.
    a_formats = [c.fmt for c in cands["model.layers.0.linear_attn.in_proj_a"]]
    assert "MXFP8_E4M3" not in a_formats, (
        f"MXFP8 should be shape-masked for (5120, 48); got {a_formats}"
    )
    assert "NVFP4" in a_formats and "BF16" in a_formats

    # gate_proj (5120, 17408): MXFP8 should be kept.
    g_formats = [c.fmt for c in cands["model.layers.0.mlp.gate_proj"]]
    assert "MXFP8_E4M3" in g_formats
