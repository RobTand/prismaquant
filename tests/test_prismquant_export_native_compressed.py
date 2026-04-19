"""Tests for the native compressed-tensors exporter.

Covers the math (NVFP4 / FP8 round-trip) and the wire-format
plumbing (`_to_vllm_internal_name`, `build_quantization_config`)
that has to stay in sync with vLLM's compressed-tensors loader.
"""
from __future__ import annotations

import unittest

import torch

from prismquant.export_native_compressed import (
    DEFAULT_INPUT_GLOBAL_SCALE,
    FLOAT_TO_E2M1,
    FP8_E4M3_MAX,
    NVFP4_MAX,
    PER_EXPERT_MOE_REGEX,
    _quantize_2d,
    _quantize_3d_packed,
    _round_to_codebook,
    _to_vllm_internal_name,
    build_quantization_config,
    canonicalize_format,
    pack_fp4_indices,
    quantize_dequantize_fp8_dynamic,
    quantize_dequantize_fp8_dynamic_packed,
    quantize_dequantize_nvfp4,
    quantize_dequantize_nvfp4_packed,
)


def _nvfp4_dequantize(weight_packed, weight_scale_fp8, weight_global_scale_divisor):
    """Reproduce vLLM's NVFP4 dequant convention to verify round-trip.
    The on-disk `weight_global_scale` is `1/global_real`; vLLM inverts
    on load. Per-element dequant: `codebook[idx] * fp8_scale * global_real`.
    """
    rows = weight_packed.shape[0]
    cols = weight_packed.shape[1] * 2
    cb = torch.tensor(FLOAT_TO_E2M1, dtype=torch.float32)
    lo = (weight_packed & 0xF).long()
    hi = ((weight_packed >> 4) & 0xF).long()
    idx = torch.stack([lo, hi], dim=-1).reshape(rows, cols)
    abs_idx = idx & 0x7
    sign = -((idx >> 3).to(torch.float32) * 2 - 1)
    vals = sign * cb[abs_idx]
    fp8_per_col = (
        weight_scale_fp8.float()
        .unsqueeze(-1)
        .expand(-1, -1, cols // weight_scale_fp8.shape[1])
        .reshape(rows, cols)
    )
    global_real = 1.0 / weight_global_scale_divisor.item()
    return vals * fp8_per_col * global_real


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_nvfp4_2d_roundtrip_mse_small(self):
        W = torch.randn(64, 128) * 0.1
        wp, ws, wg = quantize_dequantize_nvfp4(W)
        self.assertEqual(wp.dtype, torch.uint8)
        self.assertEqual(ws.dtype, torch.float8_e4m3fn)
        self.assertEqual(wg.dtype, torch.float32)
        self.assertEqual(tuple(wp.shape), (64, 64))
        self.assertEqual(tuple(ws.shape), (64, 8))
        self.assertEqual(tuple(wg.shape), (1,))
        # fp8 scale must use the FP8 representable range, not be
        # squashed into [0, 1] (the latter loses precision).
        self.assertGreater(ws.float().max().item(), 32.0,
                           "fp8 scale appears to be normalized to [0,1]; "
                           "vLLM's NVFP4 path expects the full FP8 range")

        dequant = _nvfp4_dequantize(wp, ws, wg)
        mse = (W - dequant).pow(2).mean().item()
        self.assertLess(mse, 1e-3,
                        f"NVFP4 round-trip MSE {mse:.3e} too large")
        # max-abs preserved (NVFP4 has explicit ±6 codes covering the peak)
        self.assertAlmostEqual(
            dequant.abs().max().item(),
            W.abs().max().item(),
            places=3,
        )

    def test_nvfp4_packed_per_expert_global_scale(self):
        # Each expert's global_scale is independent.
        E, M, N = 4, 32, 64
        P = torch.randn(E, M, N) * 0.05
        wp, ws, wg = quantize_dequantize_nvfp4_packed(P)
        self.assertEqual(tuple(wp.shape), (E, M, N // 2))
        self.assertEqual(tuple(ws.shape), (E, M, N // 16))
        self.assertEqual(tuple(wg.shape), (E,))
        # Distinct experts → distinct per-tensor scales.
        self.assertGreater(wg.unique().numel(), 1)

    def test_fp8_dynamic_2d_per_channel_scale(self):
        W = torch.randn(64, 128) * 0.1
        w, s = quantize_dequantize_fp8_dynamic(W)
        self.assertEqual(w.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(s.shape), (64, 1))
        self.assertEqual(s.dtype, torch.float32)
        self.assertFalse(torch.isnan(w.float()).any().item(),
                         "fp8 cast NaN — likely overflow in scale")
        # Round-trip MSE
        dequant = w.float() * s
        mse = (W - dequant).pow(2).mean().item()
        self.assertLess(mse, 1e-4)

    def test_fp8_dynamic_packed_3d(self):
        E, M, N = 4, 32, 64
        P = torch.randn(E, M, N) * 0.1
        w, s = quantize_dequantize_fp8_dynamic_packed(P)
        self.assertEqual(tuple(w.shape), (E, M, N))
        self.assertEqual(tuple(s.shape), (E, M, 1))


class TestPackBits(unittest.TestCase):
    def test_round_to_codebook_signed(self):
        # Known mapping: 0→0, 0.5→1, 1.0→2, 6.0→7, -6.0→15
        v = torch.tensor([0.0, 0.5, 1.0, 6.0, -6.0])
        idx = _round_to_codebook(v)
        self.assertEqual(idx.tolist(), [0, 1, 2, 7, 15])

    def test_pack_fp4_two_per_byte(self):
        # Indices 1, 2 packed as low=1, high=2 → byte 0x21 = 33
        idx = torch.tensor([[1, 2, 3, 4]])
        packed = pack_fp4_indices(idx, 4)
        self.assertEqual(packed.shape, torch.Size([1, 2]))
        self.assertEqual(packed[0, 0].item(), (1 | (2 << 4)))
        self.assertEqual(packed[0, 1].item(), (3 | (4 << 4)))


class TestRecipeParsing(unittest.TestCase):
    def test_canonicalize_autoround_dict(self):
        nv = {"bits": 4, "data_type": "nv_fp"}
        mx8 = {"bits": 8, "data_type": "mx_fp"}
        bf = {"bits": 16, "data_type": "float"}
        self.assertEqual(canonicalize_format(nv), "NVFP4")
        self.assertEqual(canonicalize_format(mx8), "MXFP8")
        self.assertEqual(canonicalize_format(bf), "BF16")
        # mx_fp/4 collapses to NVFP4 (only 4-bit format vLLM-served).
        self.assertEqual(canonicalize_format({"bits": 4, "data_type": "mx_fp"}), "NVFP4")


class TestVLLMInternalNaming(unittest.TestCase):
    """vLLM's qwen3_5 hf_to_vllm_mapper transforms source HF names to
    internal module names. The exporter's `quantization_config` targets
    + ignore must match the INTERNAL form so `find_matched_target`
    succeeds."""

    def test_text_only_recipe_naming_remap(self):
        self.assertEqual(
            _to_vllm_internal_name("model.layers.0.linear_attn.in_proj_qkv"),
            "language_model.model.layers.0.linear_attn.in_proj_qkv",
        )
        self.assertEqual(
            _to_vllm_internal_name("model.embed_tokens"),
            "language_model.model.embed_tokens",
        )

    def test_lm_head_remap(self):
        self.assertEqual(
            _to_vllm_internal_name("lm_head"),
            "language_model.lm_head",
        )

    def test_multimodal_source_naming_remap(self):
        # Source on-disk uses `model.language_model.X`; vLLM internal
        # is `language_model.model.X` (the prefix swap).
        self.assertEqual(
            _to_vllm_internal_name(
                "model.language_model.layers.5.mlp.shared_expert_gate"),
            "language_model.model.layers.5.mlp.shared_expert_gate",
        )

    def test_visual_remap(self):
        self.assertEqual(
            _to_vllm_internal_name("model.visual.blocks.0.attn.proj"),
            "visual.blocks.0.attn.proj",
        )


class TestBuildQuantizationConfig(unittest.TestCase):
    def test_minimal_two_format_assignment(self):
        # Lots of NVFP4, fewer MXFP8 → NVFP4 becomes the catch-all
        # bucket (largest count) and gets the per-expert pattern.
        assignment = {
            f"model.layers.{i}.self_attn.k_proj": "MXFP8"
            for i in range(2)  # 2 MXFP8 entries
        }
        for i in range(5):  # 5 NVFP4 entries
            assignment[f"model.layers.{i}.mlp.experts.down_proj"] = "NVFP4"
        qc = build_quantization_config(
            assignment, bf16_passthrough={"lm_head"},
        )
        self.assertEqual(qc["quant_method"], "compressed-tensors")
        self.assertEqual(qc["format"], "mixed-precision")
        self.assertEqual(len(qc["config_groups"]), 2)
        # Find each group by num_bits — order isn't part of the contract
        groups_by_bits = {
            g["weights"]["num_bits"]: g
            for g in qc["config_groups"].values()
        }
        mxfp8 = groups_by_bits[8]
        nvfp4 = groups_by_bits[4]
        # MXFP8 group: explicit per-name regex targets only
        self.assertTrue(all(t.startswith("re:^language_model[.]")
                            for t in mxfp8["targets"]))
        self.assertNotIn(PER_EXPERT_MOE_REGEX, mxfp8["targets"])
        # NVFP4 catch-all: explicit + the per-expert pattern
        self.assertEqual(nvfp4["weights"]["strategy"], "tensor_group")
        self.assertEqual(nvfp4["weights"]["group_size"], 16)
        self.assertIn(PER_EXPERT_MOE_REGEX, nvfp4["targets"])
        # NVFP4 group must declare its per-group format so vLLM's
        # is_activation_quantization_format check enables W4A4 dispatch.
        self.assertEqual(nvfp4["format"], "nvfp4-pack-quantized")

    def test_ignore_uses_vllm_internal_naming(self):
        assignment = {
            "model.layers.0.mlp.gate_proj": "NVFP4",
            "model.layers.0.mlp.shared_expert_gate": "BF16",
        }
        qc = build_quantization_config(
            assignment, bf16_passthrough={"lm_head"},
            extra_ignore=["model.layers.0.mlp.gate"],
        )
        ignore = qc["ignore"]
        self.assertIn("language_model.lm_head", ignore)
        self.assertIn(
            "language_model.model.layers.0.mlp.shared_expert_gate", ignore)
        self.assertIn(
            "language_model.model.layers.0.mlp.gate", ignore)

    def test_no_class_name_catchall_target(self):
        # The class-name catch-all "Linear" short-circuits vLLM's
        # fused-layer match path and was the bug that produced wrong
        # scheme allocation. Make sure we don't reintroduce it.
        assignment = {"model.layers.0.mlp.gate_proj": "NVFP4"}
        qc = build_quantization_config(assignment, bf16_passthrough=set())
        for group in qc["config_groups"].values():
            for t in group["targets"]:
                self.assertNotEqual(t, "Linear",
                                    "do not use a 'Linear' class-name catch-all; "
                                    "it short-circuits fused-layer match")


class TestQuantize2DDispatch(unittest.TestCase):
    def test_nvfp4_emits_input_global_scale(self):
        """vLLM's CompressedTensorsW4A4Nvfp4 process_weights_after_loading
        does `1 / input_global_scale.max()`. Without an emitted value,
        the param defaults to zeros and vLLM produces 1/0 = inf →
        degenerate output. Make sure we always emit it."""
        W = torch.randn(8, 16) * 0.1
        out = _quantize_2d(W, "NVFP4")
        self.assertIn("weight_packed", out)
        self.assertIn("weight_scale", out)
        self.assertIn("weight_global_scale", out)
        self.assertIn("input_global_scale", out)
        self.assertEqual(out["input_global_scale"].dtype, torch.float32)
        self.assertEqual(out["input_global_scale"].numel(), 1)
        self.assertAlmostEqual(
            out["input_global_scale"].item(), DEFAULT_INPUT_GLOBAL_SCALE)

    def test_mxfp8_emits_fp8_dense(self):
        # Routed through CompressedTensorsW8A8Fp8 — wants a `weight`
        # tensor in fp8_e4m3fn + per-channel `weight_scale`.
        W = torch.randn(8, 16) * 0.1
        out = _quantize_2d(W, "MXFP8")
        self.assertIn("weight", out)
        self.assertEqual(out["weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(out["weight_scale"].dtype, torch.float32)
        self.assertEqual(tuple(out["weight_scale"].shape), (8, 1))


class TestFusedSiblingJointGlobalScale(unittest.TestCase):
    """vLLM warns when q/k/v/gate/up have different weight_global_scale.
    The exporter pre-computes a joint per-tensor scale across each
    fused-sibling group so the warning goes away (and the per-tensor
    scale on disk is correct under vLLM's fused-loader rules)."""

    def test_fused_dense_group_self_attn(self):
        from prismquant.export_native_compressed import _fused_dense_group
        g = _fused_dense_group("model.layers.5.self_attn.q_proj")
        self.assertIsNotNone(g)
        pre, members = g
        self.assertEqual(pre, "model.layers.5")
        self.assertIn("k_proj", members)

    def test_fused_dense_group_mlp_gate_up(self):
        from prismquant.export_native_compressed import _fused_dense_group
        g = _fused_dense_group("model.layers.0.mlp.shared_expert.up_proj")
        self.assertIsNotNone(g)
        self.assertEqual(set(g[1]), {"gate_proj", "up_proj"})

    def test_fused_dense_group_qwen36_linear_attn(self):
        from prismquant.export_native_compressed import _fused_dense_group
        for sib in ("in_proj_qkv", "in_proj_z"):
            g = _fused_dense_group(f"model.layers.7.linear_attn.{sib}")
            self.assertIsNotNone(g, f"missing fused-group pattern for {sib}")
            self.assertEqual(set(g[1]), {"in_proj_qkv", "in_proj_z"})

    def test_compute_nvfp4_joint_global_picks_max(self):
        from prismquant.export_native_compressed import (
            _compute_nvfp4_joint_global, compute_nvfp4_global_real,
        )

        # Build a tiny model with two fused-sibling Linears (different
        # max-abs values). The joint scale must be the max of their
        # natural per-tensor scales.
        class TinyAttn(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.q_proj = torch.nn.Linear(32, 32, bias=False)
                s.k_proj = torch.nn.Linear(32, 32, bias=False)
                s.v_proj = torch.nn.Linear(32, 32, bias=False)

        class TinyLayer(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.self_attn = TinyAttn()

        class TinyModel(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.model = torch.nn.Module()
                s.model.layers = torch.nn.ModuleList([TinyLayer()])

        torch.manual_seed(0)
        m = TinyModel()
        # Force k_proj to have the largest max-abs.
        with torch.no_grad():
            m.model.layers[0].self_attn.k_proj.weight.mul_(10.0)

        assignment = {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.layers.0.self_attn.k_proj": "NVFP4",
            "model.layers.0.self_attn.v_proj": "NVFP4",
        }
        joint = _compute_nvfp4_joint_global(m, assignment)
        self.assertEqual(len(joint), 3)
        joint_value = next(iter(joint.values())).item()
        # All three must point to the SAME scalar.
        for v in joint.values():
            self.assertAlmostEqual(v.item(), joint_value)
        # And it must be at least the natural scale of the max sibling.
        natural = compute_nvfp4_global_real(
            m.model.layers[0].self_attn.k_proj.weight.float()).item()
        self.assertAlmostEqual(joint_value, natural, places=5)


class TestPackedExpertSplit(unittest.TestCase):
    def test_quantize_3d_packed_nvfp4_returns_per_expert_dim(self):
        # 3D packed `[E, M, N]` produces tensors with leading expert
        # dim preserved. Splitting into per-expert-per-projection is
        # done in materialize_tensors, not _quantize_3d_packed.
        E, M, N = 4, 32, 64
        P = torch.randn(E, M, N) * 0.05
        out = _quantize_3d_packed(P, "NVFP4")
        self.assertEqual(out["weight_packed"].shape[0], E)
        self.assertEqual(out["weight_global_scale"].shape, torch.Size([E]))


if __name__ == "__main__":
    unittest.main()
