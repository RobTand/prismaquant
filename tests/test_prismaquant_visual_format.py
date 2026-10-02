"""Tests for the Phase 1 visual-encoder --visual-format override.

The override lives in three places the body never touches:
  1. allocator.apply_visual_format_override — stamps visual Linears
     in an assignment dict with a uniform target format.
  2. allocator.discover_visual_linears_from_source — scans a source
     checkpoint's safetensors for rank-2 `model.visual.*.weight`
     entries (the enumerable visual Linears).
  3. export_native_compressed._apply_visual_recipe_quant — rewrites
     passthrough-loaded visual `.weight` tensors through
     `_quantize_2d` when the recipe assigns non-BF16.

Complete measured multimodal Fisher rows use per-Linear allocator decisions.
These tests cover the explicit uniform/text-only control and source census;
the exporter dispatches on the resulting recipe's format assignment.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from prismaquant.allocator import (
    apply_visual_format_override,
    discover_visual_linear_stats_from_source,
    discover_visual_linears_from_source,
    validate_source_visual_passthrough_contract,
    _is_visual_linear,
)
from prismaquant.export_native_compressed import (
    _apply_visual_recipe_quant,
)


class TestIsVisualLinear(unittest.TestCase):
    def test_matches_model_visual_prefix(self):
        self.assertTrue(_is_visual_linear("model.visual.blocks.0.attn.qkv"))
        self.assertTrue(_is_visual_linear("model.visual.merger.mlp.0"))

    def test_matches_post_remap_visual_prefix(self):
        # Some profiles strip `model.` so recipe names land as
        # `visual.blocks.X.*`. Both forms must be recognized.
        self.assertTrue(_is_visual_linear("visual.blocks.5.attn.proj"))

    def test_rejects_body_and_mtp(self):
        self.assertFalse(_is_visual_linear("model.layers.0.self_attn.q_proj"))
        self.assertFalse(_is_visual_linear("mtp.layers.0.mlp.gate_proj"))
        self.assertFalse(_is_visual_linear("lm_head"))


class TestApplyVisualFormatOverride(unittest.TestCase):
    def test_override_forces_uniform_nvfp4_for_visual(self):
        # Mixed assignment: body stays untouched; visual Linears are
        # forced uniformly to the override format even when their prior
        # assignment varied.
        assignment = {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.layers.0.mlp.gate_proj": "MXFP8",
            "model.visual.blocks.0.attn.qkv": "BF16",
            "model.visual.blocks.1.mlp.fc1": "MXFP8",
            "visual.merger.mlp.0": "BF16",
        }
        out = apply_visual_format_override(assignment, "NVFP4")
        # Body untouched.
        self.assertEqual(out["model.layers.0.self_attn.q_proj"], "NVFP4")
        self.assertEqual(out["model.layers.0.mlp.gate_proj"], "MXFP8")
        # All visual Linears are uniformly NVFP4 regardless of prior.
        self.assertEqual(out["model.visual.blocks.0.attn.qkv"], "NVFP4")
        self.assertEqual(out["model.visual.blocks.1.mlp.fc1"], "NVFP4")
        self.assertEqual(out["visual.merger.mlp.0"], "NVFP4")

    def test_bf16_override_keeps_visual_entries_but_stamps_bf16(self):
        assignment = {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.visual.blocks.0.attn.qkv": "MXFP8",
        }
        out = apply_visual_format_override(assignment, "BF16")
        self.assertEqual(out["model.layers.0.self_attn.q_proj"], "NVFP4")
        self.assertEqual(out["model.visual.blocks.0.attn.qkv"], "BF16")

    def test_override_does_not_mutate_input(self):
        assignment = {"model.visual.blocks.0.attn.qkv": "BF16"}
        _ = apply_visual_format_override(assignment, "NVFP4")
        self.assertEqual(assignment["model.visual.blocks.0.attn.qkv"], "BF16")


class TestDiscoverVisualLinearsFromSource(unittest.TestCase):
    """Scan a synthetic single-shard safetensors checkpoint for visual Linears."""

    def _write_synthetic_checkpoint(self, tmpdir: Path) -> None:
        # Mix of body, visual Linears, non-2D visual params, and MTP —
        # only the rank-2 visual `.weight` entries should be returned.
        tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(16, 16),
            "model.visual.blocks.0.attn.qkv.weight": torch.zeros(48, 16),
            "model.visual.blocks.0.attn.qkv.bias": torch.zeros(48),
            "model.visual.blocks.0.norm1.weight": torch.zeros(16),    # norm, rank-1
            "model.visual.blocks.1.mlp.fc1.weight": torch.zeros(64, 16),
            "model.visual.patch_embed.conv.weight": torch.zeros(16, 3, 14, 14),  # conv, rank-4
            "mtp.layers.0.mlp.gate_up_proj.weight": torch.zeros(32, 16),
        }
        save_file(tensors, str(tmpdir / "model.safetensors"))

    def test_discovers_only_rank2_visual_weights(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._write_synthetic_checkpoint(tmp)
            names = discover_visual_linears_from_source(str(tmp))
        # Two rank-2 visual .weight entries; the norm (rank-1), conv
        # (rank-4), bias, and non-visual keys are excluded.
        self.assertEqual(set(names), {
            "model.visual.blocks.0.attn.qkv",
            "model.visual.blocks.1.mlp.fc1",
        })

    def test_discovery_retains_exact_shape_and_source_dtype(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            name = "model.visual.blocks.0.attn.qkv"
            save_file(
                {f"{name}.weight": torch.zeros(
                    48, 16, dtype=torch.bfloat16
                )},
                str(tmp / "model.safetensors"),
            )
            stats = discover_visual_linear_stats_from_source(str(tmp))
        self.assertEqual(stats[name], {
            "n_params": 48 * 16,
            "out_features": 48,
            "in_features": 16,
            "source_dtype": "bf16",
            "has_fp8_scale": False,
            "_nvfp4_weight_only": True,
        })
        validate_source_visual_passthrough_contract(stats, "BF16")

    def test_profile_mapping_marks_all_text_excluded_stock_targets_weight_only(self):
        from prismaquant.allocator import _mark_weight_only_nvfp4_stats

        class _Profile:
            @staticmethod
            def checkpoint_to_live_name(name, *, multimodal=False):
                if name.startswith(("model.audio_tower.", "mtp.")):
                    return None
                return name

        stats = {
            "model.layers.0.mlp.down_proj": {"n_params": 1},
            "model.audio_tower.proj": {"n_params": 1},
            "mtp.layers.0.mlp.down_proj": {"n_params": 1},
        }
        marked = _mark_weight_only_nvfp4_stats(stats, _Profile())
        self.assertNotIn(
            "_nvfp4_weight_only",
            marked["model.layers.0.mlp.down_proj"],
        )
        self.assertTrue(
            marked["model.audio_tower.proj"]["_nvfp4_weight_only"]
        )
        self.assertTrue(
            marked["mtp.layers.0.mlp.down_proj"]["_nvfp4_weight_only"]
        )

    def test_non_bf16_source_cannot_receive_bf16_passthrough_budget(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            name = "model.visual.blocks.0.attn.qkv"
            save_file(
                {f"{name}.weight": torch.zeros(
                    48, 16, dtype=torch.float32
                )},
                str(tmp / "model.safetensors"),
            )
            stats = discover_visual_linear_stats_from_source(str(tmp))
        with self.assertRaisesRegex(
            ValueError, "passthrough contract.*incompatible source dtype/scale"
        ):
            validate_source_visual_passthrough_contract(stats, "BF16")

    def test_fp8_source_passthrough_requires_scale_sibling(self):
        stats = {
            "model.visual.blocks.0.attn.qkv": {
                "source_dtype": "fp8",
                "has_fp8_scale": False,
            },
        }
        with self.assertRaisesRegex(
            ValueError, "passthrough contract.*incompatible source dtype/scale"
        ):
            validate_source_visual_passthrough_contract(stats, "FP8_SOURCE")

    def test_discover_returns_empty_when_no_visual(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            save_file({"model.layers.0.mlp.gate_proj.weight": torch.zeros(8, 8)},
                      str(tmp / "model.safetensors"))
            self.assertEqual(discover_visual_linears_from_source(str(tmp)), [])

    def test_discover_with_safetensors_index(self):
        # The sharded path uses the index.json to avoid opening every
        # shard blindly. Write two shards + an index pointing into them.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            save_file({"model.visual.blocks.0.attn.qkv.weight": torch.zeros(48, 16)},
                      str(tmp / "model-00001-of-00002.safetensors"))
            save_file({"model.layers.0.self_attn.q_proj.weight": torch.zeros(16, 16)},
                      str(tmp / "model-00002-of-00002.safetensors"))
            index = {
                "metadata": {"total_size": 1},
                "weight_map": {
                    "model.visual.blocks.0.attn.qkv.weight":
                        "model-00001-of-00002.safetensors",
                    "model.layers.0.self_attn.q_proj.weight":
                        "model-00002-of-00002.safetensors",
                },
            }
            with open(tmp / "model.safetensors.index.json", "w") as f:
                json.dump(index, f)
            names = discover_visual_linears_from_source(str(tmp))
            self.assertEqual(names, ["model.visual.blocks.0.attn.qkv"])


class TestApplyVisualRecipeQuant(unittest.TestCase):
    """_apply_visual_recipe_quant rewrites visual `.weight` entries
    through _quantize_2d when the assignment says non-BF16."""

    def test_nvfp4_assignment_replaces_weight_with_compressed_tensors(self):
        # Group size 16 → shape must be divisible by 16 on the in dim.
        name = "model.visual.blocks.0.attn.qkv"
        src_extra = {
            f"{name}.weight": torch.randn(48, 32, dtype=torch.bfloat16),
            f"{name}.bias": torch.randn(48, dtype=torch.bfloat16),
            # Non-Linear visual tensor (norm) — passthrough.
            "model.visual.blocks.0.norm1.weight": torch.randn(32, dtype=torch.bfloat16),
        }
        assignment = {name: "NVFP4"}
        out = _apply_visual_recipe_quant(src_extra, assignment,
                                         device=torch.device("cpu"))
        # Quantized weight emits the compressed-tensors triple.
        self.assertIn(f"{name}.weight_packed", out)
        self.assertIn(f"{name}.weight_scale", out)
        self.assertIn(f"{name}.weight_global_scale", out)
        self.assertIn(f"{name}.input_global_scale", out)
        # The raw `.weight` key must be gone (replaced by weight_packed).
        self.assertNotIn(f"{name}.weight", out)
        # Bias and non-Linear norm pass through unchanged.
        self.assertIn(f"{name}.bias", out)
        self.assertIs(out[f"{name}.bias"], src_extra[f"{name}.bias"])
        self.assertIn("model.visual.blocks.0.norm1.weight", out)

    def test_mxfp8_assignment_replaces_weight_with_fp8_triple(self):
        # Group size 32 → in dim must be divisible by 32.
        name = "model.visual.blocks.0.mlp.fc1"
        src_extra = {
            f"{name}.weight": torch.randn(64, 64, dtype=torch.bfloat16),
        }
        out = _apply_visual_recipe_quant(src_extra, {name: "MXFP8"},
                                         device=torch.device("cpu"))
        # MXFP8 emits a `.weight` (fp8_e4m3fn) + `.weight_scale` pair.
        self.assertIn(f"{name}.weight", out)
        self.assertIn(f"{name}.weight_scale", out)
        self.assertEqual(out[f"{name}.weight"].dtype, torch.float8_e4m3fn)

    def test_bf16_assignment_is_passthrough(self):
        name = "model.visual.blocks.0.attn.qkv"
        tensor = torch.randn(48, 32, dtype=torch.bfloat16)
        src_extra = {f"{name}.weight": tensor}
        out = _apply_visual_recipe_quant(src_extra, {name: "BF16"},
                                         device=torch.device("cpu"))
        # BF16 means "leave the passthrough tensor alone".
        self.assertIn(f"{name}.weight", out)
        self.assertIs(out[f"{name}.weight"], tensor)
        self.assertNotIn(f"{name}.weight_packed", out)

    def test_missing_assignment_is_passthrough(self):
        # A visual .weight that the recipe doesn't mention (Phase 2
        # allocator could skip some, or user-supplied partial recipe)
        # must pass through unchanged.
        name = "model.visual.patch_embed.proj"
        tensor = torch.randn(16, 16, dtype=torch.bfloat16)
        src_extra = {f"{name}.weight": tensor}
        out = _apply_visual_recipe_quant(src_extra, {},
                                         device=torch.device("cpu"))
        self.assertIs(out[f"{name}.weight"], tensor)

    def test_non_visual_keys_are_passthrough(self):
        # Non-visual passthrough entries (e.g. MTP head tensors that
        # didn't get materialized) must not be touched even if they
        # appear in the recipe.
        src_extra = {
            "mtp.fc.weight": torch.randn(16, 16, dtype=torch.bfloat16),
        }
        assignment = {"mtp.fc": "NVFP4"}
        out = _apply_visual_recipe_quant(src_extra, assignment,
                                         device=torch.device("cpu"))
        self.assertIn("mtp.fc.weight", out)
        self.assertIs(out["mtp.fc.weight"], src_extra["mtp.fc.weight"])

    def test_quant_failure_raises_instead_of_bf16_passthrough(self):
        name = "model.visual.blocks.0.attn.qkv"
        src_extra = {f"{name}.weight": torch.randn(48, 32, dtype=torch.bfloat16)}
        with patch(
            "prismaquant.export_native_compressed._quantize_2d",
            side_effect=ValueError("shape mismatch"),
        ):
            with self.assertRaisesRegex(RuntimeError, "visual quant failed"):
                _apply_visual_recipe_quant(
                    src_extra,
                    {name: "NVFP4"},
                    device=torch.device("cpu"),
                )


class TestAllocatorVisualEndToEnd(unittest.TestCase):
    """End-to-end: discover visual Linears from a synthetic source +
    run apply_visual_format_override → the resulting layer_config
    assigns NVFP4 to every visual Linear."""

    def test_discover_then_override_yields_uniform_nvfp4(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            tensors = {
                "model.layers.0.self_attn.q_proj.weight": torch.zeros(16, 16),
                "model.visual.blocks.0.attn.qkv.weight": torch.zeros(48, 16),
                "model.visual.blocks.0.mlp.fc1.weight": torch.zeros(32, 16),
                "model.visual.blocks.0.mlp.fc2.weight": torch.zeros(16, 32),
                "model.visual.blocks.0.norm1.weight": torch.zeros(16),  # rank-1
            }
            save_file(tensors, str(tmp / "model.safetensors"))
            names = discover_visual_linears_from_source(str(tmp))
            # Start with an empty assignment then inject visual entries
            # (this mirrors what main() does after the knapsack DP).
            assignment = {"model.layers.0.self_attn.q_proj": "NVFP4"}
            for n in names:
                assignment[n] = "BF16"  # seed with pre-override value
            out = apply_visual_format_override(assignment, "NVFP4")
            # All three rank-2 visual Linears flipped to NVFP4.
            self.assertEqual(
                {n: out[n] for n in names},
                {n: "NVFP4" for n in names},
            )
            self.assertEqual(len(names), 3)
            self.assertEqual(out["model.layers.0.self_attn.q_proj"], "NVFP4")


if __name__ == "__main__":
    unittest.main()
