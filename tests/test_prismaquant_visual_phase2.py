"""Tests for visual-encoder staging, cost shards and export. Covers:

  1. sensitivity_probe.stage_multimodal — preserves vision_config (vs
     stage_text_only which strips it).
  2. allocator.apply_visual_format_override — the uniform visual format.
  3. export_native_compressed._apply_visual_recipe_quant passes the
     activations cached under a visual Linear's recipe name to
     native_render_codec.quantize_2d.
"""
from __future__ import annotations

import pytest

import prismaquant.export_native_compressed as _test_exporter
from prismaquant.native_render_levers import NativeRenderLevers

@pytest.fixture(autouse=True)
def _isolate_export_inputs():
    saved = (_test_exporter._RENDER_LEVERS, _test_exporter._CACHED_ACTIVATIONS,
             _test_exporter._INPUT_GLOBAL_SCALES)
    _test_exporter._RENDER_LEVERS = None
    _test_exporter._CACHED_ACTIVATIONS = None
    _test_exporter._INPUT_GLOBAL_SCALES = None
    yield
    (_test_exporter._RENDER_LEVERS, _test_exporter._CACHED_ACTIVATIONS,
     _test_exporter._INPUT_GLOBAL_SCALES) = saved


import prismaquant.native_render_codec as _native_render_codec

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from prismaquant.sensitivity_probe import (
    stage_multimodal,
    stage_text_only,
)


class TestStageMultimodalPreservesVisionConfig(unittest.TestCase):
    """stage_multimodal must keep vision_config / audio_config intact,
    vs stage_text_only which strips them."""

    def _write_multimodal_config(self, td: Path) -> None:
        cfg = {
            "architectures": ["Qwen36ForConditionalGeneration"],
            "model_type": "qwen36_vl",
            "vision_config": {
                "depth": 27,
                "hidden_size": 1280,
                "num_heads": 16,
            },
            "text_config": {
                "num_hidden_layers": 48,
                "hidden_size": 4096,
                "model_type": "qwen36_text",
            },
            "image_token_id": 151655,
        }
        with open(td / "config.json", "w") as f:
            json.dump(cfg, f)
        # Need at least one other file so symlink copy path runs.
        (td / "tokenizer_config.json").write_text("{}")

    def test_vision_config_preserved(self):
        with tempfile.TemporaryDirectory() as src_dir:
            src = Path(src_dir)
            self._write_multimodal_config(src)
            staged = stage_multimodal(str(src))
            self.assertNotEqual(staged, str(src))
            with open(Path(staged) / "config.json") as f:
                cfg = json.load(f)
            self.assertIn("vision_config", cfg)
            self.assertEqual(cfg["vision_config"]["depth"], 27)
            self.assertIn("text_config", cfg)
            self.assertIn("image_token_id", cfg)
            # Architectures aren't rewritten for the multimodal path.
            self.assertEqual(cfg["architectures"],
                             ["Qwen36ForConditionalGeneration"])

    def test_other_files_symlinked(self):
        with tempfile.TemporaryDirectory() as src_dir:
            src = Path(src_dir)
            self._write_multimodal_config(src)
            staged = Path(stage_multimodal(str(src)))
            # tokenizer_config should be symlinked, not copied.
            link = staged / "tokenizer_config.json"
            self.assertTrue(link.exists())
            self.assertTrue(link.is_symlink())

    def test_pure_text_checkpoint_is_passthrough(self):
        # stage_multimodal on a checkpoint without any multimodal keys
        # should return the source path unchanged — no symlink tree
        # needed.
        with tempfile.TemporaryDirectory() as src_dir:
            src = Path(src_dir)
            with open(src / "config.json", "w") as f:
                json.dump({"model_type": "qwen2", "hidden_size": 4096}, f)
            staged = stage_multimodal(str(src))
            self.assertEqual(staged, str(src))

    def test_stage_text_only_vs_multimodal_diverge_on_vision(self):
        # stage_text_only strips vision_config; stage_multimodal keeps it.
        with tempfile.TemporaryDirectory() as src_dir:
            src = Path(src_dir)
            self._write_multimodal_config(src)
            mm = stage_multimodal(str(src))
            txt = stage_text_only(str(src))
            with open(Path(mm) / "config.json") as f:
                cfg_mm = json.load(f)
            with open(Path(txt) / "config.json") as f:
                cfg_txt = json.load(f)
            self.assertIn("vision_config", cfg_mm)
            self.assertNotIn("vision_config", cfg_txt)

    def test_staging_uses_configured_temp_parent(self):
        with (
            tempfile.TemporaryDirectory() as src_dir,
            tempfile.TemporaryDirectory() as tmp_dir,
        ):
            src = Path(src_dir)
            temp_parent = Path(tmp_dir).resolve()
            self._write_multimodal_config(src)
            with mock.patch.dict(
                "os.environ",
                {"PRISMAQUANT_TMPDIR": str(temp_parent), "TMPDIR": ""},
            ):
                mm = Path(stage_multimodal(str(src))).resolve()
                txt = Path(stage_text_only(str(src))).resolve()

            self.assertTrue(mm.is_relative_to(temp_parent))
            self.assertTrue(txt.is_relative_to(temp_parent))


class TestAllocatorVisualSensitivityModes(unittest.TestCase):
    """Exercise the uniform visual assignment helper."""

    def test_uniform_mode_always_stamps_visual_format(self):
        # Visual Linears are stamped with --visual-format regardless of
        # whether the probe had them.
        from prismaquant.allocator import (
            apply_visual_format_override,
        )
        assignment = {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.visual.blocks.0.attn.qkv": "NVFP4",
        }
        out = apply_visual_format_override(assignment, "BF16")
        self.assertEqual(out["model.visual.blocks.0.attn.qkv"], "BF16")
        self.assertEqual(out["model.layers.0.self_attn.q_proj"], "NVFP4")

class TestQuantize2DVisualActivationsRoundtrip(unittest.TestCase):
    """The exporter hands a visual Linear's cached activations to the codec.

    ``_apply_visual_recipe_quant`` is the exporter's visual boundary: it
    looks up ``_CACHED_ACTIVATIONS`` by the recipe name (the weight key without
    ``.weight``) and passes that tensor to ``native_render_codec.quantize_2d``,
    which runs GPTQ / AR rounding on it. A spy on the codec records the tensor
    the exporter passed, and a decoy under the weight key catches a lookup by
    the wrong name."""

    def _export_visual(self, name, weight, cached):
        """Run the exporter's visual quant; return its output and the codec calls."""
        real = _native_render_codec.quantize_2d
        seen = []

        def spy(*args, **kwargs):
            seen.append((kwargs.get("linear_name"), kwargs.get("cached_activations")))
            return real(*args, **kwargs)

        with mock.patch.object(_test_exporter, "_CACHED_ACTIVATIONS", cached), \
                mock.patch.object(_native_render_codec, "quantize_2d", spy):
            out = _test_exporter._apply_visual_recipe_quant(
                {f"{name}.weight": weight}, {name: "NVFP4"},
                device=torch.device("cpu"))
        return out, seen

    def test_gptq_enabled_reads_visual_activations_from_cache(self):
        visual_name = "model.visual.blocks.0.attn.qkv"
        # Rank-2 weight with in_features divisible by NVFP4 group_size=16.
        W = torch.randn(48, 32, dtype=torch.bfloat16)
        fake_acts = torch.randn(64, 32, dtype=torch.float32)
        decoy = torch.randn(64, 32, dtype=torch.float32)
        out, seen = self._export_visual(
            visual_name, W, {visual_name: fake_acts, f"{visual_name}.weight": decoy})
        self.assertEqual([name for name, _ in seen], [visual_name])
        self.assertIs(seen[0][1], fake_acts)
        for key in ("weight_packed", "weight_scale",
                    "weight_global_scale", "input_global_scale"):
            self.assertIn(f"{visual_name}.{key}", out)

    def test_gptq_fallback_when_activation_cache_missing_key(self):
        # If _CACHED_ACTIVATIONS doesn't have the visual key, the codec gets no
        # activations (the activation-aware pass is skipped) and a valid NVFP4
        # triple is still produced.
        visual_name = "model.visual.merger.mlp.0"
        W = torch.randn(32, 32, dtype=torch.bfloat16)
        out, seen = self._export_visual(
            visual_name, W, {"some.other.linear": torch.randn(4, 32)})
        self.assertEqual(seen, [(visual_name, None)])
        self.assertIn(f"{visual_name}.weight_packed", out)

    def test_quantize_2d_multimodal_activation_reuse_body_path(self):
        # The codec's packed output for a visual Linear has the shape a body
        # Linear of the same weight shape gets: the Linear's name selects only
        # its activations, not its layout.
        W = torch.randn(32, 32, dtype=torch.bfloat16)
        acts = torch.randn(64, 32, dtype=torch.float32)
        levers = NativeRenderLevers.resolve()
        visual = _native_render_codec.quantize_2d(
            W, "NVFP4", linear_name="model.visual.blocks.0.mlp.fc1",
            gptq_enabled=True, cached_activations=acts, levers=levers)
        body = _native_render_codec.quantize_2d(
            W, "NVFP4", linear_name="model.layers.0.mlp.gate_proj",
            gptq_enabled=True, cached_activations=acts, levers=levers)
        self.assertEqual(visual["weight_packed"].shape,
                         body["weight_packed"].shape)
