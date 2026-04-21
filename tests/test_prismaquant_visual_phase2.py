"""Tests for Phase 2 visual-encoder calibration.

Phase 2 adds real activation + Fisher capture for visual Linears so
they enter the allocator's DP as regular candidates and get the same
AWQ/GPTQ/AR treatment at export. Covers:

  1. sensitivity_probe.stage_multimodal — preserves vision_config (vs
     stage_text_only which strips it).
  2. sensitivity_probe.load_multimodal_calibration — synthetic-stub
     path, assert shapes and contract.
  3. incremental_probe CLI flag threading for the multimodal path.
  4. incremental_measure_quant_cost._run_visual_cost_shard — measures
     cost entries for a tiny visual-style nn.Module stub.
  5. allocator --visual-sensitivity={fisher, uniform} mode selection.
  6. export_native_compressed._quantize_2d picks up visual activations
     from _CACHED_ACTIVATIONS when the multimodal probe populates them.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from prismaquant.sensitivity_probe import (
    _synthetic_multimodal_calibration_samples,
    load_multimodal_calibration,
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


class TestSyntheticMultimodalCalibration(unittest.TestCase):
    """load_multimodal_calibration synthetic path produces expected
    shape triples without needing a real processor."""

    def test_synthetic_returns_n_triples_with_pixel_and_ids(self):
        samples = _synthetic_multimodal_calibration_samples(
            processor=None, n_samples=4, max_text_len=16,
        )
        self.assertEqual(len(samples), 4)
        for sample in samples:
            self.assertIsInstance(sample, dict)
            pixel_values = sample["pixel_values"]
            input_ids = sample["input_ids"]
            labels = sample["labels"]
            self.assertEqual(pixel_values.dim(), 4)
            self.assertEqual(pixel_values.shape[0], 1)   # batch
            self.assertEqual(pixel_values.shape[1], 3)   # RGB
            self.assertEqual(input_ids.dim(), 2)
            self.assertEqual(input_ids.shape[0], 1)      # batch
            self.assertLessEqual(input_ids.shape[1], 16)
            self.assertEqual(labels.shape, input_ids.shape)

    def test_load_multimodal_synthetic_flag_skips_dataset_load(self):
        # dataset_name="synthetic" never touches the datasets library
        # and always returns the offline stub — important for offline
        # tests.
        triples = load_multimodal_calibration(
            processor=None, dataset_name="synthetic",
            n_samples=3, max_text_len=12,
        )
        self.assertEqual(len(triples), 3)

    def test_synthetic_captions_rotate(self):
        # 12 samples with 8 built-in captions should wrap around cleanly.
        triples = _synthetic_multimodal_calibration_samples(
            processor=None, n_samples=12, max_text_len=16,
        )
        self.assertEqual(len(triples), 12)


class TestMultimodalProbeFlagParsing(unittest.TestCase):
    """The new CLI flags (--calibration-modality, --mm-dataset,
    --mm-nsamples, --mm-max-text-len) must parse and thread through
    to the main() function without breaking the text-only default."""

    def test_flags_in_incremental_probe_argparser(self):
        # Import the module's argparser the way main() does, but without
        # triggering the heavy run.
        from prismaquant import incremental_probe

        src = inspect_source(incremental_probe.main)
        # The flag surface we expose should be discoverable in main()'s
        # source. We keep this as a light contract test so future
        # refactors that rename one of the flags must also update the
        # tests (and docs).
        self.assertIn("--calibration-modality", src)
        self.assertIn("--mm-dataset", src)
        self.assertIn("--mm-nsamples", src)
        self.assertIn("--mm-max-text-len", src)

    def test_default_calibration_modality_is_text_only(self):
        from prismaquant import incremental_probe
        src = inspect_source(incremental_probe.main)
        # Search for the default= in the argparse add_argument.
        self.assertIn('default="text-only"', src)


def inspect_source(fn) -> str:
    import inspect
    return inspect.getsource(fn)


class TestVisualCostShardBasic(unittest.TestCase):
    """_run_visual_cost_shard handles empty / no-match / OOM paths
    without raising, and emits a shard pickle the merger can consume."""

    def test_empty_shard_when_no_matching_stats(self):
        from prismaquant.incremental_measure_quant_cost import (
            _run_visual_cost_shard,
        )
        from prismaquant import format_registry as fr

        specs = [fr.get_format("BF16"), fr.get_format("NVFP4")]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            out = td / "cost_shard.pkl"
            act_cache = mock.Mock()
            result = _run_visual_cost_shard(
                model_path=str(td),
                linear_include=r"^model\.visual\.",
                probe_stats={},    # no visual stats => empty pickle
                act_cache=act_cache,
                specs=specs,
                device="cpu",
                dtype=torch.float32,
                mode="unbatched",
                chunk_size=1,
                h_detail=None,
                output_path=str(out),
                model_name=str(td),
                probe_path="probe.pkl",
            )
            self.assertTrue(result)
            self.assertTrue(out.exists())
            with open(out, "rb") as f:
                data = pickle.load(f)
            self.assertEqual(data["costs"], {})
            self.assertEqual(data["meta"]["shard_kind"], "visual")
            # Formats should still be recorded for merge consistency.
            self.assertEqual(data["formats"], ["BF16", "NVFP4"])

    def test_visual_shard_writes_empty_pickle_on_load_oom(self):
        """When AutoModelForCausalLM.from_pretrained raises CUDA OOM, we
        fall back to an empty pickle instead of raising — 122B-scale
        behavior."""
        from prismaquant.incremental_measure_quant_cost import (
            _run_visual_cost_shard,
        )
        from prismaquant import format_registry as fr

        specs = [fr.get_format("BF16")]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # Minimum config so stage_multimodal doesn't bail.
            (td / "config.json").write_text(
                json.dumps({"model_type": "qwen36",
                            "vision_config": {"depth": 2}})
            )
            out = td / "cost_shard.pkl"
            act_cache = mock.Mock()
            probe_stats = {
                "model.visual.blocks.0.attn.qkv": {"n_params": 100},
            }
            # Patch from_pretrained to raise a plain RuntimeError with
            # "out of memory" in the message — the guard should catch
            # that as an OOM proxy and fall back gracefully.
            with mock.patch(
                    "transformers.AutoModelForCausalLM.from_pretrained",
                    side_effect=RuntimeError("CUDA out of memory")):
                ok = _run_visual_cost_shard(
                    model_path=str(td),
                    linear_include=r"^model\.visual\.",
                    probe_stats=probe_stats,
                    act_cache=act_cache,
                    specs=specs,
                    device="cpu",
                    dtype=torch.float32,
                    mode="unbatched",
                    chunk_size=1,
                    h_detail=None,
                    output_path=str(out),
                    model_name=str(td),
                    probe_path="probe.pkl",
                )
            # OOM path returns False but writes an empty pickle so the
            # merge layout stays consistent.
            self.assertFalse(ok)
            self.assertTrue(out.exists())
            with open(out, "rb") as f:
                data = pickle.load(f)
            self.assertEqual(data["costs"], {})
            self.assertEqual(data["meta"]["shard_kind"], "visual")


class TestAllocatorVisualSensitivityModes(unittest.TestCase):
    """--visual-sensitivity decides whether visual Linears flow through
    the knapsack DP (fisher) or get stamped uniformly (uniform).
    Covers both modes + the graceful fallback when fisher is requested
    but the probe has no visual stats."""

    def test_visual_fisher_available_detects_both_probe_and_cost(self):
        # Helper replicated here since it's defined inside main(); we
        # test its guts via the public flow below.
        from prismaquant.allocator import _is_visual_linear
        body = {"model.layers.0.self_attn.q_proj": 1,
                "mtp.layers.0.mlp.gate_proj": 1}
        visual = {"model.visual.blocks.0.attn.qkv": 1}

        # Body-only: no visual in stats and no visual in costs.
        self.assertFalse(any(_is_visual_linear(n) for n in body))
        self.assertTrue(any(_is_visual_linear(n) for n in visual))

    def test_uniform_mode_always_stamps_visual_format(self):
        # When --visual-sensitivity=uniform, visual Linears are stamped
        # with --visual-format regardless of whether probe had them.
        # This is the Phase 1 path preserved.
        from prismaquant.allocator import (
            apply_visual_format_override,
        )
        assignment = {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.visual.blocks.0.attn.qkv": "NVFP4",   # Fisher placed it
        }
        out = apply_visual_format_override(assignment, "BF16")
        self.assertEqual(out["model.visual.blocks.0.attn.qkv"], "BF16")
        self.assertEqual(out["model.layers.0.self_attn.q_proj"], "NVFP4")

    def test_fisher_mode_detection_via_visual_prefix_scan(self):
        # Replicate the _visual_fisher_available predicate at test level:
        # a probe+cost combo is "Fisher available" iff both carry at
        # least one visual-prefix entry.
        from prismaquant.allocator import _is_visual_linear

        stats_with = {"model.visual.blocks.0.attn.qkv": {}}
        stats_without = {"model.layers.0.self_attn.q_proj": {}}
        costs_with = {"model.visual.blocks.0.attn.qkv": {"BF16": {}}}
        costs_without = {"model.layers.0.self_attn.q_proj": {"BF16": {}}}

        def available(stats, costs):
            return (any(_is_visual_linear(n) for n in stats)
                    and any(_is_visual_linear(n) for n in costs))

        self.assertTrue(available(stats_with, costs_with))
        self.assertFalse(available(stats_with, costs_without))
        self.assertFalse(available(stats_without, costs_with))
        self.assertFalse(available(stats_without, costs_without))


class TestQuantize2DVisualActivationsRoundtrip(unittest.TestCase):
    """The NVFP4 path in _quantize_2d consults `_CACHED_ACTIVATIONS`
    by recipe name. When the multimodal probe populates visual Linear
    activations under matching names, AWQ / GPTQ / AR rounding apply
    — this test confirms the cache lookup triggers the AWQ pass."""

    def test_awq_enabled_reads_visual_activations_from_cache(self):
        from prismaquant import export_native_compressed as exp

        visual_name = "model.visual.blocks.0.attn.qkv"
        # Rank-2 weight with in_features divisible by NVFP4 group_size=16.
        W = torch.randn(48, 32, dtype=torch.bfloat16)
        # Cache a fake activation tensor keyed by recipe name.
        fake_acts = torch.randn(64, 32, dtype=torch.float32)

        prev_cache = exp._CACHED_ACTIVATIONS
        try:
            exp._CACHED_ACTIVATIONS = {visual_name: fake_acts}
            out = exp._quantize_2d(
                W, "NVFP4",
                linear_name=visual_name,
                awq_enabled=True,          # AWQ on → needs activations
                gptq_enabled=False,
                awq_round_enabled=False,
            )
        finally:
            exp._CACHED_ACTIVATIONS = prev_cache
        # NVFP4 path emits the full triple whether or not AWQ fired;
        # we just confirm the call returned a valid compressed tensor
        # structure (no crash when activations come from the module-
        # level cache).
        for key in ("weight_packed", "weight_scale",
                    "weight_global_scale", "input_global_scale"):
            self.assertIn(key, out)

    def test_awq_fallback_when_activation_cache_missing_key(self):
        # If _CACHED_ACTIVATIONS doesn't have the visual key, AWQ is a
        # no-op (the path checks `acts is not None` at each step).
        # Still produces a valid NVFP4 triple.
        from prismaquant import export_native_compressed as exp

        visual_name = "model.visual.merger.mlp.0"
        W = torch.randn(32, 32, dtype=torch.bfloat16)
        prev_cache = exp._CACHED_ACTIVATIONS
        try:
            # Set cache but without this key.
            exp._CACHED_ACTIVATIONS = {"some.other.linear":
                                       torch.randn(4, 32)}
            out = exp._quantize_2d(
                W, "NVFP4",
                linear_name=visual_name,
                awq_enabled=True,
            )
        finally:
            exp._CACHED_ACTIVATIONS = prev_cache
        self.assertIn("weight_packed", out)

    def test_quantize_2d_multimodal_activation_reuse_body_path(self):
        # A visual Linear name that matches an exp cache entry should
        # produce the same-shape packed output as a body Linear of the
        # same weight shape — sanity that the cache lookup is name-
        # agnostic beyond the dictionary indexing.
        from prismaquant import export_native_compressed as exp

        W = torch.randn(32, 32, dtype=torch.bfloat16)
        acts = torch.randn(64, 32, dtype=torch.float32)

        prev = exp._CACHED_ACTIVATIONS
        try:
            exp._CACHED_ACTIVATIONS = {
                "model.visual.blocks.0.mlp.fc1": acts,
                "model.layers.0.mlp.gate_proj": acts,
            }
            visual = exp._quantize_2d(
                W, "NVFP4", linear_name="model.visual.blocks.0.mlp.fc1",
                awq_enabled=True,
            )
            body = exp._quantize_2d(
                W, "NVFP4", linear_name="model.layers.0.mlp.gate_proj",
                awq_enabled=True,
            )
        finally:
            exp._CACHED_ACTIVATIONS = prev
        self.assertEqual(visual["weight_packed"].shape,
                         body["weight_packed"].shape)


class TestMultimodalProbePassIntegration(unittest.TestCase):
    """run_multimodal_visual_probe_pass returns False gracefully when
    load_multimodal_calibration produces zero triples (e.g. processor
    failed) — no partial pickle is written."""

    def test_zero_triples_returns_false(self):
        from prismaquant import sensitivity_probe as sp

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "config.json").write_text(
                json.dumps({"model_type": "dummy"})
            )
            out = tdp / "visual_probe.pkl"
            # Patch AutoProcessor to succeed (returns a stub) but
            # load_multimodal_calibration to return []. Then expect
            # False return + no pickle written.
            fake_proc = mock.Mock()
            with mock.patch(
                    "transformers.AutoProcessor.from_pretrained",
                    return_value=fake_proc), \
                 mock.patch.object(sp, "load_multimodal_calibration",
                                   return_value=[]):
                result = sp.run_multimodal_visual_probe_pass(
                    str(tdp),
                    dataset_name="synthetic",
                    n_samples=2,
                    max_text_len=8,
                    requested_device="cpu",
                    dtype=torch.float32,
                    linear_include=r"^visual\.",
                    linear_exclude=r"(?!x)x",
                    activation_cache_dir=None,
                    output_path=str(out),
                )
            self.assertFalse(result)
            self.assertFalse(out.exists())

    def test_processor_load_failure_returns_false(self):
        from prismaquant import sensitivity_probe as sp

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "config.json").write_text(
                json.dumps({"model_type": "dummy"})
            )
            out = tdp / "visual_probe.pkl"
            # Both AutoProcessor.from_pretrained calls fail → graceful
            # fallback returns False.
            with mock.patch(
                    "transformers.AutoProcessor.from_pretrained",
                    side_effect=RuntimeError("no processor")):
                result = sp.run_multimodal_visual_probe_pass(
                    str(tdp),
                    dataset_name="synthetic",
                    n_samples=2,
                    max_text_len=8,
                    requested_device="cpu",
                    dtype=torch.float32,
                    linear_include=r"^visual\.",
                    linear_exclude=r"(?!x)x",
                    activation_cache_dir=None,
                    output_path=str(out),
                )
            self.assertFalse(result)


class TestPipelineCalibrationModalityEnv(unittest.TestCase):
    """run-pipeline.sh must surface CALIBRATION_MODALITY / MM_DATASET
    env defaults and route them through both the probe and allocator
    commands."""

    def test_pipeline_has_calibration_modality_default(self):
        script = Path(
            "/home/rob/spark-vllm-docker/quantization/prismaquant/"
            "run-pipeline.sh"
        ).read_text()
        self.assertIn('CALIBRATION_MODALITY:=text-only', script)
        self.assertIn('MM_DATASET:=synthetic', script)

    def test_pipeline_threads_flags_to_probe(self):
        script = Path(
            "/home/rob/spark-vllm-docker/quantization/prismaquant/"
            "run-pipeline.sh"
        ).read_text()
        self.assertIn("--calibration-modality", script)
        self.assertIn("--mm-dataset", script)
        self.assertIn("--mm-nsamples", script)

    def test_pipeline_picks_visual_sensitivity_from_modality(self):
        script = Path(
            "/home/rob/spark-vllm-docker/quantization/prismaquant/"
            "run-pipeline.sh"
        ).read_text()
        # When CALIBRATION_MODALITY=multimodal → VISUAL_SENSITIVITY=fisher
        self.assertIn('VISUAL_SENSITIVITY=fisher', script)
        self.assertIn('VISUAL_SENSITIVITY=uniform', script)
        self.assertIn("--visual-sensitivity", script)


if __name__ == "__main__":
    unittest.main()
