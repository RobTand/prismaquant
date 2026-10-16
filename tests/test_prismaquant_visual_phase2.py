"""Tests for Phase 2 visual-encoder calibration.

Phase 2 adds real activation + Fisher capture for visual Linears so
they enter the allocator's DP as regular candidates and get the same
GPTQ/scale-sweep/AR treatment at export. Covers:

  1. sensitivity_probe.stage_multimodal — preserves vision_config (vs
     stage_text_only which strips it).
  2. sensitivity_probe.load_multimodal_calibration — synthetic-stub
     path, assert shapes and contract.
  3. incremental_probe CLI flag threading for the multimodal path.
  4. incremental_measure_quant_cost._run_visual_cost_shard — measures
     cost entries for a tiny visual-style nn.Module stub.
  5. allocator --visual-sensitivity={fisher, uniform} mode selection.
  6. export_native_compressed._apply_visual_recipe_quant passes the
     activations the multimodal probe cached under a visual Linear's recipe
     name to native_render_codec.quantize_2d.
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

import argparse
import contextlib
import io
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image

from prismaquant.sensitivity_probe import (
    _streaming_visual_layer_kwargs,
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
        triples, composition = load_multimodal_calibration(
            processor=None, dataset_name="synthetic",
            n_samples=3, max_text_len=12,
        )
        self.assertEqual(len(triples), 3)
        self.assertEqual(
            composition,
            {"dataset": "synthetic", "requested": 3,
             "real_loaded": 0, "synthetic_loaded": 3},
        )

    def test_synthetic_captions_rotate(self):
        # 12 samples with 8 built-in captions should wrap around cleanly.
        triples = _synthetic_multimodal_calibration_samples(
            processor=None, n_samples=12, max_text_len=16,
        )
        self.assertEqual(len(triples), 12)


class TestCalibrationCompositionProvenance(unittest.TestCase):
    """The real-vs-synthetic composition of multimodal calibration must be
    counted, printed and stamped into probe provenance (#1921 visual
    capture gate). Partial real `--mm-dataset` yields blend synthetic
    stub rows; the counts and persisted stamp must expose that blend.
    Whether blending should refuse outright stays the filed coordinator
    decision; this gate only makes the composition unmissable."""

    def test_synthetic_shortfall_records_rows_returned(self):
        # The stub continues past rows whose processor calls fail, so the
        # pure-synthetic path can return fewer rows than requested. The
        # composition must record the rows actually returned, never the
        # requested count.
        class _FailingProcessor:
            def apply_chat_template(self, *args, **kwargs):
                raise NotImplementedError("no chat template")

            def __call__(self, *args, **kwargs):
                raise RuntimeError("processor failure")

        triples, composition = load_multimodal_calibration(
            _FailingProcessor(), dataset_name="synthetic",
            n_samples=3, max_text_len=12,
        )
        self.assertEqual(triples, [])
        self.assertEqual(
            composition,
            {"dataset": "synthetic", "requested": 3,
             "real_loaded": 0, "synthetic_loaded": 0},
        )

    def test_partial_real_dataset_blend_is_counted_and_loud(self):
        from prismaquant import sensitivity_probe as sp

        class _FakeDataset:
            def __init__(self, rows):
                self._rows = rows

            def __iter__(self):
                return iter(self._rows)

        img = Image.new("RGB", (4, 4))
        enc = {
            "pixel_values": torch.zeros(1, 3, 4, 4),
            "input_ids": torch.tensor([[1, 2, 3]]),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }
        proc = mock.Mock(return_value=enc)
        # A real AutoProcessor without chat-template support raises from
        # apply_chat_template; a bare Mock would return a Mock and the
        # synthetic stub would then hand Mock tensors to
        # _samples_from_encoding.
        proc.apply_chat_template = mock.Mock(
            side_effect=NotImplementedError("no chat template"))
        stub = mock.Mock()
        stub.load_dataset = mock.Mock(
            return_value=_FakeDataset(
                [{"image": img, "caption": "a real photo"}]))
        buffer = io.StringIO()
        with mock.patch.dict(sys.modules, {"datasets": stub}), \
                contextlib.redirect_stdout(buffer):
            triples, composition = sp.load_multimodal_calibration(
                proc, "real-vision-dataset", n_samples=3, max_text_len=12)
        self.assertEqual(len(triples), 3)
        self.assertEqual(
            composition,
            {"dataset": "real-vision-dataset", "requested": 3,
             "real_loaded": 1, "synthetic_loaded": 2},
        )
        self.assertIn("synthetic stub row(s) were blended",
                      buffer.getvalue())

    def test_fully_real_dataset_reports_zero_synthetic(self):
        from prismaquant import sensitivity_probe as sp

        class _FakeDataset:
            def __init__(self, rows):
                self._rows = rows

            def __iter__(self):
                return iter(self._rows)

        img = Image.new("RGB", (4, 4))
        enc = {
            "pixel_values": torch.zeros(1, 3, 4, 4),
            "input_ids": torch.tensor([[1, 2, 3]]),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }
        proc = mock.Mock(return_value=enc)
        proc.apply_chat_template = mock.Mock(
            side_effect=NotImplementedError("no chat template"))
        stub = mock.Mock()
        stub.load_dataset = mock.Mock(
            return_value=_FakeDataset(
                [{"image": img, "caption": f"photo {i}"} for i in range(5)]))
        buffer = io.StringIO()
        with mock.patch.dict(sys.modules, {"datasets": stub}), \
                contextlib.redirect_stdout(buffer):
            triples, composition = sp.load_multimodal_calibration(
                proc, "real-vision-dataset", n_samples=3, max_text_len=12)
        self.assertEqual(len(triples), 3)
        self.assertEqual(
            composition,
            {"dataset": "real-vision-dataset", "requested": 3,
             "real_loaded": 3, "synthetic_loaded": 0},
        )
        self.assertNotIn("blended", buffer.getvalue())


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

    def test_meta_visual_module_detection_includes_buffers(self):
        from prismaquant.streaming_model import _module_has_meta_tensors

        class TinyVisual(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(2, 2, device="meta")
                self.register_buffer("pos", torch.empty(2, device="meta"))

        module = TinyVisual()
        self.assertTrue(_module_has_meta_tensors(module))
        module.to_empty(device="cpu")
        self.assertFalse(_module_has_meta_tensors(module))

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
            # New signature: returns `mm_ctx` (StreamingContext or None).
            # For an empty probe_stats shard, the function short-circuits
            # before building any context — so mm_ctx stays None. What
            # we care about is that an empty pickle was emitted.
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
                mm_ctx=None,
                mm_offload_folder=str(td / "offload"),
            )
            # Empty-shard short-circuit never builds a context.
            self.assertIsNone(result)
            self.assertTrue(out.exists())
            with open(out, "rb") as f:
                data = pickle.load(f)
            self.assertEqual(data["costs"], {})
            self.assertEqual(data["meta"]["shard_kind"], "visual")
            # Formats should still be recorded for merge consistency.
            self.assertEqual(data["formats"], ["BF16", "NVFP4"])

    def test_visual_shard_writes_empty_pickle_on_load_oom(self):
        """When the multimodal streaming context build raises CUDA OOM,
        we fall back to an empty pickle instead of raising — 122B-scale
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
            # Patch the streaming-context builder to raise a plain
            # RuntimeError with "out of memory" in the message — the
            # guard should catch that as an OOM proxy and fall back
            # gracefully with an empty pickle.
            with mock.patch(
                    "prismaquant.incremental_measure_quant_cost."
                    "_build_streaming_context",
                    side_effect=RuntimeError("CUDA out of memory")):
                result = _run_visual_cost_shard(
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
                    mm_ctx=None,
                    mm_offload_folder=str(td / "offload"),
                )
            # OOM path returns None (no context built) but writes an
            # empty pickle so the merge layout stays consistent.
            self.assertIsNone(result)
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


class TestStreamingMultimodalProfileState(unittest.TestCase):
    def test_streaming_visual_kwargs_merge_profile_and_pass_state(self):
        class Profile:
            def __init__(self):
                self.extra_input_ids = []
                self.isolated_calls = []

            def extra_layer_kwargs(self, *, input_ids=None):
                self.extra_input_ids.append(input_ids)
                return {"profile_input_ids": input_ids}

            def isolated_layer_pass_state(self, captured, layer):
                self.isolated_calls.append((captured, layer))
                return {"shared_kv": captured["shared_kv"]}

        profile = Profile()
        input_ids = torch.tensor([[1, 2, 3]])
        layer = object()

        forward = _streaming_visual_layer_kwargs(
            profile,
            input_ids=input_ids,
            pass_state={"live_state": "forward"},
        )
        self.assertIs(forward["profile_input_ids"], input_ids)
        self.assertEqual(forward["live_state"], "forward")
        self.assertEqual(profile.extra_input_ids, [input_ids])

        replay = _streaming_visual_layer_kwargs(
            profile,
            input_ids=input_ids,
            captured_pass_state={"shared_kv": "captured"},
            layer=layer,
        )
        self.assertIs(replay["profile_input_ids"], input_ids)
        self.assertEqual(replay["shared_kv"], "captured")
        self.assertEqual(profile.isolated_calls, [({"shared_kv": "captured"},
                                                   layer)])

    def test_streaming_probe_threads_profile_pass_state(self):
        from prismaquant import sensitivity_probe as sp

        src = inspect_source(sp.run_streaming_multimodal_visual_probe_pass)
        helper_src = inspect_source(sp._streaming_visual_layer_kwargs)
        self.assertIn("new_forward_pass_state", src)
        self.assertIn("capture_forward_pass_state", src)
        self.assertIn("isolated_layer_pass_state", helper_src)
        self.assertIn("_streaming_visual_layer_kwargs", src)


class TestMultimodalProbePassIntegration(unittest.TestCase):
    """Persist actual loader provenance through populated CPU probe passes."""

    class Processor:
        def __init__(self):
            self.rows = 0

        def apply_chat_template(self, *args, **kwargs):
            raise NotImplementedError("no chat template")

        def __call__(self, **kwargs):
            self.rows += 1
            return {
                "pixel_values": torch.tensor(
                    [[float(self.rows), 0.5]], requires_grad=True),
                "input_ids": torch.tensor([[0, 1, 1]]),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            }

    class BodyLayer(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states * 1.5

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Sequential(nn.Linear(2, 2, bias=False))
            self.base = nn.Module()
            self.base.embed_tokens = nn.Embedding(2, 2)
            self.base.norm = nn.Identity()
            self.base.layers = nn.ModuleList([
                TestMultimodalProbePassIntegration.BodyLayer()])
            self.lm_head = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.visual[0].weight.copy_(
                    torch.tensor([[0.2, 0.1], [-0.1, 0.3]]))
                self.base.embed_tokens.weight.zero_()
                self.lm_head.weight.copy_(
                    torch.tensor([[0.3, -0.2], [-0.1, 0.2]]))
            self.attempted_rows = []

        def get_image_features(self, pixel_values, **kwargs):
            row = int(pixel_values[0, 0].detach())
            self.attempted_rows.append(row)
            if row == 2:
                raise RuntimeError("deliberate synthetic-row forward failure")
            return self.visual(pixel_values)

        def get_placeholder_mask(self, input_ids, *, inputs_embeds,
                                 image_features):
            return (input_ids == 0).unsqueeze(-1).expand_as(inputs_embeds), None

        def forward(self, input_ids, pixel_values, labels, **kwargs):
            features = self.get_image_features(pixel_values)
            embeds = self.base.embed_tokens(input_ids)
            mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=embeds, image_features=features)
            hidden = embeds.masked_scatter(mask, features)
            hidden = self.base.layers[0](hidden_states=hidden)
            return SimpleNamespace(logits=self.lm_head(self.base.norm(hidden)))

    def _probe_pickle(self, *, streaming, tracked=True):
        from prismaquant import sensitivity_probe as sp

        model = self.Model()
        processor = self.Processor()
        # Only external dataset/model loading is replaced. The actual loader
        # accepts one real row and constructs two synthetic rows itself.
        dataset = SimpleNamespace(load_dataset=lambda *args, **kwargs: iter([
            {"image": Image.new("RGB", (4, 4)), "caption": "real photo"}]))
        ctx = SimpleNamespace(
            model=model, base_model=model.base, layers=model.base.layers,
            num_layers=1, layers_prefix="base.layers.",
            visual_module=model.visual, visual_prefix="visual.",
            layer_cache={}, schedule_prefetch=lambda index: None,
            install=lambda index: None, unload=lambda index: None,
            shutdown=lambda: None,
        )
        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            root = Path(td)
            (root / "config.json").write_text(json.dumps({"model_type": "dummy"}))
            out = root / "visual_probe.pkl"
            stack.enter_context(mock.patch.dict(sys.modules, {"datasets": dataset}))
            stack.enter_context(mock.patch(
                "transformers.AutoProcessor.from_pretrained", return_value=processor))
            if streaming:
                stack.enter_context(mock.patch(
                    "prismaquant.streaming_model._build_streaming_context",
                    return_value=ctx))
                probe = sp.run_streaming_multimodal_visual_probe_pass
                extra = {"offload_folder": str(root / "offload")}
            else:
                stack.enter_context(mock.patch(
                    "transformers.AutoConfig.from_pretrained",
                    return_value=SimpleNamespace(architectures=[])))
                stack.enter_context(mock.patch(
                    "transformers.AutoModelForCausalLM.from_pretrained",
                    return_value=model))
                probe = sp.run_multimodal_visual_probe_pass
                extra = {}
            self.assertTrue(probe(
                str(root), dataset_name="fixture-real-vision", n_samples=3,
                max_text_len=3, requested_device="cpu", dtype=torch.float32,
                linear_include=r"^visual\." if tracked else r"^absent\.",
                linear_exclude=r"(?!x)x", activation_cache_dir=None,
                output_path=str(out), **extra))
            with out.open("rb") as stream:
                payload = pickle.load(stream)
        self.assertEqual(processor.rows, 3)
        self.assertEqual(payload["meta"]["calibration_source"], {
            "dataset": "fixture-real-vision", "requested": 3,
            "real_loaded": 1, "synthetic_loaded": 2,
        })
        self.assertEqual(payload["meta"]["execution_device"], "cpu")
        if tracked:
            self.assertEqual(model.attempted_rows, [1, 2, 3])
            stat = payload["stats"]["visual.0"]
            self.assertEqual(stat["n_tokens_seen"], 2)
            self.assertGreater(stat["h_trace"], 0)
        else:
            self.assertEqual(model.attempted_rows, [])
            self.assertEqual(payload["stats"], {})
        return payload

    def test_nonstreaming_pickle_preserves_blend_after_failed_forward(self):
        payload = self._probe_pickle(streaming=False)
        # The non-streaming pass retains its existing loaded-row budget.
        self.assertEqual(payload["meta"]["nsamples"], 3)
        self.assertEqual(payload["meta"]["fisher_norm_tokens"], 9)

    def test_streaming_pickle_keeps_loaded_composition_separate_from_successes(self):
        payload = self._probe_pickle(streaming=True)
        self.assertEqual(payload["meta"]["nsamples"], 2)
        self.assertEqual(payload["meta"]["fisher_norm_tokens"], 6)

    def test_streaming_no_match_pickle_preserves_actual_loader_composition(self):
        payload = self._probe_pickle(streaming=True, tracked=False)
        self.assertEqual(payload["meta"]["nsamples"], 3)
        self.assertTrue(payload["meta"]["streaming"])

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
                                   return_value=([], {
                                       "dataset": "synthetic",
                                       "requested": 2,
                                       "real_loaded": 0,
                                       "synthetic_loaded": 0,
                                   })):
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
        script = (
            Path(__file__).resolve().parent.parent / "prismaquant" / "run-pipeline.sh"
        ).read_text()
        self.assertIn('CALIBRATION_MODALITY:=text-only', script)
        self.assertIn('MM_DATASET:=synthetic', script)

    def test_pipeline_threads_flags_to_probe(self):
        script = (
            Path(__file__).resolve().parent.parent / "prismaquant" / "run-pipeline.sh"
        ).read_text()
        self.assertIn("--calibration-modality", script)
        self.assertIn("--mm-dataset", script)
        self.assertIn("--mm-nsamples", script)

    def test_pipeline_picks_visual_sensitivity_from_modality(self):
        script = (
            Path(__file__).resolve().parent.parent / "prismaquant" / "run-pipeline.sh"
        ).read_text()
        # When CALIBRATION_MODALITY=multimodal → VISUAL_SENSITIVITY=fisher
        self.assertIn('VISUAL_SENSITIVITY=fisher', script)
        self.assertIn('VISUAL_SENSITIVITY=uniform', script)
        self.assertIn("--visual-sensitivity", script)


if __name__ == "__main__":
    unittest.main()
