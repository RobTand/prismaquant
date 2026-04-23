import unittest

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.allocator import build_candidates
from prismaquant.calibrate_allocator import install_activation_hooks, select_targets
from prismaquant.sensitivity_probe import discover_moe_structure


class TestPrismaQuantFormatRegistry(unittest.TestCase):
    def test_block_formats_have_expected_shape_aware_bits(self):
        shape = (128, 128)
        self.assertAlmostEqual(fr.get_format("NVFP4").effective_bits_for_shape(shape), 4.5)
        self.assertAlmostEqual(fr.get_format("INT2").effective_bits_for_shape(shape), 2.5)
        self.assertAlmostEqual(fr.get_format("INT3").effective_bits_for_shape(shape), 3.5)
        self.assertAlmostEqual(fr.get_format("MXFP4").effective_bits_for_shape(shape), 4.25)
        self.assertAlmostEqual(fr.get_format("MXFP8").effective_bits_for_shape(shape), 8.25)
        self.assertAlmostEqual(fr.get_format("FP8_SOURCE").effective_bits_for_shape(shape), 8.001953125)
        self.assertAlmostEqual(fr.get_format("BF16").effective_bits_for_shape(shape), 16.0)

    def test_legacy_nvint_names_are_aliases(self):
        self.assertIs(fr.get_format("NVINT2"), fr.get_format("INT2"))
        self.assertIs(fr.get_format("NVINT3"), fr.get_format("INT3"))
        self.assertEqual(fr.canonical_format_name("NVINT2"), "INT2")
        self.assertEqual(fr.aliases_for("INT3"), ("INT3", "NVINT3"))

    def test_source_fp8_uses_2d_scale_blocks(self):
        spec = fr.get_format("FP8_SOURCE")
        shape = (256, 256)
        expected_bytes = 256 * 256 + 4 * 4  # fp8 weights + four fp32 128x128 scales

        self.assertAlmostEqual(spec.effective_bits, 8.001953125)
        self.assertEqual(spec.scale_count_for_shape(shape), 4)
        self.assertEqual(spec.memory_bytes_for_shape(shape), expected_bytes)
        self.assertAlmostEqual(
            spec.effective_bits_for_shape(shape),
            8.0 * expected_bytes / (256 * 256),
        )

    def test_per_channel_formats_use_row_scale_count(self):
        shape = (5, 7)
        spec = fr.get_format("FP8_E4M3")
        expected_bytes = 5 * 7 + 5 * 4  # one byte per weight, fp32 row scales
        self.assertEqual(spec.scale_count_for_shape(shape), 5)
        self.assertEqual(spec.memory_bytes_for_shape(shape), expected_bytes)
        self.assertAlmostEqual(spec.effective_bits_for_shape(shape), 8.0 * expected_bytes / 35.0)

    def test_activation_quantization_changes_native_a4_a8_formats(self):
        x = torch.tensor([[0.13, -0.51, 1.77, -3.25]], dtype=torch.float32)
        for fmt in ("NVFP4", "MXFP8", "FP8_E4M3"):
            y = fr.get_format(fmt).activation_quantize_dequantize(x.clone())
            self.assertFalse(torch.equal(x, y), msg=fmt)

    def test_activation_quantization_skips_a16_formats(self):
        x = torch.tensor([[0.13, -0.51, 1.77, -3.25]], dtype=torch.float32)
        for fmt in ("MXFP8A16", "NVFP4A16", "BF16"):
            y = fr.get_format(fmt).activation_quantize_dequantize(x.clone())
            self.assertTrue(torch.equal(x, y), msg=fmt)


class TestPrismaQuantAllocatorMath(unittest.TestCase):
    def test_build_candidates_uses_shape_aware_bits(self):
        # Predicted Δloss = 0.5 · h_trace · weight_mse  (closed-form
        # diagonal-Fisher term; see allocator.py module docstring eq. 3).
        stats = {
            "layer.weight": {
                "h_trace": 2.0,
                "out_features": 5,
                "in_features": 7,
                "n_params": 35,
            }
        }
        costs = {
            "layer.weight": {
                "FP8_E4M3": {"weight_mse": 0.10, "output_mse": 0.25},
                "BF16": {"weight_mse": 0.0, "output_mse": 0.0},
            }
        }
        cands = build_candidates(stats, costs, [fr.get_format("FP8_E4M3"), fr.get_format("BF16")])
        by_fmt = {cand.fmt: cand for cand in cands["layer.weight"]}
        self.assertAlmostEqual(
            by_fmt["FP8_E4M3"].bits_per_param,
            fr.get_format("FP8_E4M3").effective_bits_for_shape((5, 7)),
        )
        self.assertEqual(
            by_fmt["FP8_E4M3"].memory_bytes,
            fr.get_format("FP8_E4M3").memory_bytes_for_shape((5, 7)),
        )
        self.assertAlmostEqual(by_fmt["FP8_E4M3"].predicted_dloss, 0.5 * 2.0 * 0.10)
        self.assertAlmostEqual(by_fmt["BF16"].predicted_dloss, 0.0)

    def test_build_candidates_prices_source_fp8_below_mxfp8(self):
        stats = {
            "layer.weight": {
                "h_trace": 2.0,
                "out_features": 128,
                "in_features": 128,
                "n_params": 128 * 128,
            }
        }
        costs = {
            "layer.weight": {
                "FP8_SOURCE": {"weight_mse": 0.0},
                "MXFP8": {"weight_mse": 0.01},
            }
        }
        cands = build_candidates(
            stats,
            costs,
            [fr.get_format("FP8_SOURCE"), fr.get_format("MXFP8")],
        )
        by_fmt = {cand.fmt: cand for cand in cands["layer.weight"]}

        self.assertLess(by_fmt["FP8_SOURCE"].bits_per_param, by_fmt["MXFP8"].bits_per_param)
        self.assertEqual(by_fmt["FP8_SOURCE"].memory_bytes, 128 * 128 + 4)
        self.assertAlmostEqual(by_fmt["FP8_SOURCE"].bits_per_param, 8.001953125)
        self.assertAlmostEqual(by_fmt["MXFP8"].bits_per_param, 8.25)

    def test_build_candidates_applies_calibrated_gains(self):
        stats = {
            "layer.weight": {
                "h_trace": 2.0,
                "out_features": 4,
                "in_features": 4,
                "n_params": 16,
            }
        }
        costs = {
            "layer.weight": {
                "NVFP4": {"weight_mse": 0.10},
                "MXFP8": {"weight_mse": 0.02},
            }
        }
        # Without calibration: NVFP4 = 0.10, MXFP8 = 0.02 (per-element MSE).
        # With α_NVFP4=2 the NVFP4 candidate's predicted Δloss should double.
        cands = build_candidates(
            stats, costs,
            [fr.get_format("NVFP4"), fr.get_format("MXFP8")],
            calibrated_gains={"NVFP4": 2.0, "MXFP8": 1.0},
        )
        by_fmt = {c.fmt: c for c in cands["layer.weight"]}
        self.assertAlmostEqual(by_fmt["NVFP4"].predicted_dloss, 0.5 * 2.0 * 0.10 * 2.0)
        self.assertAlmostEqual(by_fmt["MXFP8"].predicted_dloss, 0.5 * 2.0 * 0.02 * 1.0)

    def test_select_targets_returns_baseline_knee_high(self):
        curve = [
            {"feasible": True, "achieved_bits": 4.5, "predicted_dloss": 10.0},
            {"feasible": True, "achieved_bits": 5.0, "predicted_dloss": 4.0},
            {"feasible": True, "achieved_bits": 6.0, "predicted_dloss": 3.0},
            {"feasible": True, "achieved_bits": 8.0, "predicted_dloss": 2.8},
        ]
        picks = select_targets(curve, "baseline,knee,high")
        self.assertEqual(picks[0], 0)
        self.assertEqual(picks[-1], 3)
        self.assertEqual(len(picks), 3)


class TestCalibrationHooks(unittest.TestCase):
    def test_install_activation_hooks_skips_conflicting_module_formats(self):
        linear = torch.nn.Linear(4, 4, bias=False)
        quant_map = {
            "a.weight": (linear, "weight"),
            "b.weight": (linear, "weight"),
        }
        handles, active, skipped = install_activation_hooks(
            {"a.weight": "NVFP4", "b.weight": "MXFP8"},
            quant_map,
        )
        try:
            self.assertEqual(active, [])
            self.assertEqual(len(skipped), 1)
            self.assertEqual(set(skipped[0]["formats"]), {"MXFP8", "NVFP4"})
        finally:
            for handle in handles:
                handle.remove()

    def test_install_activation_hooks_quantizes_input(self):
        linear = torch.nn.Linear(4, 4, bias=False)
        quant_map = {"a.weight": (linear, "weight")}
        handles, active, skipped = install_activation_hooks({"a.weight": "NVFP4"}, quant_map)
        seen = {}

        def recorder(_mod, args):
            seen["input"] = args[0].detach().clone()

        capture = linear.register_forward_pre_hook(recorder)
        x = torch.tensor([[0.13, -0.51, 1.77, -3.25]], dtype=torch.float32)
        try:
            linear(x)
            self.assertEqual(len(active), 1)
            self.assertEqual(skipped, [])
            self.assertFalse(torch.equal(seen["input"], x))
        finally:
            capture.remove()
            for handle in handles:
                handle.remove()


class _ToyExpertsLinearLoop(nn.Module):
    def __init__(self, num_experts=3, hidden=4, intermediate=6):
        super().__init__()
        self.gate_up_proj = nn.ModuleList(
            [nn.Linear(hidden, 2 * intermediate, bias=False) for _ in range(num_experts)]
        )
        self.down_proj = nn.ModuleList(
            [nn.Linear(intermediate, hidden, bias=False) for _ in range(num_experts)]
        )


class _ToyMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 3, bias=False)
        self.experts = _ToyExpertsLinearLoop()


class _ToyRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3, 4))


class _ToyMoeBlockCustomRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _ToyRouter()
        self.experts = _ToyExpertsLinearLoop()


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = _ToyMoeBlock()


class TestMoeDiscovery(unittest.TestCase):
    def test_discover_moe_structure_handles_linear_loop_projection_lists(self):
        toy = _ToyModel()
        info = discover_moe_structure(toy)
        self.assertEqual(info["model.layers.0.mlp.experts.gate_up_proj.0"], ("model.layers.0.mlp.gate", "0"))
        self.assertEqual(info["model.layers.0.mlp.experts.down_proj.2"], ("model.layers.0.mlp.gate", "2"))
        self.assertEqual(len(info), 6)

    def test_discover_moe_structure_handles_router_modules_with_weight(self):
        toy = _ToyModel()
        toy.model.layers[0].mlp = _ToyMoeBlockCustomRouter()
        info = discover_moe_structure(toy)
        self.assertEqual(info["model.layers.0.mlp.experts.gate_up_proj.1"], ("model.layers.0.mlp.gate", "1"))
        self.assertEqual(len(info), 6)


if __name__ == "__main__":
    unittest.main()
