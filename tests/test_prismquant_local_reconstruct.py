import unittest

import torch

from prismquant import format_registry as fr
from prismquant.local_reconstruct import (
    _gptq_lite_refine_rows,
    _groupwise_refine_weight_clip,
    _measure_entry,
    _refine_measurement,
    _summarize_weight_clip,
    _sym_clip,
)


class TestLocalReconstruct(unittest.TestCase):
    def test_sym_clip_accepts_rowwise_tensor_factor(self):
        x = torch.tensor([[2.0, -1.0], [4.0, -3.0]], dtype=torch.float32)
        factor = torch.tensor([[0.5], [1.0]], dtype=torch.float32)
        y = _sym_clip(x, factor)
        self.assertTrue(torch.allclose(y[0], torch.tensor([1.0, -1.0])))
        self.assertTrue(torch.allclose(y[1], x[1]))

    def test_sym_clip_accepts_groupwise_tensor_factor(self):
        x = torch.tensor([[4.0, -3.0, 2.0, -1.0]], dtype=torch.float32)
        factor = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
        y = _sym_clip(x, factor, group_size=2)
        self.assertTrue(torch.allclose(y, torch.tensor([[2.0, -2.0, 2.0, -1.0]])))

    def test_rowwise_refine_produces_rowwise_summary(self):
        torch.manual_seed(0)
        W = torch.tensor(
            [
                [4.0, -4.0, 0.2, -0.1],
                [0.3, -0.2, 0.1, -0.1],
                [0.4, 0.5, -0.2, 0.2],
            ],
            dtype=torch.float32,
        )
        X = torch.randn(16, 4)
        entry = _refine_measurement(
            W,
            X,
            fr.get_format("NVFP4"),
            [1.0, 0.98, 0.95],
            [1.0, 0.98],
            rounds=1,
            rowwise_topk=2,
            rowwise_rounds=1,
            groupwise_topk=0,
            groupwise_rounds=0,
            gptq_topk=0,
            gptq_damping=1e-4,
        )
        self.assertIsNotNone(entry)
        summary = _summarize_weight_clip(entry["weight_clip"])
        self.assertEqual(summary["mode"], "rowwise")
        self.assertEqual(len(summary["values"]), W.shape[0])

    def test_groupwise_refine_produces_groupwise_summary(self):
        torch.manual_seed(0)
        W = torch.tensor(
            [
                [6.0, -5.5, 0.2, -0.1],
                [0.3, -0.2, 4.5, -4.0],
            ],
            dtype=torch.float32,
        )
        X = torch.randn(16, 4)
        spec = fr.get_format("NVFP4")
        base = _measure_entry(W, X, spec, 1.0, 1.0)
        entry = _groupwise_refine_weight_clip(
            W,
            X,
            spec,
            base,
            groupwise_topk=2,
            groupwise_rounds=1,
        )
        self.assertIsNotNone(entry)
        summary = _summarize_weight_clip(entry["weight_clip"])
        self.assertIn(summary["mode"], {"rowwise", "groupwise"})
        self.assertEqual(len(summary["values"]), W.shape[0] * (W.shape[1] // fr.get_format("NVFP4").group_size))

    def test_gptq_lite_refine_rows_is_non_regressing_on_small_case(self):
        torch.manual_seed(0)
        W = torch.tensor(
            [
                [5.0, -4.5, 0.4, -0.2],
                [0.2, -0.1, 3.5, -3.0],
            ],
            dtype=torch.float32,
        )
        X = torch.randn(32, 4)
        spec = fr.get_format("NVFP4")
        base = _measure_entry(W, X, spec, 1.0, 1.0)
        refined = _gptq_lite_refine_rows(
            W,
            X,
            spec,
            base,
            gptq_topk=2,
            gptq_damping=1e-4,
        )
        self.assertTrue(refined["output_mse"] <= base["output_mse"] + 1e-9)


if __name__ == "__main__":
    unittest.main()
