import pickle
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from prismaquant.incremental_measure_quant_cost import merge_cost_pickles
from prismaquant.measure_quant_cost import (
    ActivationIndex,
    HDetailIndex,
    measure_batched_gpu,
    measure_unbatched,
)
from prismaquant import format_registry as fr


class TestIncrementalMeasureQuantCost(unittest.TestCase):
    def test_merge_cost_pickles_combines_disjoint_shards(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p1 = td / "a.pkl"
            p2 = td / "b.pkl"
            out = td / "merged.pkl"
            with open(p1, "wb") as f:
                pickle.dump({
                    "costs": {"layer.0": {"NVFP4": {"output_mse": 1.0}}},
                    "formats": ["NVFP4"],
                    "meta": {"part": 1},
                }, f)
            with open(p2, "wb") as f:
                pickle.dump({
                    "costs": {"layer.1": {"BF16": {"output_mse": 0.0}}},
                    "formats": ["NVFP4"],
                    "meta": {"part": 2},
                }, f)

            merge_cost_pickles([p1, p2], out)
            with open(out, "rb") as f:
                merged = pickle.load(f)
            self.assertEqual(set(merged["costs"]), {"layer.0", "layer.1"})
            self.assertEqual(merged["formats"], ["NVFP4"])
            self.assertEqual(merged["meta"]["n_shards"], 2)

    def test_batched_cost_matches_unbatched_for_grouped_linears(self):
        torch.manual_seed(0)

        model = nn.Module()
        model.a = nn.Linear(16, 4, bias=False)
        model.b = nn.Linear(16, 4, bias=False)
        model.c = nn.Linear(16, 4, bias=False)
        target_names = {"a", "b", "c"}
        specs = [fr.get_format("BF16"), fr.get_format("NVINT3")]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            act_dir = root / "act"
            h_dir = root / "h"
            act_dir.mkdir()
            h_dir.mkdir()

            for name in sorted(target_names):
                safe = ActivationIndex._FNAME_SUB.sub("__", name) + ".pt"
                torch.save(
                    {"inputs": torch.randn(7, 16), "name": name},
                    act_dir / safe,
                )
                torch.save(
                    {"h_diag": torch.rand(4, 16), "name": name},
                    h_dir / safe,
                )

            act_cache = ActivationIndex(act_dir, target_names)
            h_detail = HDetailIndex(h_dir, target_names)
            unbatched = measure_unbatched(
                model,
                act_cache,
                target_names,
                specs,
                device="cpu",
                dtype=torch.float32,
                h_detail=h_detail,
            )
            batched = measure_batched_gpu(
                model,
                act_cache,
                target_names,
                specs,
                device="cpu",
                dtype=torch.float32,
                chunk_size=2,
                h_detail=h_detail,
            )

        self.assertEqual(set(batched), target_names)
        self.assertEqual(set(unbatched), target_names)
        for name in sorted(target_names):
            self.assertEqual(set(batched[name]), {s.name for s in specs})
            for spec in specs:
                fmt = spec.name
                for field in (
                    "weight_mse",
                    "output_mse",
                    "rel_output_mse",
                    "predicted_dloss",
                ):
                    self.assertAlmostEqual(
                        batched[name][fmt][field],
                        unbatched[name][fmt][field],
                        places=6,
                        msg=f"{name} {fmt} {field}",
                    )


if __name__ == "__main__":
    unittest.main()
