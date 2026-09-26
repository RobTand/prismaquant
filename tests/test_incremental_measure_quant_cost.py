import pickle
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
import torch.nn as nn

from prismaquant.incremental_measure_quant_cost import (
    _run_cost_measurement,
    _scheduled_probe_targets,
    cost_shard_is_reusable,
    merge_cost_pickles,
)
from prismaquant.measure_quant_cost import (
    ActivationIndex,
    HDetailIndex,
    canonical_linear_name,
    measure_batched_gpu,
    measure_unbatched,
)
from prismaquant.sensitivity_probe import FisherAccumulator
from prismaquant import format_registry as fr


class _CustomPackedCostProfile:
    def packed_expert_parent_for_projection(self, projection_name: str) -> str | None:
        if projection_name in {"w1_proj", "w3_proj"}:
            return "w13"
        if projection_name == "w2":
            return "w2"
        return None


class TestIncrementalMeasureQuantCost(unittest.TestCase):
    def test_canonical_linear_name_uses_profile_decomposition(self):
        profile = _CustomPackedCostProfile()

        self.assertEqual(
            canonical_linear_name(
                "model.layers.0.mlp.experts.3.w1_proj",
                profile,
            ),
            "model.layers.0.mlp.experts.w13.3",
        )
        self.assertEqual(
            canonical_linear_name(
                "model.layers.0.mlp.experts.3.w2",
                profile,
            ),
            "model.layers.0.mlp.experts.w2.3",
        )
        self.assertEqual(
            canonical_linear_name(
                "model.layers.0.mlp.experts.3.side",
                profile,
            ),
            "model.layers.0.mlp.experts.3.side",
        )

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

    def test_scheduled_probe_targets_scopes_optional_regions(self):
        stats = {
            "model.layers.0.self_attn.q_proj": {},
            "model.layers.0.mlp.experts": {},
            "model.layers.1.self_attn.q_proj": {},
            "mtp.layers.0.mlp.experts.down_proj": {},
            "model.visual.blocks.0.attn.qkv": {},
        }

        got = _scheduled_probe_targets(
            stats,
            [r"model\.layers\.0\."],
        )

        self.assertEqual(
            got,
            {
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.mlp.experts",
            },
        )

    def test_cost_shard_reuse_matches_menu_and_refuses_an_empty_hole(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "shard.pkl"
            payload = {
                "costs": {"layer.0": {"NVFP4": {"output_mse": 1.0}}},
                "formats": ["NVFP4"],
                "meta": {"incremental_shard": {"kind": "body"}},
            }
            with path.open("wb") as fh:
                pickle.dump(payload, fh)
            self.assertTrue(cost_shard_is_reusable(path, {"formats": ["NVFP4"]}))
            self.assertFalse(
                cost_shard_is_reusable(path, {"formats": ["NVFP4", "BF16"]}))

            payload["costs"] = {}
            with path.open("wb") as fh:
                pickle.dump(payload, fh)
            self.assertFalse(cost_shard_is_reusable(
                path, {"formats": ["NVFP4"], "n_linears_expected": 1}))

    def test_merge_refuses_a_retired_codebook_shard(self):
        # A stale shard naming a retired codebook rung (the retired codebook
        # lane, archived 2026-09-25, #1304) refuses instead of merging.
        fmt = "NVFP4_CB_K16"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "0.pkl"
            with path.open("wb") as fh:
                pickle.dump({
                    "costs": {"layer.0": {fmt: {"output_mse": 1.0}}},
                    "formats": [fmt],
                    "meta": {},
                }, fh)
            with self.assertRaisesRegex(fr.RetiredFormatError, "gridbook_lane"):
                merge_cost_pickles([path], root / "merged.pkl")
            self.assertFalse((root / "merged.pkl").exists())

    def test_batched_cost_matches_unbatched_for_grouped_linears(self):
        torch.manual_seed(0)

        model = nn.Module()
        model.a = nn.Linear(16, 4, bias=False)
        model.b = nn.Linear(16, 4, bias=False)
        model.c = nn.Linear(16, 4, bias=False)
        target_names = {"a", "b", "c"}
        specs = [fr.get_format("BF16"), fr.get_format("NVFP4")]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            act_dir = root / "act"
            h_dir = root / "h"
            act_dir.mkdir()
            h_dir.mkdir()

            for name in sorted(target_names):
                safe = ActivationIndex._FNAME_SUB.sub("__", name) + ".pt"
                torch.save(
                    {
                        "inputs": torch.randn(7, 16),
                        "row_indices": torch.arange(7, dtype=torch.long),
                        "name": name,
                    },
                    act_dir / safe,
                )
                torch.save(
                    {
                        "H": torch.rand(4, 16),
                        "g2_per_token": torch.linspace(0.25, 2.0, steps=7),
                        "name": name,
                        # Audit M9: incremental-style blobs must carry the
                        # per-token unit marker; unmarked raw "H" blobs are
                        # refused by HDetailIndex.h_diag_from_blob.
                        "units": "per_token",
                    },
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
                    "fisher_output_mse",
                ):
                    self.assertAlmostEqual(
                        batched[name][fmt][field],
                        unbatched[name][fmt][field],
                        places=6,
                        msg=f"{name} {fmt} {field}",
                    )

    def test_fisher_output_mse_uses_activation_row_indices(self):
        model = nn.Module()
        model.a = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.a.weight.copy_(torch.tensor([[1.0, 2.0]]))

        target_names = {"a"}
        spec = fr.get_format("FP8_E4M3")
        X = torch.tensor(
            [
                [0.2, -0.3],
                [3.0, 0.5],
                [-0.7, 1.2],
            ],
            dtype=torch.float32,
        )
        row_indices = torch.tensor([5, 2, 9], dtype=torch.long)
        g2 = torch.ones(10, dtype=torch.float32)
        g2[2] = 10.0
        g2[5] = 1.0
        g2[9] = 4.0

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            act_dir = root / "act"
            h_dir = root / "h"
            act_dir.mkdir()
            h_dir.mkdir()
            safe = ActivationIndex._FNAME_SUB.sub("__", "a") + ".pt"
            torch.save(
                {"inputs": X, "row_indices": row_indices, "name": "a"},
                act_dir / safe,
            )
            torch.save(
                {"h_diag": torch.ones(1, 2), "g2_per_token": g2, "name": "a"},
                h_dir / safe,
            )

            act_cache = ActivationIndex(act_dir, target_names)
            h_detail = HDetailIndex(h_dir, target_names)
            got = measure_unbatched(
                model,
                act_cache,
                target_names,
                [spec],
                device="cpu",
                dtype=torch.float32,
                h_detail=h_detail,
            )["a"][spec.name]

        W = model.a.weight.detach()
        W_hat = spec.quantize_dequantize(W.clone())
        y_err_sq = (X @ W.T - spec.activation_quantize_dequantize(X.clone()) @ W_hat.T).pow(2)
        weights = g2.index_select(0, row_indices)
        weights = weights / weights.mean()
        expected = float((y_err_sq * weights.unsqueeze(1)).mean().item())
        self.assertAlmostEqual(got["fisher_output_mse"], expected, places=6)

    def test_production_render_path_uses_production_renderer(self):
        model = nn.Module()
        model.a = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.a.weight.copy_(torch.tensor([[1.0, -2.0]]))

        target_names = {"a"}
        spec = fr.get_format("FP8_E4M3")
        X = torch.tensor([[0.5, -0.25], [1.0, 2.0]], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            act_dir = root / "act"
            act_dir.mkdir()
            safe = ActivationIndex._FNAME_SUB.sub("__", "a") + ".pt"
            torch.save({"inputs": X, "name": "a"}, act_dir / safe)
            act_cache = ActivationIndex(act_dir, target_names)

            calls = []

            def fake_render(weight, fmt, *, qname, activations, levers, **kwargs):
                calls.append((fmt, qname, sorted(activations), dict(levers)))
                return torch.zeros_like(weight)

            with patch(
                "prismaquant.production_weight_cache.render_production_weight",
                side_effect=fake_render,
            ):
                got = _run_cost_measurement(
                    model,
                    act_cache=act_cache,
                    target_names=target_names,
                    specs=[spec],
                    device="cpu",
                    dtype=torch.float32,
                    mode="unbatched",
                    chunk_size=1,
                    h_detail=None,
                    log_prefix="[test]",
                    render_path="production",
                )["a"][spec.name]

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "FP8_E4M3")
        self.assertEqual(calls[0][1], "a")
        self.assertEqual(calls[0][2], ["a"])
        self.assertTrue(calls[0][3]["gptq"])
        self.assertTrue(calls[0][3]["static_act_order"])
        self.assertEqual(got["render_path"], "production")
        self.assertGreater(got["output_mse"], 0.0)

    def test_fisher_accumulator_writes_mtp_h_detail_and_row_indices(self):
        class TinyMtpWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.mtp = nn.Module()
                self.mtp.fc = nn.Linear(2, 1, bias=False)

            def forward(self, x):
                return self.mtp.fc(x)

        torch.manual_seed(0)
        model = TinyMtpWrapper()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            act_dir = root / "act"
            h_dir = root / "h"
            acc = FisherAccumulator(
                model,
                ["mtp.fc"],
                {},
                act_cache_dir=act_dir,
                input_rows=3,
                h_detail_dir=h_dir,
            )
            try:
                x = torch.tensor(
                    [[[0.25, -0.5], [1.0, 0.75], [-0.125, 0.5]]],
                    dtype=torch.float32,
                    requires_grad=True,
                )
                loss = model(x).pow(2).sum()
                loss.backward()
                acc.finalize(tracker=None, global_tokens=3)
            finally:
                acc.remove_hooks()

            safe = ActivationIndex._FNAME_SUB.sub("__", "mtp.fc") + ".pt"
            act_payload = torch.load(act_dir / safe, map_location="cpu")
            h_payload = torch.load(h_dir / safe, map_location="cpu")

        self.assertEqual(act_payload["name"], "mtp.fc")
        self.assertEqual(tuple(act_payload["inputs"].shape), (3, 2))
        self.assertTrue(torch.equal(
            act_payload["row_indices"],
            torch.tensor([0, 1, 2], dtype=torch.long),
        ))
        self.assertEqual(h_payload["name"], "mtp.fc")
        self.assertEqual(h_payload["kind"], "linear")
        self.assertEqual(tuple(h_payload["h_diag"].shape), (1, 2))
        self.assertEqual(tuple(h_payload["g2_per_token"].shape), (3,))
        self.assertEqual(
            acc.stats["mtp.fc"]["h_detail_path"],
            safe,
        )

    def test_fisher_accumulator_skips_profile_pinned_head(self):
        class TinyHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(2, 2, bias=False)

        class Profile:
            def is_pinned_name(self, qname):
                return qname == "head"

        model = TinyHead()
        acc = FisherAccumulator(
            model,
            ["head"],
            {},
            model_profile=Profile(),
        )
        try:
            self.assertIn("head", acc._fisher_skip)
        finally:
            acc.remove_hooks()


if __name__ == "__main__":
    unittest.main()
