import unittest

from prismquant.allocator import Candidate
from prismquant.interaction_refine import (
    build_refinement_units,
    expand_unit_assignment,
    neighborhood_options,
    select_critical_units,
    sparse_local_refine,
)
from prismquant.local_reconstruct import expand_live_target_layers


class TestInteractionRefine(unittest.TestCase):
    def test_build_refinement_units_groups_fused_mlp_siblings(self):
        stats = {
            "model.layers.0.mlp.gate_proj": {"n_params": 8, "out_features": 4, "in_features": 2},
            "model.layers.0.mlp.up_proj": {"n_params": 8, "out_features": 4, "in_features": 2},
            "model.layers.0.self_attn.q_proj": {"n_params": 8, "out_features": 4, "in_features": 2},
        }
        candidates = {
            name: [
                Candidate("NVFP4", 4.5, 0, 10.0),
                Candidate("MXFP8", 8.25, 0, 1.0),
            ]
            for name in stats
        }
        assignment = {
            "model.layers.0.mlp.gate_proj": "NVFP4",
            "model.layers.0.mlp.up_proj": "NVFP4",
            "model.layers.0.self_attn.q_proj": "MXFP8",
        }
        units = build_refinement_units(stats, candidates, assignment)
        by_key = {u.key: u for u in units}
        fused_key = "model.layers.0.mlp.gate_proj|model.layers.0.mlp.up_proj"
        self.assertIn(fused_key, by_key)
        self.assertEqual(by_key[fused_key].base_fmt, "NVFP4")
        self.assertEqual({opt.fmt for opt in by_key[fused_key].options}, {"NVFP4", "MXFP8"})

    def test_select_critical_units_prefers_high_precision_benefit(self):
        stats = {
            "a": {"n_params": 8, "out_features": 4, "in_features": 2},
            "b": {"n_params": 8, "out_features": 4, "in_features": 2},
        }
        candidates = {
            "a": [Candidate("NVFP4", 4.5, 0, 10.0), Candidate("MXFP8", 8.25, 0, 2.0)],
            "b": [Candidate("NVFP4", 4.5, 0, 6.0), Candidate("MXFP8", 8.25, 0, 5.5)],
        }
        assignment = {"a": "MXFP8", "b": "MXFP8"}
        units = build_refinement_units(stats, candidates, assignment)
        picked = select_critical_units(units, 1)
        self.assertEqual(picked[0].key, "a")

    def test_sparse_local_refine_uses_pairwise_synergy(self):
        units = build_refinement_units(
            {
                "a": {"n_params": 8, "out_features": 4, "in_features": 2},
                "b": {"n_params": 8, "out_features": 4, "in_features": 2},
            },
            {
                "a": [Candidate("NVFP4", 4.5, 0, 2.0), Candidate("MXFP8", 8.25, 0, 0.8)],
                "b": [Candidate("NVFP4", 4.5, 0, 2.0), Candidate("MXFP8", 8.25, 0, 0.8)],
            },
            {"a": "NVFP4", "b": "NVFP4"},
        )
        allowed = {u.key: neighborhood_options(u, 1) for u in units}
        unary = {
            "a": {"NVFP4": 0.0, "MXFP8": 0.9},
            "b": {"NVFP4": 0.0, "MXFP8": 0.9},
        }
        # Each single upgrade hurts KL, but upgrading both together helps.
        pairwise = {("a", "MXFP8", "b", "MXFP8"): -2.2}
        base_bits = sum(u.option_map[u.base_fmt].bits_total for u in units)
        refined = sparse_local_refine(
            units=units,
            unary=unary,
            pairwise=pairwise,
            target_total_bits=base_bits + 100.0,
            fixed_bits_total=0.0,
            allowed=allowed,
            max_passes=4,
        )
        self.assertEqual(refined["choices"]["a"], "MXFP8")
        self.assertEqual(refined["choices"]["b"], "MXFP8")
        self.assertLess(refined["objective_delta"], 0.0)

    def test_expand_unit_assignment_projects_to_members(self):
        units = build_refinement_units(
            {
                "model.layers.0.mlp.gate_proj": {"n_params": 8, "out_features": 4, "in_features": 2},
                "model.layers.0.mlp.up_proj": {"n_params": 8, "out_features": 4, "in_features": 2},
            },
            {
                "model.layers.0.mlp.gate_proj": [Candidate("NVFP4", 4.5, 0, 1.0), Candidate("MXFP8", 8.25, 0, 0.5)],
                "model.layers.0.mlp.up_proj": [Candidate("NVFP4", 4.5, 0, 1.0), Candidate("MXFP8", 8.25, 0, 0.5)],
            },
            {
                "model.layers.0.mlp.gate_proj": "NVFP4",
                "model.layers.0.mlp.up_proj": "NVFP4",
            },
        )
        choice = {units[0].key: "MXFP8"}
        expanded = expand_unit_assignment(units, choice)
        self.assertEqual(expanded["model.layers.0.mlp.gate_proj"], "MXFP8")
        self.assertEqual(expanded["model.layers.0.mlp.up_proj"], "MXFP8")

    def test_expand_unit_assignment_restores_mixed_base_state(self):
        stats = {
            "model.layers.0.self_attn.q_proj": {"n_params": 8, "out_features": 4, "in_features": 2},
            "model.layers.0.self_attn.k_proj": {"n_params": 8, "out_features": 4, "in_features": 2},
        }
        candidates = {
            name: [Candidate("NVFP4", 4.5, 0, 10.0), Candidate("MXFP8", 8.25, 0, 1.0)]
            for name in stats
        }
        units = build_refinement_units(
            stats,
            candidates,
            {
                "model.layers.0.self_attn.q_proj": "MXFP8",
                "model.layers.0.self_attn.k_proj": "NVFP4",
            },
            unit_scope="layer",
        )
        self.assertEqual(units[0].base_fmt, "__base__")
        expanded = expand_unit_assignment(units, {units[0].key: "__base__"})
        self.assertEqual(expanded["model.layers.0.self_attn.q_proj"], "MXFP8")
        self.assertEqual(expanded["model.layers.0.self_attn.k_proj"], "NVFP4")

    def test_expand_live_target_layers_unrolls_fused_moe_members(self):
        stats = {
            "model.layers.0.mlp.experts.__fused__.gate_proj": {
                "n_params": 8,
                "out_features": 4,
                "in_features": 2,
                "_fused_members": (
                    "model.layers.0.mlp.experts.0.gate_proj",
                    "model.layers.0.mlp.experts.1.gate_proj",
                ),
            }
        }
        candidates = {
            "model.layers.0.mlp.experts.__fused__.gate_proj": [
                Candidate("NVFP4", 4.5, 0, 10.0),
                Candidate("MXFP8", 8.25, 0, 1.0),
            ]
        }
        units = build_refinement_units(
            stats,
            candidates,
            {"model.layers.0.mlp.experts.__fused__.gate_proj": "NVFP4"},
        )
        expanded = expand_live_target_layers(units, stats)
        self.assertEqual(
            expanded,
            {
                "model.layers.0.mlp.experts.0.gate_proj",
                "model.layers.0.mlp.experts.1.gate_proj",
            },
        )

    def test_build_refinement_units_block_scope_groups_attention_block(self):
        stats = {
            f"model.layers.0.self_attn.{name}": {"n_params": 8, "out_features": 4, "in_features": 2}
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        }
        candidates = {
            name: [
                Candidate("NVFP4", 4.5, 0, 10.0),
                Candidate("MXFP8", 8.25, 0, 1.0),
            ]
            for name in stats
        }
        assignment = {name: "NVFP4" for name in stats}
        units = build_refinement_units(stats, candidates, assignment, unit_scope="block")
        self.assertEqual(len(units), 1)
        self.assertEqual(
            units[0].members,
            tuple(sorted(stats.keys())),
        )

    def test_build_refinement_units_layer_scope_groups_full_layer(self):
        stats = {
            f"model.layers.0.self_attn.{name}": {"n_params": 8, "out_features": 4, "in_features": 2}
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        }
        stats.update(
            {
                f"model.layers.0.mlp.{name}": {"n_params": 8, "out_features": 4, "in_features": 2}
                for name in ("gate_proj", "up_proj", "down_proj")
            }
        )
        candidates = {
            name: [
                Candidate("NVFP4", 4.5, 0, 10.0),
                Candidate("MXFP8", 8.25, 0, 1.0),
            ]
            for name in stats
        }
        assignment = {name: "NVFP4" for name in stats}
        units = build_refinement_units(stats, candidates, assignment, unit_scope="layer")
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].members, tuple(sorted(stats.keys())))

    def test_build_refinement_units_layer_scope_keeps_mixed_base(self):
        stats = {
            "model.layers.0.self_attn.q_proj": {"n_params": 8, "out_features": 4, "in_features": 2},
            "model.layers.0.self_attn.k_proj": {"n_params": 8, "out_features": 4, "in_features": 2},
            "model.layers.0.mlp.gate_proj": {"n_params": 8, "out_features": 4, "in_features": 2},
        }
        candidates = {
            name: [Candidate("NVFP4", 4.5, 0, 10.0), Candidate("MXFP8", 8.25, 0, 1.0)]
            for name in stats
        }
        units = build_refinement_units(
            stats,
            candidates,
            {
                "model.layers.0.self_attn.q_proj": "MXFP8",
                "model.layers.0.self_attn.k_proj": "NVFP4",
                "model.layers.0.mlp.gate_proj": "NVFP4",
            },
            unit_scope="layer",
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].base_fmt, "__base__")
        self.assertIn("__base__", {opt.fmt for opt in units[0].options})


if __name__ == "__main__":
    unittest.main()
