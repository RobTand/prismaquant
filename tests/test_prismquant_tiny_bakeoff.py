import tempfile
import unittest
from types import SimpleNamespace

from prismquant.tiny_bakeoff import build_bakeoff_commands


class TestTinyBakeoff(unittest.TestCase):
    def _args(self, skip_oracle=False):
        td = tempfile.mkdtemp()
        return SimpleNamespace(
            model="/tmp/model",
            probe="/tmp/probe.pkl",
            costs="/tmp/costs.pkl",
            activation_cache_dir="/tmp/act",
            formats="NVFP4,MXFP8,BF16",
            target_bits=4.8,
            target_band=0.1,
            target_grid="",
            top_units=6,
            unit_scope="block",
            neighbor_radius=1,
            refine_rounds=2,
            rowwise_topk=8,
            rowwise_rounds=1,
            groupwise_topk=16,
            groupwise_rounds=1,
            gptq_topk=8,
            gptq_damping=1e-4,
            n_calib_samples=2,
            calib_seqlen=64,
            device="cuda",
            oracle_max_combos=1024,
            output_dir=td,
            skip_oracle=skip_oracle,
            dry_run=True,
        )

    def test_build_bakeoff_commands_with_oracle(self):
        paths_by_target, cmds = build_bakeoff_commands(self._args(skip_oracle=False))
        self.assertEqual(len(paths_by_target), 3)
        self.assertEqual(len(cmds), 18)
        first_target = sorted(paths_by_target)[0]
        first_paths = paths_by_target[first_target]
        self.assertIn("prismquant.calibrate_allocator", cmds[2])
        self.assertIn("prismquant.quadratic_refine_allocator", cmds[3])
        self.assertIn("--calibration", cmds[3])
        self.assertIn(str(first_paths["calibration"]), cmds[3])
        self.assertIn("--unit-scope", cmds[0])
        self.assertIn("block", cmds[0])
        self.assertIn("--refine-rounds", cmds[0])
        self.assertIn("--rowwise-topk", cmds[0])
        self.assertIn("--rowwise-rounds", cmds[0])
        self.assertIn("--groupwise-topk", cmds[0])
        self.assertIn("--groupwise-rounds", cmds[0])
        self.assertIn("--gptq-topk", cmds[0])
        self.assertIn("--gptq-damping", cmds[0])
        self.assertIn("prismquant.oracle_search", cmds[4])
        self.assertIn("--oracle", cmds[-1])
        self.assertTrue(str(first_paths["oracle"]).endswith("oracle.json"))

    def test_build_bakeoff_commands_without_oracle(self):
        _paths, cmds = build_bakeoff_commands(self._args(skip_oracle=True))
        self.assertEqual(len(cmds), 15)
        self.assertNotIn("prismquant.oracle_search", " ".join(" ".join(c) for c in cmds))
        self.assertNotIn("--oracle", cmds[-1])


if __name__ == "__main__":
    unittest.main()
