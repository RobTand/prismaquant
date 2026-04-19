import json
import os
import tempfile
import unittest

from prismquant.bakeoff import (
    _decision,
    _load_calibration_point,
    _load_oracle_best,
    _load_refined_point,
    _summarize,
)


class TestBakeoff(unittest.TestCase):
    def test_decision_keep_on_meaningful_improvement(self):
        summary = {
            "delta_kl_vs_baseline": -2e-3,
            "oracle_gap_abs": 1e-3,
        }
        decision, _reason = _decision(summary, max_kl_regression=1e-3, min_kl_gain=1e-3, max_oracle_gap=5e-3)
        self.assertEqual(decision, "keep")

    def test_decision_reject_on_regression(self):
        summary = {
            "delta_kl_vs_baseline": 2e-3,
        }
        decision, _reason = _decision(summary, max_kl_regression=1e-3, min_kl_gain=1e-3, max_oracle_gap=None)
        self.assertEqual(decision, "reject")

    def test_decision_investigate_when_not_close_enough_to_oracle(self):
        summary = {
            "delta_kl_vs_baseline": -2e-3,
            "oracle_gap_abs": 1e-2,
        }
        decision, _reason = _decision(summary, max_kl_regression=1e-3, min_kl_gain=1e-3, max_oracle_gap=5e-3)
        self.assertEqual(decision, "investigate")

    def test_loaders_and_summary(self):
        calibration = {
            "results": [
                {"achieved_bits": 4.5, "actual_last_token_kl": 0.10},
                {"achieved_bits": 4.8, "actual_last_token_kl": 0.08},
                {"achieved_bits": 16.0, "actual_last_token_kl": 0.0},
            ]
        }
        refined = {"bits_per_param": 4.9, "refined_delta_kl_estimate": -0.01}
        oracle = {"best": {"bits_per_param": 4.9, "actual_last_token_kl": 0.06}}
        with tempfile.TemporaryDirectory() as td:
            c = os.path.join(td, "cal.json")
            r = os.path.join(td, "ref.json")
            o = os.path.join(td, "oracle.json")
            for path, payload in ((c, calibration), (r, refined), (o, oracle)):
                with open(path, "w") as f:
                    json.dump(payload, f)

            baseline = _load_calibration_point(c, "baseline")
            knee = _load_calibration_point(c, "knee")
            refined_point = _load_refined_point(r, knee.kl)
            oracle_point = _load_oracle_best(o)
            summary = _summarize(refined_point, baseline, oracle_point)

            self.assertEqual(baseline.kl, 0.10)
            self.assertEqual(knee.kl, 0.08)
            self.assertAlmostEqual(refined_point.kl, 0.07)
            self.assertAlmostEqual(oracle_point.kl, 0.06)
            self.assertAlmostEqual(summary["oracle_gap_abs"], 0.01)
            self.assertAlmostEqual(summary["oracle_gap_signed"], 0.01)

    def test_load_refined_point_prefers_calibrated_estimate(self):
        refined = {
            "bits_per_param": 4.9,
            "refined_delta_kl_estimate": -0.01,
            "calibrated_last_token_kl_estimate": 0.055,
        }
        with tempfile.TemporaryDirectory() as td:
            r = os.path.join(td, "ref.json")
            with open(r, "w") as f:
                json.dump(refined, f)
            refined_point = _load_refined_point(r, calibrated_kl=0.08)
            self.assertAlmostEqual(refined_point.kl, 0.055)


if __name__ == "__main__":
    unittest.main()
