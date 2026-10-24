"""Driver boundary checks only; these cannot establish W2 qualification."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from experiments.qualified_w2_replay_2905 import (
    HandoffError, _read, _sha, _qualification, _measured_only,
    _measured_overhead, _layer_configs, main, run,
)


class ExplicitInputTests(unittest.TestCase):
    def test_missing_cli_input_refuses(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            main([])
        self.assertEqual(cm.exception.code, 2)

    def test_missing_file_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                run(Path(tmp) / "not-published.json")

    def test_reject_duplicate_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            for text in ('{"schema": 1, "schema": 2}', '{"value": NaN}',
                         '{"value": Infinity}'):
                path.write_text(text)
                with self.assertRaises(HandoffError):
                    _read(path)

    def test_refuse_wrong_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_text('{}')
            with self.assertRaisesRegex(HandoffError, "schema"):
                run(path)

    def test_digest_checked_before_production_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stats.json").write_text('{}')
            path = root / "input.json"
            path.write_text(json.dumps({
                "schema": "prismaquant.o5_explicit_w2_inputs.v2",
                "stats": {"path": "stats.json", "sha256": "0" * 64},
            }))
            with self.assertRaisesRegex(HandoffError, "SHA-256 mismatch"):
                run(path)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        # Deliberately synthetic boundary fixture, NOT a W2 artifact.
        self.stats = {"unit": {}}
        self.names = ["measured", "refused"]
        self.costs = {"unit": {"measured": {"value": 1}}}
        self.hashes = {k + "_identity": str(i) * 64 for i, k in enumerate(
            ("source", "probe", "calibration", "teacher"), 1)}
        self.hashes.update(inputs="0" * 64, qualification="5" * 64)
        self.config = {"targets": [2.9], "qualification": {"sha256": "5" * 64}}
        self.roster = {
            "schema": "prismaquant.w2_qualification_roster.v1",
            "replay_config_sha256": _sha({"targets": [2.9]}),
            "artifact_sha256": {k: v for k, v in self.hashes.items()
                                if k not in ("inputs", "qualification")},
            "identities": {k: self.hashes[k + "_identity"] for k in
                           ("source", "probe", "calibration", "teacher")},
            "cells": [
                {"unit": "unit", "format": "measured", "status": "qualified",
                 "row_sha256": _sha(self.costs["unit"]["measured"])},
                {"unit": "unit", "format": "refused", "status": "refused",
                 "reason": "synthetic refusal"}],
        }
        self.roster["identities"]["scope"] = _sha(["unit"])

    def check(self, roster=None):
        return _qualification(self.roster if roster is None else roster, self.hashes,
                              self.stats, self.costs, self.names, self.config)

    def test_refusals_frozen(self):
        self.assertEqual(self.check(), frozenset({("unit", "refused")}))
        self.assertIsInstance(self.check(), frozenset)

    def test_roster_coverage_duplicate_status_and_reason(self):
        for cells in (self.roster["cells"][:1], self.roster["cells"] * 2,
                      [dict(self.roster["cells"][0], status="unknown"), self.roster["cells"][1]],
                      [self.roster["cells"][0], dict(self.roster["cells"][1], reason="")]):
            roster = dict(self.roster, cells=cells)
            with self.subTest(cells=cells), self.assertRaises(HandoffError):
                self.check(roster)

    def test_changed_row_or_resurrected_refusal(self):
        self.costs["unit"]["measured"]["value"] = 2
        with self.assertRaisesRegex(HandoffError, "changed"):
            self.check()
        self.costs["unit"]["measured"]["value"] = 1
        self.costs["unit"]["refused"] = {"value": 2}
        with self.assertRaisesRegex(HandoffError, "absent"):
            self.check()

    def test_bindings_and_identities_fail_closed(self):
        for key in ("artifact_sha256", "identities", "replay_config_sha256"):
            roster = deepcopy(self.roster)
            roster[key] = {} if key != "replay_config_sha256" else "f" * 64
            with self.subTest(key=key), self.assertRaises(HandoffError):
                self.check(roster)
        roster = deepcopy(self.roster)
        roster["identities"]["source"] = "f" * 64
        with self.assertRaisesRegex(HandoffError, "lineage"):
            self.check(roster)
        self.config["targets"] = [3.0]
        with self.assertRaisesRegex(HandoffError, "configuration"):
            self.check()


class AccountingAndExportTests(unittest.TestCase):
    def test_production_topology_cannot_be_overridden_by_empty_roster(self):
        from prismaquant.lane_eligibility import STRUCTURE_ROUTED_MOE
        contexts = {"expert": SimpleNamespace(structure=STRUCTURE_ROUTED_MOE),
                    "dense": SimpleNamespace(structure="dense")}
        self.assertEqual(_measured_only(contexts, []), frozenset({"expert"}))
        self.assertEqual(_measured_only(contexts, ["dense"]), frozenset(contexts))
        for bad in (["missing"], ["expert", "expert"], None, "expert"):
            with self.subTest(bad=bad), self.assertRaises(HandoffError):
                _measured_only(contexts, bad)

    def test_measured_overhead_is_not_reserve(self):
        payload = {"schema": "prismaquant.measured_overhead.v1",
                   "components": {"headers": 13, "scales": 8},
                   "measurement_reference": "synthetic-test"}
        self.assertEqual(_measured_overhead(payload), 21)
        for bad in (-1, True, 1.5, "2"):
            with self.subTest(bad=bad), self.assertRaises(HandoffError):
                _measured_overhead(dict(payload, components={"header": bad}))
        with self.assertRaises(HandoffError):
            _measured_overhead(dict(payload, measurement_reference=""))
        self.assertEqual(_measured_overhead(dict(payload, components={})), 0)

    def test_configs_use_existing_registry_and_skip_infeasible(self):
        from prismaquant import format_registry as fr
        report = {"targets": [{"target_bpp": 2.9, "arms": {
            "default": {"status": "ok", "assignment": {"layer": "BF16"}},
            "measured_only": {"status": "infeasible"}}}]}
        self.assertEqual(_layer_configs(report), {
            "2.9/default": {"layer": fr.get_format("BF16").autoround_config()}})

    def test_real_seam_reports_separate_overhead_and_generates_configs(self):
        # Synthetic prices only: exercise the production seam, never the W2 gate.
        from prismaquant import footprint as fp, format_registry as fr
        from prismaquant.allocator_solver import Candidate
        from prismaquant.interpolation_regret import replay_three_arms
        from prismaquant.model_profiles import DefaultProfile
        name = "model.layers.0.self_attn.o_proj"
        stats = {name: {"n_params": 65536, "in_features": 256,
                        "out_features": 256, "h_trace": 1.0}}
        menus, costs = [], {}
        for fmt, loss, measured in (("IQ2_XXS", 100., True), ("IQ4_XS", 8., False),
                                    ("Q6_K", 1., True)):
            size = fr.get_format(fmt).memory_bytes_for_shape((256, 256))
            menus.append(Candidate(fmt=fmt, bits_per_param=size * 8 / 65536,
                                   memory_bytes=size, predicted_dloss=loss,
                                   activation_pricing=("measured_output_mse" if measured
                                                       else "interpolated_output_mse")))
            costs[fmt] = {"output_mse": loss * 2, "output_mse_measured": measured,
                          "cost_source": "tessera_campaign_measured" if measured else "tessera_campaign_interpolated",
                          "cost_currency": "synthetic"}
        manifest = fp.SourceByteManifest({name: 131072}, spans={name: frozenset({name})})
        report = replay_three_arms(
            stats=stats, costs={name: costs}, formats=[fr.get_format(c.fmt) for c in menus],
            candidates={name: menus}, profile=DefaultProfile(), currency="synthetic",
            targets=("3.5", "4.5", "6.0"), bit_precision=0.00001,
            source_total_bytes=131072, source_manifest=manifest,
            overhead_reserve_bytes=23, measured_overhead_bytes=17,
            refused_cells=frozenset(), measured_only_units=frozenset())
        self.assertEqual(report["measured_overhead_bytes"], 17)
        self.assertEqual(report["overhead_reserve_bytes"], 23)
        for target in report["targets"]:
            for arm in target["arms"].values():
                if arm["status"] == "ok":
                    self.assertEqual(arm["achieved_bytes"],
                                     arm["accounting"]["artifact_payload_bytes"] + 40)
        self.assertTrue(_layer_configs(report))

    def test_output_exclusive_and_matches_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            with patch("experiments.qualified_w2_replay_2905.run", return_value={"test": True}):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    main(["--inputs", "not-a-campaign", "--output", str(path)])
                self.assertEqual(path.read_text(), stdout.getvalue())
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(["--inputs", "not-a-campaign", "--output", str(path)])


if __name__ == "__main__":
    unittest.main()
