"""Driver boundary checks only; these cannot establish W2 qualification."""
import contextlib
import io
import json
import sys
from pathlib import Path
import tempfile
import unittest

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from prismaquant.digests import DIRECT_ASCII_STRICT, bytes_sha256hex
from experiments.qualified_w2_replay_2905 import (
    HandoffError, _read, _qualification, _measured_only,
    _measured_overhead, _layer_configs, _scientific_rows, _row_provenance, main, run,
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

    def test_digest_owner_preserves_legacy_canonical_wire_bytes(self):
        value = {"label": "µ", "flag": True, "cells": [1, -0.0]}
        wire = b'{"cells":[1,-0.0],"flag":true,"label":"\\u00b5"}'
        self.assertEqual(DIRECT_ASCII_STRICT.encoded(value), wire)
        self.assertEqual(DIRECT_ASCII_STRICT.sha256(value), bytes_sha256hex(wire))

    def test_reject_duplicate_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            for text in ('{"schema": 1, "schema": 2}', '{"value": NaN}',
                         '{"value": Infinity}', '{"value": 1e999}'):
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
            "replay_config_sha256": DIRECT_ASCII_STRICT.sha256({"targets": [2.9]}),
            "artifact_sha256": {k: v for k, v in self.hashes.items()
                                if k not in ("inputs", "qualification")},
            "identities": {k: self.hashes[k + "_identity"] for k in
                           ("source", "probe", "calibration", "teacher")},
            "cells": [
                {"unit": "unit", "format": "measured", "status": "qualified",
                 "row_sha256": DIRECT_ASCII_STRICT.sha256(self.costs["unit"]["measured"])},
                {"unit": "unit", "format": "refused", "status": "refused",
                 "reason": "synthetic refusal"}],
        }
        self.roster["identities"]["scope"] = DIRECT_ASCII_STRICT.sha256(["unit"])

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


class EffectiveScientificRowsTests(unittest.TestCase):
    """Canonical fixture + real production build; no serving/W2 claim is made."""

    def setUp(self):
        # Reuse the repository's canonical fixtures and its explicit test-only
        # producer. Keep their sibling imports local to this test setup.
        tests = Path(__file__).resolve().parents[1] / "tests"
        with patch.object(sys, "path", [str(tests), *sys.path]):
            import canonical_quality_fixtures as fixtures
            import test_canonical_quality_quantity as canonical
        import pytest
        self.fixtures = fixtures
        self.canonical = canonical
        monkeypatch = pytest.MonkeyPatch()
        self.addCleanup(monkeypatch.undo)
        with patch.object(sys, "path", [str(tests), *sys.path]):
            self.owner = canonical.owner_for_rows(monkeypatch)
        from prismaquant import format_registry as fr
        self.F = [f"{fixtures.FAMILY}_R{rate}" for rate in (768, 896, 1024)]
        self.formats = [fr.get_format(fmt) for fmt in self.F]
        from prismaquant.tessera_formats import parse_tessera_format_name
        self.parsed = {spec.name: parse_tessera_format_name(spec.name) for spec in self.formats}
        self.stats = {"u": {"n_params": 8192, "in_features": 128,
                            "out_features": 64, "h_trace": 1.0, "unit_structure": "dense"}}
        rows = canonical.anchor_rows(currency="served_kl")
        for row in rows.values():
            row["cost_currency"] = "served_kl"
        rows[self.F[1]] = self.owner.chord_cost(self.F[1], unit="u", costs=rows)
        self.costs = {"u": rows}
        scope = rows[self.F[0]]["quality_scope"]
        probe = deepcopy(scope["probe_identity"])
        self.currency = rows[self.F[0]]["cost_currency"]
        from prismaquant.rung_allowability import qualified_cost_scope
        scientific = qualified_cost_scope(rows[self.F[0]], family=fixtures.FAMILY,
                                           unit="u", format_name=self.F[0])
        self.identities = {
            "probe": {key: deepcopy(scientific[key]) for key in (
                "probe_identity", "probe_identity_sha256", "probe_ids")},
            "teacher": deepcopy(probe["source_model"]),
            "calibration": {"calibration_sha256": probe["calibration_sha256"],
                            **deepcopy(scientific["window"])},
            "source": {"source_weights": {"u": deepcopy(scope["source_weight"])}},
        }

    def scientific(self, identities=None, costs=None, stats=None):
        _scientific_rows(self.stats if stats is None else stats,
                         self.costs if costs is None else costs, self.parsed,
                         self.currency, self.identities if identities is None else identities)

    def test_actual_scientific_documents_required(self):
        self.scientific()
        for label in self.identities:
            identities = deepcopy(self.identities)
            identities[label] = {"label": "a separately hash-bound but unrelated document"}
            with self.subTest(label=label), self.assertRaises((ValueError, KeyError, TypeError)):
                self.scientific(identities)

    def test_rehashed_handoff_coordinates_cannot_disagree_with_rows(self):
        # These are valid self-consistent documents. File/roster rehashing cannot
        # make their different scientific coordinates match the effective rows.
        from prismaquant.cost_stage_checkpoint import canonical_json_sha256
        from prismaquant.joint_aura_rows import joint_identity_sha256

        def other_teacher(doc):
            doc["config"]["fixture"] = "other teacher"
            doc["content_sha256"] = canonical_json_sha256(
                {key: doc[key] for key in ("config", "weight_map", "shards")},
                where="boundary fixture teacher content")

        def other_probe(doc):
            doc["probe_identity"]["seed_base"] = 8000
            doc["probe_identity_sha256"] = joint_identity_sha256(doc["probe_identity"])

        changes = {
            "calibration": lambda doc: doc.update(calibration_sha256="b" * 64),
            "teacher": other_teacher,
            "probe": other_probe,
            "source": lambda doc: doc["source_weights"]["u"].update(content_sha256="e" * 64),
        }
        for label, change in changes.items():
            identities = deepcopy(self.identities)
            change(identities[label])
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.scientific(identities)

    def test_units_share_one_production_scientific_scope(self):
        stats = {**self.stats, "other": deepcopy(self.stats["u"])}
        # Other unit may have its own source weight, but not another probe wall,
        # calibration/teacher/reference/window. Each row is individually valid.
        other = deepcopy(self.costs["u"][self.F[0]])
        other["quality_scope"]["unit"] = "other"
        other["quality_scope"]["source_weight"]["content_sha256"] = "a" * 64
        identities = deepcopy(self.identities)
        identities["source"]["source_weights"]["other"] = other["quality_scope"]["source_weight"]
        costs = {**self.costs, "other": {self.F[0]: other}}
        self.scientific(identities, costs, stats)
        from prismaquant.joint_aura_rows import joint_identity_sha256
        for field, value in (("calibration_sha256", "b" * 64),
                             ("token_scope", "another-window"),
                             ("temperature", 2.0), ("baseline", "another-reference")):
            changed = deepcopy(costs)
            row = changed["other"][self.F[0]]
            row["quality_scope"]["probe_identity"][field] = value
            row["quality_scope"]["probe_identity_sha256"] = joint_identity_sha256(
                row["quality_scope"]["probe_identity"])
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.scientific(identities, changed, stats)
        for field in ("objective", "probe_ids"):
            changed = deepcopy(costs)
            scope = changed["other"][self.F[0]]["quality_scope"]
            if field == "objective":
                scope["objective"]["objective"] = "another-reference"
                scope["probe_identity"]["objective"] = "another-reference"
                scope["probe_identity_sha256"] = joint_identity_sha256(scope["probe_identity"])
            else:
                scope["probe_ids"] = [99]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.scientific(identities, changed, stats)

    def test_later_units_align_activation_contracts_within_each_unit(self):
        self.driver_environment()
        from prismaquant import allocator_candidates as ac
        stats = {**self.stats, "v": deepcopy(self.stats["u"])}
        # Production may retain both independently valid measured cells even
        # when no chord can be made between them. Global same_unit=False
        # comparisons must not hide a within-v activation-contract mismatch.
        other = {fmt: deepcopy(self.costs["u"][fmt]) for fmt in (self.F[0], self.F[2])}
        for row in other.values():
            row["quality_scope"]["unit"] = "v"
            row["quality_scope"]["activation_contract"] = {"fixture": "v activation"}
        identities = deepcopy(self.identities)
        identities["source"]["source_weights"]["v"] = deepcopy(
            other[self.F[0]]["quality_scope"]["source_weight"])
        costs = {**self.costs, "v": other}
        # Different units need not have identical activation contracts.
        self.scientific(identities, costs, stats)
        other[self.F[2]]["quality_scope"]["activation_contract"] = {"fixture": "different activation"}
        built = ac.build_candidates(
            stats, deepcopy(costs), [self.formats[0], self.formats[2]], target_profile="research",
            rung_allowability={self.fixtures.FAMILY: self.owner}, allowability_m=8)
        self.assertEqual({c.fmt for c in built.candidates["v"]}, set(other))
        for stage, table in (("supplied", costs), ("effective", built.cost_rows)):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "different activation_contract"):
                self.scientific(identities, table, stats)

    def test_denominator_requires_explicit_integral_mutable_geometry(self):
        for key, values in {
            "n_params": (0, -1, True, 8192.0, "8192"),
            "out_features": (None, 0, -1, True, 64.5, "64"),
            "in_features": (None, 0, -1, True, 128.5, "128"),
            "num_experts": (None, -1, True, 1.5, "1"),
        }.items():
            for value in values:
                stats = deepcopy(self.stats)
                stats["u"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(HandoffError):
                    self.scientific(stats=stats)
        for key in ("n_params", "out_features", "in_features"):
            stats = deepcopy(self.stats)
            del stats["u"][key]
            with self.subTest(missing=key), self.assertRaises(HandoffError):
                self.scientific(stats=stats)
        stats = deepcopy(self.stats)
        stats["u"]["num_experts"] = 0  # Production's dense sentinel.
        self.scientific(stats=stats)

    def test_expert_axis_is_part_of_production_parameter_geometry(self):
        stats = deepcopy(self.stats)
        stats["u"]["num_experts"] = 4
        with self.assertRaisesRegex(HandoffError, "n_params.*32768"):
            self.scientific(stats=stats)
        stats["u"]["n_params"] = 32768
        # A matching count does not make a 3D population match qualified 2D
        # source geometry, nor extend the current owner's qualification scope.
        with self.assertRaisesRegex(HandoffError, "source shape differs"):
            self.scientific(stats=stats)

    def test_price_source_shape_must_match_actual_mutable_stats(self):
        stats = deepcopy(self.stats)
        stats["u"]["out_features"] *= 2
        stats["u"]["n_params"] *= 2
        with self.assertRaisesRegex(HandoffError, "source shape differs"):
            self.scientific(stats=stats)

    def test_effective_row_provenance_cannot_relabel_a_measured_replacement(self):
        effective = deepcopy(self.costs)
        effective["u"][self.F[0]]["production_annotation"] = "different row"
        self.scientific(costs=effective)
        with self.assertRaisesRegex(HandoffError, "replaced a non-chord"):
            _row_provenance(self.costs, effective)

    def test_effective_chord_anchors_are_bound_to_supplied_roster_rows(self):
        effective = deepcopy(self.costs)
        anchor = effective["u"][self.F[1]]["canonical_anchors"][0]
        anchor["row"] = deepcopy(anchor["row"])
        anchor["row"]["annotation"] = "not supplied"
        self.scientific(costs=effective)
        with self.assertRaisesRegex(HandoffError, "anchor differs from supplied roster"):
            _row_provenance(self.costs, effective)

    def write_handoff(self, root):
        """Build a fully hash-bound, synthetic boundary handoff, not W2 evidence."""
        stats = deepcopy(self.stats)
        values = {"stats": stats, "costs": self.costs, "model_config": {},
                  "source_manifest": {"bytes": {"u": 16384},
                                      "spans": {"u": ["u"]}, "total_bytes": 16384},
                  "kernel_builds": {self.fixtures.FAMILY: "boundary-fixture"},
                  **{k + "_identity": v for k, v in self.identities.items()},
                  "measured_overhead": {"schema": "prismaquant.measured_overhead.v1",
                               "components": {"fixture": 0},
                               "measurement_reference": "synthetic boundary fixture"}}
        config = {
            "schema": "prismaquant.o5_explicit_w2_inputs.v2",
            "formats": self.F, "currency": self.currency, "profile": "default",
            "allowability_root": ".",
            "serving_context": {"platform": "sm_120",
                                "runtime_image": "example/vllm@sha256:" + "a" * 64,
                                "execution_mode": "eager", "residency": "resident"},
            "target_profile": None, "tensor_parallel": 1,
            "targets": [3.5, 4.5, 6.0], "bit_precision": .00001,
            "overhead_reserve_bytes": 0, "measured_only_units": [],
        }
        for label, value in values.items():
            raw = json.dumps(value).encode()
            path = root / (label + ".json")
            path.write_bytes(raw)
            config[label] = {"path": path.name, "sha256": bytes_sha256hex(raw)}
        hashes = {k: config[k]["sha256"] for k in values}
        roster = {
            "schema": "prismaquant.w2_qualification_roster.v1",
            "artifact_sha256": hashes,
            "replay_config_sha256": DIRECT_ASCII_STRICT.sha256(config),
            "identities": {k: hashes[k + "_identity"] for k in self.identities},
            "cells": [{"unit": "u", "format": spec.name, "status": "qualified",
                       "row_sha256": DIRECT_ASCII_STRICT.sha256(self.costs["u"][spec.name])}
                      for spec in self.formats],
        }
        roster["identities"]["scope"] = DIRECT_ASCII_STRICT.sha256(["u"])
        raw = json.dumps(roster).encode()
        (root / "qualification.json").write_bytes(raw)
        config["qualification"] = {"path": "qualification.json", "sha256": bytes_sha256hex(raw)}
        path = root / "inputs.json"
        path.write_text(json.dumps(config))
        return path

    def driver_environment(self, *, real_replay=False):
        from prismaquant import allocator_candidates as ac, format_registry as fr
        from prismaquant import interpolation_regret as replay, rung_allowability as ra
        from prismaquant import model_profiles as mp, tessera_menu as tm, tessera_serving_scope as scope
        from prismaquant.lane_eligibility import ServingContext
        context = ServingContext(platform="sm_120", structure="dense", residency="resident",
                                 runtime_image="example/vllm@sha256:" + "a" * 64,
                                 execution_mode="eager")
        # The serving contract is a boundary stub. The production row builder
        # runs under the same emulation seam as the canonical fixture's tests;
        # never pass this stub off as an admission/reader contract to production.
        contract = SimpleNamespace(allowable_rungs={self.fixtures.FAMILY: None},
                                   fused_module=None, identity=lambda: {"fixture": True})
        self.enterContext(patch.object(tm, "tessera_runtime_contract", return_value=contract))
        # Use the already-synthesized canonical fixture specs at the driver seam.
        real_get_format = fr.get_format
        formats = {spec.name: spec for spec in self.formats}
        self.enterContext(patch.object(fr, "get_format", side_effect=lambda name: (
            formats[name] if name in formats else real_get_format(name))))
        real_build = ac.build_candidates

        def canonical_build(*args, **kwargs):
            with patch.object(tm, "tessera_runtime_contract", return_value=None):
                return real_build(*args, **kwargs)

        self.enterContext(patch.object(ac, "build_candidates", side_effect=canonical_build))
        # Native serving admission is not under test and the fixture has no
        # release contract. Preserve the canonical fixture's explicit admit stub.
        self.enterContext(patch.object(ac, "candidate_rung_admission", return_value=SimpleNamespace(
            requires_serving_context=False, admits=lambda _mode: True)))
        self.enterContext(patch.object(ra, "load_rung_allowability", return_value=self.owner))
        self.enterContext(patch.object(mp, "profile_from_config", return_value=mp.DefaultProfile()))
        self.enterContext(patch.object(scope, "context_by_unit_from_stats", return_value={"u": context}))
        if not real_replay:
            return self.enterContext(patch.object(replay, "replay_three_arms", return_value={"targets": []}))

    def test_hash_consistent_inflated_denominator_refuses_before_build_or_replay(self):
        replay = self.driver_environment()
        from prismaquant import allocator_candidates as ac
        self.stats["u"]["n_params"] *= 100
        # Regenerate every binding: a self-consistent hash roster is not a
        # certificate that 64x128 has 819200 mutable parameters.
        with tempfile.TemporaryDirectory() as tmp, patch.object(ac, "build_candidates") as build:
            with self.assertRaisesRegex(HandoffError, "u.*n_params.*819200.*8192"):
                run(self.write_handoff(Path(tmp)))
        build.assert_not_called()
        replay.assert_not_called()

    def test_stdout_json_uses_real_production_build_and_replay(self):
        from contextlib import redirect_stderr, redirect_stdout
        import io
        self.driver_environment(real_replay=True)
        for output_file in (False, True):
            with self.subTest(output_file=output_file), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest = self.write_handoff(root)
                output = root / "result.json"
                args = ["--inputs", str(manifest)] + (["--output", str(output)] if output_file else [])
                stdout, stderr = io.StringIO(), io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    main(args)
                result = json.loads(stdout.getvalue())
                if output_file:
                    self.assertEqual(result, json.loads(output.read_text()))
                report = result["replay"]
                self.assertEqual(report["frozen_mutable_n_params"], 8192)
                self.assertEqual([t["target_bpp"] for t in report["targets"]], ["3.5", "4.5", "6.0"])
                self.assertEqual([t["target_bytes"] for t in report["targets"]], [3584, 4608, 6144])
                self.assertEqual(report["status"], "surrogate_only_pending_w4")
                self.assertIn("[alloc] cost-source usage:", stderr.getvalue())

    def test_driver_replays_immutable_effective_production_rows_and_records_both_stages(self):
        # Production ignores these supplied annotations when rebuilding R3.5.
        # A supplied-table replay used to hide that the row and hash had changed.
        self.costs["u"][self.F[1]]["supplied_annotation"] = "not effective"
        original = deepcopy(self.costs)
        replay = self.driver_environment()
        from prismaquant import allocator_candidates as ac
        real_build = ac.build_candidates
        built_tables = []

        def build(*args, **kwargs):
            built = real_build(*args, **kwargs)
            built_tables.append(built.cost_rows)
            return built

        with tempfile.TemporaryDirectory() as tmp, patch.object(ac, "build_candidates", side_effect=build):
            result = run(self.write_handoff(Path(tmp)))
        effective = replay.call_args.kwargs["costs"]
        self.assertNotIn("supplied_annotation", effective["u"][self.F[1]])
        self.assertEqual(effective, built_tables[0])
        self.assertIsNot(effective, built_tables[0])
        self.assertEqual(self.costs, original)
        provenance = result["effective_row_provenance"]
        self.assertEqual(provenance["effective_costs_sha256"], DIRECT_ASCII_STRICT.sha256(effective))
        self.assertEqual(provenance["supplied_costs_sha256"], DIRECT_ASCII_STRICT.sha256(original))
        cells = {cell["format"]: cell for cell in provenance["cells"]}
        chord = cells[self.F[1]]
        self.assertTrue(chord["replaced"])
        self.assertNotEqual(chord["supplied_row_sha256"], chord["effective_row_sha256"])
        self.assertEqual(chord["effective_origin"], "production_canonical_chord")
        self.assertEqual({a["format"] for a in chord["anchors"]}, {self.F[0], self.F[2]})
        for anchor in chord["anchors"]:
            self.assertEqual(anchor["supplied_row_sha256"], cells[anchor["format"]]["supplied_row_sha256"])
        # Evidence is from immutable snapshots, not the replay's mutable copy.
        replay.call_args.kwargs["costs"]["u"][self.F[1]]["predicted_dloss"] = 999
        self.assertEqual(chord["effective_row_sha256"],
                         DIRECT_ASCII_STRICT.sha256(built_tables[0]["u"][self.F[1]]))

    def test_hash_bound_unrelated_identity_documents_still_refuse(self):
        replay = self.driver_environment()
        original = deepcopy(self.identities)
        for label in original:
            self.identities = deepcopy(original)
            self.identities[label] = {"opaque": "not a scientific coordinate"}
            # write_handoff rehashes the files, replay config and qualification
            # roster after the document change. Byte binding alone must not pass.
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(HandoffError):
                    run(self.write_handoff(Path(tmp)))
        replay.assert_not_called()

    def test_production_admission_cannot_drop_a_supplied_cell(self):
        replay = self.driver_environment()
        from prismaquant import allocator_candidates as ac
        real_build = ac.build_candidates

        def build(*args, **kwargs):
            built = real_build(*args, **kwargs)
            built.candidates["u"].pop()
            return built

        with tempfile.TemporaryDirectory() as tmp, patch.object(ac, "build_candidates", side_effect=build) as producer:
            with self.assertRaisesRegex(HandoffError, "admission changed the supplied grid"):
                run(self.write_handoff(Path(tmp)))
        producer.assert_called_once()
        replay.assert_not_called()

    def test_effective_production_scope_drift_refuses_before_replay(self):
        replay = self.driver_environment()
        from prismaquant import allocator_candidates as ac
        real_build = ac.build_candidates

        def build(*args, **kwargs):
            built = real_build(*args, **kwargs)
            built.cost_rows["u"][self.F[0]]["quality_scope"]["teacher"] = "e" * 64
            return built

        with tempfile.TemporaryDirectory() as tmp, patch.object(ac, "build_candidates", side_effect=build) as producer:
            with self.assertRaisesRegex(ValueError, "teacher|probe|identity"):
                run(self.write_handoff(Path(tmp)))
        producer.assert_called_once()
        replay.assert_not_called()


if __name__ == "__main__":
    unittest.main()
