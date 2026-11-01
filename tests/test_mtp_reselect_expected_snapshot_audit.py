"""Fixed-MTP CLI authenticates the bytes it actually consumes, not an old path read.

The bound M3/M4 inputs are existing synthetic metadata fixtures. Selection,
scope/budget admission, receipt joins and publication remain real; no model,
runtime, cached-wire decoding or serving qualification is established here.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sys
from functools import partial
from pathlib import Path

import pytest

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import selection_serving_lane_provenance
from prismaquant.footprint import whole_artifact_budget_stamp
from prismaquant.glm_mtp_selection import merge_mtp_costs
from prismaquant.model_profiles import detect_profile
from prismaquant.tessera_serving_scope import (
    ServingTarget, scope_provenance, unit_structure_from_profile,
)
from tools import reselect_mtp_fixed as cli


def _case(tmp_path, monkeypatch):
    import test_glm_mtp_priced_wires_1413 as prices
    import test_tessera_expert_projection as projection
    from test_glm_mtp_selection import CONSTANTS

    # Adapt only synthetic factory names to the declared MTP layer/profile.
    # The existing factory otherwise names layer 2 under an MTP-layer-45 probe.
    stack = "model.layers.45.feed_forward.experts"
    with monkeypatch.context() as names:
        names.setattr(projection, "STACK", stack)
        names.setattr(prices, "STACK", stack)
        names.setattr(prices, "DENSE", "model.layers.45.feed_forward.shared_experts.w1")
        names.setattr(prices, "_projection", partial(projection._projection, stacks=(stack,)))
        names.setattr(prices, "_declared", partial(projection._declared, stacks=(stack,)))
        parts, _units, _carried = prices._bound_parts_for_rates(tmp_path, (1024, 896))
    payload = merge_mtp_costs([part for part, _ in parts],
                            sources=[source for _, source in parts])
    cost = tmp_path / "merged.pkl"
    original = pickle.dumps(payload, protocol=4)
    alternate = pickle.dumps(payload, protocol=5)
    assert original != alternate
    assert pickle.loads(original) == pickle.loads(alternate)
    cost.write_bytes(original)
    model = tmp_path / "source"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "lfm2_moe"}))
    profile = detect_profile(str(model))
    target = ServingTarget("sm121", "audit.invalid/vllm@sha256:" + "a" * 64,
                           "eager", "resident")
    assignment = {"model.layers.0.feed_forward.w1": "BF16"}
    contexts = {name: target.context(unit_structure_from_profile(name, profile))
                for name in assignment}
    body = {name: fr.get_format(fmt).autoround_config()
            for name, fmt in assignment.items()}
    body["__prismaquant__"] = {
        "target_profile": "vllm_packed_moe",
        "tessera_serving_scope": scope_provenance(target, contexts),
        "serving_lane_provenance": selection_serving_lane_provenance(
            assignment, target_profile="vllm_packed_moe", context_by_unit=contexts),
        "assignment_payload_bits_total": 8 * 1_000_000,
        "whole_artifact_budget": whole_artifact_budget_stamp(
            budget_bytes=2_000_000, selection_tensor_payload_bytes=1_000_000,
            selection_non_tensor_reserve_bytes=64, selection_assignment=assignment),
    }
    body_path = tmp_path / "body.json"
    body_raw = json.dumps(body).encode()
    body_path.write_bytes(body_raw)
    export = tmp_path / "body-export"
    export.mkdir()
    (export / "observed.bin").write_bytes(b"x" * 1_000_064)
    constants = tmp_path / "constants.json"
    constants.write_text(json.dumps(CONSTANTS))
    fixed = tmp_path / "fixed.json"
    fixed.write_text(json.dumps({group: "BF16" for group in payload["groups"]}))
    output = tmp_path / "selected.json"
    argv = ["reselect_mtp_fixed.py",
            "--body-layer-config", str(body_path), "--expect-body-sha256",
            hashlib.sha256(body_raw).hexdigest(), "--mtp-joint-cost", str(cost),
            "--expect-cost-sha256", hashlib.sha256(original).hexdigest(),
            "--mtp-byte-budget", "1000000", "--whole-artifact-budget-bytes", "2000000",
            "--observed-body-export", str(export), "--new-metadata-allowance-bytes", "64",
            "--mtp-serve-constants", str(constants), "--fixed-formats", str(fixed),
            "--model", str(model), "--target-profile", "vllm_packed_moe",
            "--output", str(output), "--tessera-platform", target.platform,
            "--tessera-runtime-image", target.runtime_image,
            "--tessera-execution-mode", target.execution_mode,
            "--tessera-residency", target.residency]
    monkeypatch.setattr(sys, "argv", argv)
    return cost, original, alternate, output


def _replace_after_checked_read(monkeypatch, cost, replacement):
    read = Path.read_bytes
    seen = []

    def read_and_replace(path):
        raw = read(path)
        if path == cost:
            seen.append(hashlib.sha256(raw).hexdigest())
            if len(seen) == 1:
                staged = cost.with_suffix(".replacement")
                staged.write_bytes(replacement)
                os.replace(staged, cost)
        return raw

    monkeypatch.setattr(Path, "read_bytes", read_and_replace)
    return seen


def test_reselect_consumes_only_expected_cost_snapshot(tmp_path, monkeypatch, capsys):
    cost, original, alternate, output = _case(tmp_path, monkeypatch)
    seen = _replace_after_checked_read(monkeypatch, cost, alternate)
    try:
        result = cli.main()
    except (SystemExit, ValueError) as error:
        # A named refusal is also safe; a fixture/gate failure is not a pass.
        message = (capsys.readouterr().err if isinstance(error, SystemExit)
                   else str(error))
        assert re.search("MTP cost|cost snapshot|expected.*cost|cost.*expected", message)
        assert not output.exists()
    else:
        assert result == 0
        report = json.loads(capsys.readouterr().out.splitlines()[-1])
        selected = json.loads(output.read_bytes())["__prismaquant__"]["mtp_selection"]
        expected = hashlib.sha256(original).hexdigest()
        assert report["mtp_cost_sha256"] == expected
        assert selected["mtp_joint_cost_sha256"] == expected, (
            "reselect accepted different cost bytes after checking --expect-cost-sha256",
            seen, selected["mtp_joint_cost_sha256"], expected)
    assert seen[0] == hashlib.sha256(original).hexdigest()


@pytest.mark.parametrize("replace_same_bytes", [False, True])
def test_reselect_preserves_matching_cost_receipt(tmp_path, monkeypatch, capsys,
                                                replace_same_bytes):
    cost, original, _alternate, output = _case(tmp_path, monkeypatch)
    if replace_same_bytes:
        _replace_after_checked_read(monkeypatch, cost, original)
    assert cli.main() == 0
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    selected = json.loads(output.read_bytes())["__prismaquant__"]["mtp_selection"]
    expected = hashlib.sha256(original).hexdigest()
    assert report["mtp_cost_sha256"] == expected
    assert selected["mtp_joint_cost_sha256"] == expected
    assert selected["mtp_expert_wires"] == {}
    assert set(selected["rung_by_group"].values()) == {"BF16"}


@pytest.mark.parametrize("suffix", [".json", ".pkl"])
def test_cost_input_binds_one_read_and_preserves_supplied_path(tmp_path, suffix):
    from prismaquant.glm_mtp_selection import SCHEMA, load_mtp_cost

    payload = {"schema": SCHEMA, "audit_input": "one owned input"}
    raw = (json.dumps(payload).encode() if suffix == ".json" else pickle.dumps(payload))
    directory = tmp_path / "spelling"
    directory.mkdir()
    path = tmp_path / ("cost" + suffix)
    path.write_bytes(raw)
    spelling = str(directory) + "/../" + path.name
    cost = load_mtp_cost(spelling, expected_sha256=hashlib.sha256(raw).hexdigest())
    path.write_bytes(b"different later bytes")
    assert cost.payload == payload
    assert cost.path == spelling
    assert cost.sha256 == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("suffix", [".json", ".pkl"])
def test_expected_cost_is_checked_before_any_parser(tmp_path, suffix):
    from prismaquant.glm_mtp_selection import load_mtp_cost

    path = tmp_path / ("invalid" + suffix)
    path.write_bytes(b"not valid JSON or pickle")
    with pytest.raises(ValueError, match="MTP cost differs from expected bytes"):
        load_mtp_cost(path, expected_sha256="0" * 64)
