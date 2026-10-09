"""Option B enumerates MTP choices with a body each (PQ #2532).

CPU fixtures only: enumeration, byte accounting and record identities carry
no serving qualification.
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from prismaquant import allocator, footprint as fp, format_registry as fr
from test_glm_mtp_selection import CONSTANTS, PARAMS, _probe, _row

MTP_UNITS = ("model.layers.45.mlp.experts.0.gate_proj",
             "model.layers.45.mlp.experts.1.gate_proj",
             "model.layers.45.mlp.shared_experts.gate_proj")
BODY = "model.layers.0.self_attn.o_proj"
BODY_FORMATS = ("NVFP4", "FP8_E4M3", "BF16")


def _tiny_case(tmp_path):
    """One body Linear plus a 2-group MTP menu under a declared MoE profile.

    The profile matters: the MTP scope owner refuses a unit topology it
    cannot classify, so the fixture declares the Qwen3 MoE grammar the same
    way the card-rebase fixture does. One body unit keeps the exhaustive
    oracle exact: every probe can only land on one of three formats, and the
    ratchet ships min body dloss among the fitting ones.
    """
    from test_allocator_byte_budget_selection import _FLOOR_TENSORS
    from test_footprint import _write_safetensors

    probe = _probe()
    wire_bytes = fr.get_format("FP8_E4M3").memory_bytes_for_shape((64, 128))
    payload = {
        "schema": "prismaquant.glm_mtp_cost.v1", "mtp_layer": 45,
        "groups": {"pair": list(MTP_UNITS[:2]), "single": [MTP_UNITS[2]]},
        "params": {name: PARAMS for name in MTP_UNITS},
        "source_dtype": {name: "bfloat16" for name in MTP_UNITS},
        "costs": {name: {"FP8_E4M3": _row(name, "FP8_E4M3", [0.02] * 4, probe)}
                  for name in MTP_UNITS},
        "wire_bytes": {name: {"FP8_E4M3": wire_bytes} for name in MTP_UNITS},
    }
    model = tmp_path / "model"
    model.mkdir()
    tensors = dict(_FLOOR_TENSORS)
    tensors[f"{BODY}.weight"] = ("BF16", (256, 256))
    for name in MTP_UNITS:
        tensors[f"{name}.weight"] = ("BF16", (64, 128))
    _write_safetensors(model / "model-00001.safetensors", tensors)
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"]}))
    stats = {BODY: {"h_trace": 1.0, "n_params": 256 * 256,
                     "in_features": 256, "out_features": 256}}
    probe_path = tmp_path / "probe.pkl"
    cost_path = tmp_path / "cost.pkl"
    probe_path.write_bytes(pickle.dumps({"stats": stats, "meta": {"model": str(model)}}))
    cost_path.write_bytes(pickle.dumps({
        "costs": {BODY: {
            "NVFP4": {"weight_mse": 1.0, "output_mse": 1.0,
                      "output_mse_measured": True, "predicted_dloss": 1.0},
            "FP8_E4M3": {"weight_mse": 0.1, "output_mse": 0.1,
                         "output_mse_measured": True, "predicted_dloss": 0.1}}},
        "meta": {"formats": ["NVFP4", "FP8_E4M3"]}}))
    mtp_path = tmp_path / "mtp.pkl"
    mtp_path.write_bytes(pickle.dumps(payload))
    constants = tmp_path / "constants.json"
    constants.write_text(json.dumps(CONSTANTS))
    return model, probe_path, cost_path, stats, payload, mtp_path, constants


def _floor_bytes(model, stats):
    """Immutable bytes plus every source span the body does not re-encode."""
    return int(fp.floor_bytes_for_model(str(model), [BODY], stats)["floor_bytes"])


def _body_bytes(assignment):
    return sum(fr.get_format(fmt).memory_bytes_for_shape((256, 256))
               + (fp.nvfp4_global_sidecar_bytes(name, (256, 256)) if fmt == "NVFP4" else 0)
               for name, fmt in assignment.items())


def _mtp_combo_bytes(payload):
    """Selected bytes of the four declared group choices, cheapest first."""
    wire = payload["wire_bytes"]
    pair = sorted({4 * PARAMS, 2 * wire[MTP_UNITS[0]]["FP8_E4M3"]})
    single = sorted({2 * PARAMS, wire[MTP_UNITS[2]]["FP8_E4M3"]})
    return sorted(a + b for a in pair for b in single)


def _argv(case, output, cap):
    root = output.parent
    return ["--probe", str(case[1]), "--costs", str(case[2]),
            "--formats", ",".join(BODY_FORMATS), "--target-profile", "research",
            "--allow-default-profile", "--target-disk-gb", repr(cap / fp.GB),
            "--artifact-overhead-reserve-bytes", "512",
            "--layer-config", str(root / "main-layer.json"),
            "--pareto-csv", str(root / "main-pareto.csv"),
            "--mtp-joint-cost", str(case[5]), "--mtp-byte-budget", "1000000",
            "--mtp-serve-constants", str(case[6]), "--mtp-formats", "BF16,FP8_E4M3",
            "--mtp-option-b-dir", str(output)]


def test_option_b_matches_exhaustive_body_oracle(tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "format_is_producer_eligible", lambda name, **_: True)
    case = _tiny_case(tmp_path)
    payload = case[4]
    floor = _floor_bytes(case[0], case[3])
    mtp_source = 3 * 2 * PARAMS
    fp8_body = _body_bytes({BODY: "FP8_E4M3"})
    combos = _mtp_combo_bytes(payload)
    # The cheapest MTP choice affords FP8; every dearer choice drops the body
    # to NVFP4, so the body differs across the remainder.
    cap = floor - mtp_source + 512 + fp8_body + (combos[0] + combos[1]) // 2
    output = tmp_path / "candidates"
    allocator.main(_argv(case, output, cap))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["winner_selected"] is False
    assert manifest["schema"] == "prismaquant.mtp_option_b_candidates.v1"
    assert len(manifest["candidates"]) == 4
    assert manifest["rejected"] == []
    assignments = set()
    losses = {}
    for candidate in manifest["candidates"]:
        mtp = candidate["mtp_selection"]
        body = candidate["body_assignment"]
        mtp_assignment = candidate["mtp_assignment"]
        assignments.add(tuple(sorted(body.items())))
        assert mtp["selection"]["winner_selected"] is False
        assert not set(body) & set(mtp_assignment)
        assert {mtp_assignment[name] for name in MTP_UNITS[:2]} == {mtp["rung_by_group"]["pair"]}
        expected_mtp = sum(2 * PARAMS if fmt == "BF16" else payload["wire_bytes"][name][fmt]
                           for name, fmt in mtp_assignment.items())
        assert mtp["resident_bytes"] == expected_mtp
        assert mtp["objective"] == "mtp_head_self_kl"
        allowance = cap - 512 - (floor - mtp_source) - expected_mtp
        # Exhaustive oracle over the three one-unit options: the most precise
        # format that fits. Every probe lands on one of them, so the
        # min-dloss ratchet must agree exactly.
        expected_band = next(fmt for fmt in ("BF16", "FP8_E4M3", "NVFP4")
                             if _body_bytes({BODY: fmt}) <= allowance)
        assert body == {BODY: expected_band}
        losses.setdefault(expected_band, candidate["body_predicted_dloss"])
        assert candidate["body_predicted_dloss"] == losses[expected_band]
        assert candidate["body_byte_allowance"] == allowance
        assert candidate["immutable_and_fixed_bytes"] == floor - mtp_source
        stamp = candidate["whole_artifact_budget"]
        assert stamp["selection_tensor_payload_bytes"] == floor - mtp_source + expected_mtp + _body_bytes(body)
        assert stamp["selection_non_tensor_reserve_bytes"] == 512
        assert stamp["selection_whole_artifact_upper_bound_bytes"] <= cap
        combined = {**body, **mtp_assignment}
        assert stamp["selection_assignment_sha256"] == fp.assignment_serialization_sha256(combined)
        assert candidate["combined_assignment_sha256"] == stamp["selection_assignment_sha256"]
        from prismaquant.layer_config import load_assignment
        config_path = output / candidate["layer_config"]
        assert load_assignment(config_path) == combined
        config = json.loads(config_path.read_text())
        assert config["__prismaquant__"]["whole_artifact_budget"] == stamp
        assert config["__prismaquant__"]["mtp_selection"]["rung"] == mtp["rung"]
        assert config["__prismaquant__"]["mtp_selection"]["resident_bytes"] == expected_mtp
    assert len(assignments) > 1
    # MTP self-KL never enters the body objective: identical bodies carry
    # identical loss whatever the MTP choice, and the denser fitting body
    # carries strictly lower loss.
    assert losses["FP8_E4M3"] < losses["NVFP4"]


def test_option_b_command_exact_cap_and_refusal(tmp_path):
    case = _tiny_case(tmp_path)
    payload = case[4]
    floor = _floor_bytes(case[0], case[3])
    mtp_source = 3 * 2 * PARAMS
    combos = _mtp_combo_bytes(payload)
    cheapest_body = _body_bytes({BODY: "NVFP4"})
    # The dearest MTP choice (the independent winner's all-BF16 pair) prices
    # its cheapest body tight against the cap; cheaper choices keep headroom.
    # Integer accounting below is exact; the 512 B headroom keeps the CLI's
    # decimal-GB float conversion out of the boundary verdict.
    tight_upper = floor - mtp_source + cheapest_body + combos[-1] + 512
    cap = tight_upper + 512
    output = tmp_path / "command"
    command = [sys.executable, "-m", "prismaquant.allocator", *_argv(case, output, cap)]
    success = subprocess.run(command, capture_output=True, text=True)
    assert success.returncode == 0, success.stdout + success.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["candidates"]) == 4
    assert manifest["rejected"] == []
    tight = next(row for row in manifest["candidates"] if row["mtp_rung"] == "pair=BF16|single=BF16")
    assert tight["whole_artifact_budget"]["selection_whole_artifact_upper_bound_bytes"] == tight_upper
    assert 0 <= manifest["budget_bytes"] - tight_upper <= 1024
    assert tight["body_assignment"] == {BODY: "NVFP4"}
    refused_cap = floor - mtp_source + cheapest_body + combos[0] + 512 - 512
    refused_output = tmp_path / "refused"
    refused = subprocess.run([sys.executable, "-m", "prismaquant.allocator",
                              *_argv(case, refused_output, refused_cap)], capture_output=True, text=True)
    assert refused.returncode == 2, refused.stdout + refused.stderr
    refusal = json.loads((refused_output / "manifest.json").read_text())
    assert refusal["candidates"] == []
    assert len(refusal["rejected"]) == 4
    assert {row["reason"] for row in refusal["rejected"]} == {"below_floor"}


def test_option_b_refuses_body_mtp_source_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "format_is_producer_eligible", lambda name, **_: True)
    case = _tiny_case(tmp_path)
    original = fp.source_tensor_bytes_manifest

    def aliased(*args, **kwargs):
        manifest = original(*args, **kwargs)
        manifest.spans[BODY] = manifest.spans[MTP_UNITS[0]]
        return manifest

    monkeypatch.setattr(fp, "source_tensor_bytes_manifest", aliased)
    with pytest.raises(SystemExit, match="charged twice"):
        allocator.main(_argv(case, tmp_path / "alias", 1000000))


def test_enumeration_retains_bound_wires_and_exact_unit_prices(tmp_path):
    from test_glm_mtp_priced_wires_1413 import DENSE, _bound_parts_for_rates
    from prismaquant.glm_mtp_selection import enumerate_mtp_rungs, merge_mtp_costs
    from prismaquant import tessera_expert_projection as tep
    parts, units, carried = _bound_parts_for_rates(tmp_path, (1024, 896))
    merged = merge_mtp_costs([part for part, _source in parts], sources=[source for _part, source in parts])
    # The dense shared unit carries no bound M3 receipt, so a choice that
    # quantizes it cannot retain wire identities. Pin the group to BF16
    # passthrough, the same choice the winner selector ships, and enumerate
    # the stack's three admitted rungs.
    records = enumerate_mtp_rungs(merged, byte_budget=1000000, constants=CONSTANTS,
                                  fixed_formats={"shared": "BF16"})
    assert len(records) == 3
    for record in records:
        assert record["assignment"][DENSE] == "BF16"
        assert record["E"] == pytest.approx(sum(row["E"] for row in record["unit_prices"].values()))
        assert record["resident_bytes"] == sum(row["resident_bytes"] for row in record["unit_prices"].values())
        if all(fmt == "BF16" for fmt in record["assignment"].values()):
            assert record.get("mtp_expert_wires", {}) == {}
            continue
        assert record["mtp_expert_projection"] == carried
        for name in units:
            fmt = record["assignment"][name]
            if fmt == "BF16":
                assert name not in record["mtp_expert_wires"]
                assert record["unit_prices"][name]["E"] == 0
                continue
            index = 0 if fmt.endswith("R1024") else 1
            m3 = pickle.loads(Path(parts[index][0]["provenance"]["tessera_joint_anchors"]["inputs"]["merged_cost"]["path"]).read_bytes())
            assert record["mtp_expert_wires"][name] == m3[tep.EXPERT_WIRES_KEY][name][fmt]
            assert record["mtp_expert_source_bindings"][name]["m4"] == parts[index][1]
            assert record["unit_prices"][name].get("joint_operator_identity_sha256") == merged["costs"][name][fmt].get("joint_operator_identity_sha256")


def test_enumeration_is_order_independent_and_refuses_body_objective():
    from prismaquant.glm_mtp_selection import enumerate_mtp_rungs
    from test_glm_mtp_selection import _payload
    payload = _payload()
    records = enumerate_mtp_rungs(payload, byte_budget=1000000, constants=CONSTANTS)
    reordered = {**payload, "groups": {group: list(reversed(members))
                                       for group, members in reversed(list(payload["groups"].items()))}}
    assert enumerate_mtp_rungs(reordered, byte_budget=1000000, constants=CONSTANTS) == records
    with pytest.raises(ValueError, match="MTP objective"):
        enumerate_mtp_rungs(_payload(probe=_probe(objective=False)), byte_budget=1000000, constants=CONSTANTS)
