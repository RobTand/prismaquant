"""Allocator decisions and historical output bytes with explicit pin inputs.

Stock allocation clears the optional Tessera development pin. Tessera allocation
uses the test-owned v5 contract and clears the requested override too. Layer
hashes retain every field except the historical Tessera
contract-version/reviewed-digest normalization. Explicit accounting, assignment
and installed-pin checks accompany the complete file digests.
"""
from __future__ import annotations

import hashlib
import json
import csv
import math
import pickle
import sys

import pytest

STOCK_DIGESTS = {
    "layer_config.json": "ffd88f90f039f69e559be73ce9e302ce5406d4f48f8b7998db2a1156918a9942",
    "format_applicability.json": (
        "08076992b5150799091dc3a2c6cf63ad580c3c31e53e6c6e12cda1b715b848f8"
    ),
    "pareto.csv": (
        "0805fbc4dbffb240025dc5bfb8a4e97ddb241eda8e3c80818f07cdec425cfb32"
    ),
    "pareto.knees.json": (
        "2bc195e1fb1a76e2311faf91b35d1e700b8fc96306345f248896af4a46457dba"
    ),
    "pareto_seeds/allocator_target_4p6000_achieved_4p5020_c3f5ef1e4ab4.json": (
        "1cbec100e954d8ff94f065bc45a6808a41c65eb6da77e1f3a64d02c16172fa42"
    ),
    "pareto_seeds/allocator_target_5p5000_achieved_5p2783_81f5decbbbb0.json": (
        "fce87d165fe2660bbf03ada8414e3201b7ae12663be12fc232f6e7374832f7ad"
    ),
    "pareto_seeds/allocator_target_6p0000_achieved_5p7959_6ef0bae1963d.json": (
        "2196c306af84cc710084b358b553dab190450985cb1c15cb07716b671c010810"
    ),
    "pareto_seeds/allocator_target_8p0000_achieved_7p8662_9e13e69f37a9.json": (
        "a7fc3b502cdccbfec7c7adeeed890954dc032e8a2103ecdca75873e9da21b0b7"
    ),
    "pareto_seeds/manifest.json": (
        "7e80038d29a02961ddf43d27ab92268ef5506ac9b9c944308cfdb56c6aa7abcd"
    ),
}
TESSERA_DIGESTS = {
    "layer.json": "86645b65e35c088897347b4ca0f3454202d3b2194313b89485da8c41eb696275",
    "format_applicability.json": (
        "0e3d1f837879b2e9880c3dfbcfea38b7568f5a9a7c2e6bd898ec317fdd4dcfe2"
    ),
    "pareto.csv": (
        "e172f4262b5a094db870ed0a904d516d0b030926ee89431fe8472e8a5465583d"
    ),
    "pareto.knees.json": (
        "3735ae8eb6ecf8837e168542460da5eb085119614eac16568f57df72acc3edff"
    ),
}


def _canonical_tessera_layer(text):
    payload = json.loads(text)
    identity = payload["__prismaquant__"]["tessera_dev_pin"]
    for field in ("contract_version", "reviewed_contract_sha256"):
        del identity[field]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _digests(root):
    out = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel.endswith((".pkl",)) or rel.startswith("model/"):
            continue
        text = path.read_bytes().replace(str(root).encode(), b"<TMP>")
        if rel == "layer.json":
            text = _canonical_tessera_layer(text)
        out[rel] = hashlib.sha256(text).hexdigest()
    return out


def _assert_accounting(metadata, *, params, bits, target):
    assert metadata["schema"] == "prismaquant.layer_config_meta.v1"
    assert metadata["body_assignment_quantizable_params"] == params
    assert metadata["body_assignment_payload_bits_total"] == metadata["assignment_payload_bits_total"] == bits
    assert metadata["achieved_bits"] == bits / params
    assert metadata["target_bits"] == target
    assert metadata["body_shared_cb_sidecar_bits"] == 0
    assert metadata["assignment_payload_missing_stats_names"] == []
    assert metadata["lm_head_format"] == "BF16" and metadata["lm_head_mode"] == "profile_pinned_bf16"
    assert metadata["solver_contract"] == "additive_candidate_proposal_then_exact_assignment_filter"
    assert metadata["global_optimality_claimed"] is False
    for diagnostic in metadata["solve_diagnostics"].values():
        assert diagnostic["solver_calls"] == 1


def _assert_pareto_point(root, *, target, bits_per_param, loss):
    with (root / "pareto.csv").open() as source:
        row, = [row for row in csv.DictReader(source) if float(row["target_bits"]) == target]
    assert row["feasible"] == "True"
    assert float(row["achieved_bits"]) == bits_per_param
    assert math.isclose(float(row["predicted_dloss"]), loss, rel_tol=1e-12)
    assert math.isclose(float(row["total_predicted_dloss_with_aux"]), loss, rel_tol=1e-12)


def _stock_inputs(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
    }))
    names = [
        f"model.layers.{layer}.{leaf}"
        for layer in range(2)
        for leaf in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
            "mlp.down_proj",
        )
    ]
    stats, costs = {}, {}
    for idx, name in enumerate(names):
        stats[name] = {
            "h_trace": float(1 + (idx * 7) % 5),
            "n_params": 128 * 256,
            "in_features": 256,
            "out_features": 128,
        }
        costs[name] = {
            "NVFP4": {"predicted_dloss": 1.0 + 0.37 * idx},
            "FP8_DYNAMIC": {"predicted_dloss": 0.05 + 0.011 * idx},
            "BF16": {"predicted_dloss": 0.0},
        }
    (tmp_path / "probe.pkl").write_bytes(
        pickle.dumps({"stats": stats, "meta": {"model": str(model_dir)}}))
    (tmp_path / "cost.pkl").write_bytes(pickle.dumps({
        "costs": costs, "formats": ["NVFP4", "FP8_DYNAMIC", "BF16"]}))
    return [
        "--probe", str(tmp_path / "probe.pkl"),
        "--costs", str(tmp_path / "cost.pkl"),
        "--model-override", str(model_dir),
        "--formats", "NVFP4,FP8_DYNAMIC,BF16",
        "--target-bits", "6.0",
        "--pareto-targets", "4.6,5.5,6.0,8.0",
        "--bit-precision", "0.1",
        "--layer-config", str(tmp_path / "layer_config.json"),
        "--pareto-csv", str(tmp_path / "pareto.csv"),
        "--pareto-output-dir", str(tmp_path / "pareto_seeds"),
    ]


def test_stock_menu_preserves_decisions_accounting_and_companion_bytes(tmp_path, monkeypatch):
    from prismaquant import allocator

    monkeypatch.delenv("PRISMAQUANT_TESSERA_DEV_PIN", raising=False)
    monkeypatch.setattr(sys, "argv", ["allocator", *_stock_inputs(tmp_path)])
    allocator.main()
    from prismaquant.layer_config import load_assignment
    payload = json.loads((tmp_path / "layer_config.json").read_text())
    fp8 = {"model.layers.0.mlp.down_proj", *(
        "model.layers.1." + leaf for leaf in ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))}
    names = [f"model.layers.{layer}.{leaf}" for layer in range(2) for leaf in
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    expected = {name: "FP8_E4M3" if name in fp8 else "NVFP4" for name in names}
    assert load_assignment(tmp_path / "layer_config.json") == expected
    assert set(payload) == {*names, "__prismaquant__"}
    for name in names:
        if name in fp8:
            assert payload[name] == dict(bits=8, group_size=0, sym=True, data_type="fp8_e4m3",
                act_bits=8, act_group_size=0, act_sym=True, act_data_type="fp8_e4m3", act_dynamic=True)
        else:
            assert payload[name] == dict(bits=4, group_size=16, sym=True, data_type="nv_fp",
                act_bits=4, act_group_size=16, act_sym=True, act_data_type="nv_fp4_with_static_gs", act_dynamic=True)
    # UE4M3 per-16 scales and two global FP32 scales; FP8 row scales are FP32.
    n = 128 * 256
    bits = 9 * (n * 4 + (n // 16) * 8 + 64) + 5 * (n * 8 + 128 * 32)
    metadata = payload["__prismaquant__"]
    _assert_accounting(metadata, params=14 * n, bits=bits, target=6.0)
    assert "tessera_dev_pin" not in metadata
    provenance = metadata["serving_lane_provenance"]
    assert provenance["units_total"] == 14
    assert provenance["route_status_counts"] == {"no_declared_lane": 14}
    assert {name: row["units"] for name, row in provenance["by_format"].items()} == {"FP8_E4M3": 5, "NVFP4": 9}
    loss = sum((0.05 + 0.011 * index) if name in fp8 else (1.0 + 0.37 * index)
               for index, name in enumerate(names))
    _assert_pareto_point(tmp_path, target=6.0, bits_per_param=bits / (14 * n), loss=loss)
    got = _digests(tmp_path)
    assert got == STOCK_DIGESTS, json.dumps(got, indent=1, sort_keys=True)


def test_tessera_menu_keeps_allocation_and_nonidentity_outputs(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from test_tessera_scope_endpoints import (
        DENSE, EXPERT, SHARED, _allocator_inputs, _cli_scope, _v5_contract,
    )

    from prismaquant import allocator
    from prismaquant.layer_config import load_assignment

    monkeypatch.delenv("PRISMAQUANT_TESSERA_DEV_PIN", raising=False)
    _v5_contract(monkeypatch)
    argv = _allocator_inputs(tmp_path, "TESSERA_E2M1_K2_R896")
    monkeypatch.setattr(sys, "argv", [
        "allocator", *argv, "--no-fused-aggregation",
        "--no-packed-aggregation", *_cli_scope()])
    allocator.main()
    assert load_assignment(tmp_path / "layer.json") == {
        unit: "TESSERA_E2M1_K2_R896" for unit in (DENSE, EXPERT, SHARED)
    }
    from prismaquant import tessera_runtime_contract as runtime
    from prismaquant import tessera_serving_runtime_pin as pin
    payload = json.loads((tmp_path / "layer.json").read_text())
    identity = payload["__prismaquant__"]["tessera_dev_pin"]
    raw = runtime.contract_path().read_bytes()
    installed = json.loads(raw)
    serving_pin = pin.load_tessera_serving_runtime_pin()
    assert serving_pin.commit == pin.TESSERA_SERVING_RUNTIME_PINNED_COMMIT == runtime.TESSERA_DEV_PIN_COMMIT
    assert serving_pin.contract_sha256 == hashlib.sha256(raw).hexdigest() == runtime.TESSERA_DEV_PIN_CONTRACT_SHA256
    assert identity["requested"] == ""
    assert identity["commit"] == identity["contract_sha256"] == identity["contract_path"] == "fixture"
    assert identity["bytes_are_the_reviewed_bytes"] is False
    assert identity["contract_version"] == installed["contract_version"]
    assert identity["reviewed_contract_sha256"] == serving_pin.contract_sha256
    assert identity["plugin_version"] == serving_pin.version == "0.1.0"
    assert identity["quant_method"] == installed["quant_method"]["canonical"] == "tessera"
    assert identity["default_serve_image"] == installed["versions"]["default_serve_image"]
    for observed, published in zip(identity["native_extensions"], installed["native_extensions"], strict=True):
        for field in ("module_name_prefix", "filename_glob", "match", "source", "loaded_by", "routes", "when_unavailable"):
            assert observed[field] == published[field]
    metadata = payload["__prismaquant__"]
    # The unchanged historical side-byte oracle is also locked by the Pareto CSV.
    _assert_accounting(metadata, params=3 * 256 * 256, bits=815784, target=16.0)
    _assert_pareto_point(tmp_path, target=16.0, bits_per_param=815784 / (3 * 256 * 256), loss=3 * 0.5 * 1.0 * 4e-4)
    expected_scope = {DENSE: "dense", EXPERT: "routed_moe", SHARED: "dense"}
    provenance = metadata["serving_lane_provenance"]
    assert provenance["units_total"] == 3
    assert provenance["activation_pricing_branches"] == {"measured_output_mse": 3}
    assert provenance["route_status_counts"] == {"backed_with_serve_flag": 3}
    for name, structure in expected_scope.items():
        assert payload[name] == dict(data_type="tessera", bits=5, tessera_format="TESSERA_E2M1_K2_R896",
            tessera_family="TESSERA_E2M1_K2", tessera_body_rate_q256=896, tessera_body="TCQ",
            tessera_scale_plane="lut16", sym=True, group_size=16, act_bits=4,
            act_data_type="fp4_e2m1", act_group_size=16, act_dynamic=True, act_sym=True)
        row = provenance["by_unit"][name]
        assert row["format"] == "TESSERA_E2M1_K2_R896"
        assert row["serving_context"]["structure"] == structure
        assert row["route"]["route_status"] == "backed_with_serve_flag"
        assert row["route"]["requires_serve_flags"] == ["TESSERA_SERVE_MODE=resident"]
    got = _digests(tmp_path)
    assert got == TESSERA_DIGESTS, json.dumps(got, indent=1, sort_keys=True)
