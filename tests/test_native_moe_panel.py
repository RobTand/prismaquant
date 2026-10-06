"""Whole-stack boundary tests. These synthetic records are never measurements."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from prismaquant.joint_aura import arithmetic_identity, identity_sha256, make_joint_aura_entry
from prismaquant.native_moe_panel import (EXECUTION, FORMAT, INPUT_SCHEMA, ROLES, consume_moe_receipt,
    freeze_moe_panel, packed_reference, validate_routing, validate_transport, _validate_phase_tensors)
from prismaquant.production_weight_cache import _cb_cache_tensor_identity as tensor_id
from test_native_moe_glm_geometry import glm_routing, glm_shape


def routing():
    return {"activation": "silu", "scoring_func": "sigmoid", "renormalize": True,
        "routed_scaling_factor": 1.0, "apply_router_weight_on_input": False, "expert_map": None,
        "input_dtype": "torch.bfloat16", "topk_weights_dtype": "torch.float32",
        "topk_ids_dtype": "torch.int32", "device": "cuda:0",
        "weights_contract": "post_renormalization_and_routed_scaling", "source_protocol": {
            "router_class": "fixture.Lfm2MoeTopKRouter", "router_source_sha256": "d" * 64,
            "selection_bias": tensor_id(torch.zeros(32)), "normalization_epsilon": 1e-6,
            "expert_bias_affects": "selection_only"}}


def phase_tensors():
    raw_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64)
    # BF16 source normalization need not sum exactly to one. Never repair it.
    raw_weights = torch.tensor([[.498046875, .5], [.75, .2490234375]], dtype=torch.bfloat16)
    values = {"input": torch.ones(2, 4, dtype=torch.bfloat16), "topk_ids": raw_ids.int(),
        "topk_weights": raw_weights.float(), "source_topk_ids": raw_ids, "source_topk_weights": raw_weights}
    transport = {name: {"source": tensor_id(values["source_" + name]), "supplied": tensor_id(values[name]),
                       "operation": "lossless_dtype_conversion"} for name in ("topk_ids", "topk_weights")}
    return values, transport


@pytest.fixture
def joined():
    unit = "model.layers.2.feed_forward.experts"
    shape = {"experts": 32, "hidden_size": 4, "intermediate_size": 3, "top_k": 2}
    members = []
    activation = {"schema": "prismaquant.joint_aura.activation.v1", "quantizes_input": True,
                  "activation_max_abs": None, "input_global_scale": None, "clip_enabled": False}
    for expert in range(32):
        for role in ROLES:
            dims = [4, 3] if role == "w2" else [3, 4]
            weight = tensor_id(torch.ones(dims, dtype=torch.bfloat16))
            members.append({"unit": f"{unit}.{expert}.{role}", "expert": expert, "role": role,
                "format": FORMAT, "shape": dims, "source_weight": weight, "rendered_weight": weight,
                "activation": copy.deepcopy(activation), "wire": {"blob_sha256": "3" * 64, "blob_bytes": 42,
                    "record": {"unit": f"{unit}.{expert}.{role}"}}})
    source = {"files": {"fixture.safetensors": "8" * 64}, "config_sha256": "9" * 64,
              "auxiliary_sha256": {"config.json": "9" * 64},
              "tensors": {member["unit"] + ".weight": "fixture.safetensors" for member in members}}
    config = {"model_type": "lfm2_moe", "fixture": True}
    source_execution = {"schema": "prismaquant.joint_aura.source_execution.v1", "modules": {
        "": {"attention": "eager", "experts": "grouped_mm"},
        unit: {"attention": "eager", "experts": "grouped_mm"}}}
    value = {"config": config, "weight_map": {name: name for name in source["tensors"]},
             "checkpoint_weight_map": source["tensors"],
             "shards": [{"path": "/fixture/fixture.safetensors", "size": 1, "sha256": "8" * 64}]}
    model = {"schema": "prismaquant.streamed_model.identity.v1", "source": "/fixture", "resolved_commit": None,
             "content_sha256": identity_sha256(value), **value}
    arithmetic = arithmetic_identity(torch.bfloat16)
    probe = {"schema": "prismaquant.joint_aura.probes.v2", "source_model": model,
        "calibration_sha256": "1" * 64, "calibration_shape": [1, 2], "calibration_dtype": "torch.int64",
        "producer_source_sha256": "2" * 64, "n_probes": 3, "seed_base": 7, "token_scope": "causal",
        "distribution": "rademacher", "normalization": "global_kl_fisher", "temperature": 1.0,
        "arithmetic": arithmetic, "source_execution": copy.deepcopy(source_execution)}
    rows = {}
    for member in members:
        joint = {"schema": "prismaquant.joint_aura.operator.v2", "qname": member["unit"], "format": FORMAT,
            **{key: member[key] for key in ("source_weight", "rendered_weight", "activation")},
            "arithmetic": arithmetic, "probe_identity_sha256": identity_sha256(probe)}
        rows[member["unit"]] = make_joint_aura_entry(operator_identity=joint, probe_identity=probe,
            signed_components=[{"weight": v, "activation": 0., "mixed": 0., "total": v} for v in (.1, -.2, .3)])
    values, transport = phase_tensors()
    tensor_fields = {key: tensor_id(value) for key, value in values.items() if not key.startswith("source_")}
    phases = {phase: {"m": 2, **tensor_fields, "reference_qdq": tensor_fields["input"],
                     "reference_output": tensor_fields["input"], "transport": copy.deepcopy(transport)}
              for phase in ("prefill", "decode")}
    calibration = {"schema": "prismaquant.calibration_input.v1", "calibration_sha256": "1" * 64,
                   "shape": [1, 2], "dtype": "torch.int64"}
    capture = {"schema": "prismaquant.routed_boundary_capture.v1", "unit": unit, "shape": shape,
        "routing": routing(), "calibration_sha256": "1" * 64, "calibration_shape": [1, 2],
        "calibration_dtype": "torch.int64", "producer_source": source, "runtime_config": config,
        "source_execution": copy.deepcopy(source_execution),
        "capture_source_sha256": "c" * 64, "phases": phases,
        "model_load_contract": {"schema": "prismaquant.pretrained_initialization.v1", "scope": "checkpoint_missing_state",
                                "status": "completed", "transformers_version": "fixture-transformers"},
        "attention_implementation": "eager", "capture_runtime": {"torch": "fixture-torch", "cuda": "fixture-cuda",
                                                                   "transformers": "fixture-transformers"}}
    inputs = {"schema": INPUT_SCHEMA, "unit": unit, "format": FORMAT, "shape": shape, "members": members,
        "profile_role_order": list(ROLES), "routing": routing(), "execution": dict(EXECUTION),
        "calibration": calibration, "routing_capture": capture, "routing_capture_sha256": identity_sha256(capture),
        "runtime_image": "fixture/image@sha256:" + "a" * 64, "serving_config_sha256": "b" * 64, "numerics": {"atol": .015625, "rtol": .015625},
        "phases": phases, "probe_request": {
            **{key: probe[key] for key in ("n_probes", "seed_base", "token_scope", "temperature", "distribution", "normalization")},
            "source_model": "/fixture", "source_shards": source["files"], "source_config_sha256": source["config_sha256"],
            "source_auxiliary_sha256": source["auxiliary_sha256"]}}
    native_members = [{**{key: member[key] for key in ("unit", "expert", "role", "format", "shape", "source_weight", "rendered_weight")},
        "wire_sha256": member["wire"]["blob_sha256"], "wire_record_sha256": identity_sha256(member["wire"]["record"])} for member in members]
    native = {"members": native_members, "shape": shape, "routing": routing(), "profile_role_order": list(ROLES),
        "routing_capture_sha256": inputs["routing_capture_sha256"], "serving_config_sha256": "b" * 64, "native_tensors": {"fixture": tensor_fields["input"]},
        "scheme": {"fixture": True}, "config": {"fixture": "actual MoE config"},
        "phases": {phase: {"transport": value["transport"]} for phase, value in phases.items()},
        "declared_route": {"kind": "moe", "policy": "TESSERA_FP8:resident",
            "symbol": "vllm.fused_moe.modular_kernel:fixture", "decoder": "torch_materialize_stock",
            "contract": "fp8_per_token_dynamic"}}
    native["config_sha256"] = identity_sha256(native["config"])
    runtime = {"schema": "tessera.native_moe_runtime.v1", "execution": dict(EXECUTION),
        "image": inputs["runtime_image"], "resource_collector": {"library_sha256": "5" * 64}}
    workspace = {"schema": "tessera.native_moe_workspace.v1", "owner": "vllm.WorkspaceManager",
        "num_ubatches": 1, "num_lanes": 1, "locked": True, "slots": [{"index": 0, "shape": [64],
            "dtype": "torch.uint8", "device": "cuda:0", "storage_bytes": 64, "logical_bytes": 64,
            "stride": [1], "storage_offset": 0}], "resident_bytes": 64}
    preflight = {"schema": "tessera.native_moe_preflight.v1", "status": "untimed_preparation", "operator": native,
        "runtime": runtime, "runtime_sha256": identity_sha256(runtime), "workspace": workspace,
        "workspace_sha256": identity_sha256(workspace), "native_tensors_sha256": identity_sha256(native["native_tensors"]),
        "scheme_sha256": identity_sha256(native["scheme"])}
    return inputs, preflight, rows


def test_complete_runtime_binding_has_96_members_and_no_new_group_cost(joined):
    panel = freeze_moe_panel(*joined, cost_sha256="4" * 64)
    assert len(panel["runtime_binding"]["member_operator_identity_sha256"]) == 96
    assert "predicted_dloss" not in panel and "group_cost" not in panel
    assert panel["profile_role_order"] == ["w1", "w3", "w2"]


@pytest.mark.parametrize("change", ["experts", "attention", "missing_capture", "missing_probe"])
def test_freeze_rejects_changed_or_unrecorded_source_backend(joined, change):
    inputs, preflight, rows = joined
    if change == "missing_capture":
        inputs["routing_capture"].pop("source_execution")
    elif change == "missing_probe":
        for row in rows.values():
            row["probe_identity"].pop("source_execution", None)
            row["probe_identity_sha256"] = identity_sha256(row["probe_identity"])
            row["joint_operator_identity"]["probe_identity_sha256"] = row["probe_identity_sha256"]
            row["joint_operator_identity_sha256"] = identity_sha256(row["joint_operator_identity"])
    else:
        inputs["routing_capture"]["source_execution"]["modules"][inputs["unit"]][change] = "eager" if change == "experts" else "sdpa"
    digest = identity_sha256(inputs["routing_capture"])
    inputs["routing_capture_sha256"] = digest
    preflight["operator"]["routing_capture_sha256"] = digest
    with pytest.raises(ValueError, match="source execution"):
        freeze_moe_panel(inputs, preflight, rows, cost_sha256="4" * 64)


@pytest.mark.parametrize("change", [None, "file", "artifact", "runtime", "source", "calibration",
                                     "backend", "bytes", "dtype", "missing_tensor", "not_equal"])
def test_retained_boundary_requires_independent_exact_source_qualification(joined, tmp_path, change):
    inputs, preflight, rows = joined
    capture = inputs["routing_capture"]
    execution = capture.pop("source_execution")
    inputs["routing_capture_sha256"] = identity_sha256(capture)
    preflight["operator"]["routing_capture_sha256"] = inputs["routing_capture_sha256"]
    inputs["source_capture"] = {"routing_boundary_sha256": "f" * 64}
    inputs["calibration"]["artifact_sha256"] = "a" * 64
    first = next(iter(rows.values()))["probe_identity"]
    values, transport = phase_tensors()
    tensors = {"inputs": values["input"], "top_k_index": values["source_topk_ids"],
        "top_k_weights": values["source_topk_weights"], "expert_bias": torch.zeros(32),
        "coordinates": torch.tensor([[0, 0], [0, 1]], dtype=torch.int64)}
    identities = {name: tensor_id(value) for name, value in tensors.items()}
    proof = {"schema": "prismaquant.packed_source_boundary_qualification.v1", "unit_qname": inputs["unit"],
        "artifact_sha256": "f" * 64, "boundary_metadata": {**copy.deepcopy(capture),
            "profile_role_order": list(ROLES), "tensors": identities},
        "source_execution_identity": execution, "streamed_source_execution_identity": copy.deepcopy(execution),
        "source_model_identity": copy.deepcopy(first["source_model"]), "runtime": copy.deepcopy(capture["capture_runtime"]),
        "calibration_subset": {"artifact_sha256": "a" * 64, "full_shape": [1, 2], "row": 0,
            "shape": [1, 2], "dtype": "torch.int64", "subset_artifact_sha256": "a" * 64, "sha256": "1" * 64},
        "tensor_comparisons": {name: {"equal": True, "shape": value["shape"], "dtype": value["dtype"],
            "actual_sha256": value["content_sha256"], "captured_sha256": value["content_sha256"]}
            for name, value in identities.items()}}
    if change == "artifact":
        proof["artifact_sha256"] = "0" * 64
    elif change == "runtime":
        proof["runtime"]["torch"] = "another-runtime"
    elif change == "source":
        proof["source_model_identity"]["source"] = "/other"
    elif change == "calibration":
        proof["calibration_subset"]["row"] = 1
    elif change == "backend":
        proof["source_execution_identity"]["modules"][inputs["unit"]]["experts"] = "eager"
    elif change == "bytes":
        proof["tensor_comparisons"]["inputs"]["actual_sha256"] = "0" * 64
    elif change == "dtype":
        proof["tensor_comparisons"]["top_k_weights"]["dtype"] = "torch.float32"
    elif change == "missing_tensor":
        proof["tensor_comparisons"].pop("expert_bias")
    elif change == "not_equal":
        proof["tensor_comparisons"]["inputs"]["equal"] = False
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"schema": "prismaquant.packed_joint_screen.v1", "mode": "source",
                               "passed": True, "retained_boundary_qualification": proof}))
    digest = "0" * 64 if change == "file" else hashlib.sha256(path.read_bytes()).hexdigest()
    kwargs = {"cost_sha256": "4" * 64, "source_execution_qualification_path": path,
              "source_execution_qualification_sha256": digest}
    if change is None:
        panel = freeze_moe_panel(inputs, preflight, rows, **kwargs)
        assert panel["source_execution"] == execution
        assert panel["source_execution_qualification_sha256"] == digest
        assert "source_execution" not in capture
    else:
        with pytest.raises(ValueError):
            freeze_moe_panel(inputs, preflight, rows, **kwargs)


@pytest.mark.parametrize("change", [None, "bias_bytes", "bias_dtype", "bias_shape"])
def test_glm_source_qualification_binds_original_correction_bias(tmp_path, change):
    from prismaquant.native_moe_panel import _qualified_source_execution, validate_geometry

    unit = "model.language_model.layers.3.mlp.experts"
    shape = glm_shape(tensor_parallel=2, tensor_parallel_rank=1)
    validate_geometry(shape)
    bias = tensor_id(torch.arange(shape["n_routed_experts"], dtype=torch.float32))
    route = glm_routing()
    route["source_protocol"]["correction_bias"] = {
        key: bias[key] for key in ("content_sha256", "dtype")}
    validate_routing(route)
    identities = {
        "inputs": tensor_id(torch.ones(2, shape["hidden_size"], dtype=torch.bfloat16)),
        "top_k_index": tensor_id(torch.arange(16, dtype=torch.int64).reshape(2, 8)),
        "top_k_weights": tensor_id(torch.full((2, 8), .125, dtype=torch.bfloat16)),
        "expert_bias": bias,
        "coordinates": tensor_id(torch.tensor([[0, 0], [0, 1]], dtype=torch.int64)),
    }
    execution = {"schema": "prismaquant.joint_aura.source_execution.v1", "modules": {
        "": {"attention": "eager", "experts": "grouped_mm"},
        unit: {"attention": "eager", "experts": "grouped_mm"}}}
    capture = {"unit": unit, "shape": shape, "profile_role_order": list(ROLES),
        "calibration_sha256": "1" * 64, "calibration_shape": [1, 2],
        "calibration_dtype": "torch.int64", "producer_source": {"fixture": True},
        "runtime_config": {"model_type": "glm5_next"}, "capture_source_sha256": "2" * 64,
        "model_load_contract": {"fixture": True}, "attention_implementation": "eager",
        "capture_runtime": {"torch": "fixture-torch", "cuda": "fixture-cuda", "transformers": "fixture-transformers"}}
    inputs = {"unit": unit, "shape": shape, "profile_role_order": list(ROLES), "routing": route,
        "routing_capture": capture, "source_capture": {"routing_boundary_sha256": "3" * 64},
        "calibration": {"artifact_sha256": "4" * 64, "shape": [1, 2], "dtype": "torch.int64", "calibration_sha256": "1" * 64},
        "phases": {"prefill": {"m": 2, "input": identities["inputs"], "transport": {
            "topk_ids": {"source": identities["top_k_index"]},
            "topk_weights": {"source": identities["top_k_weights"]}}}}}
    probe = {"source_model": {"source": "/fixture/original-glm"}}
    proof = {"schema": "prismaquant.packed_source_boundary_qualification.v1", "unit_qname": unit,
        "artifact_sha256": "3" * 64, "boundary_metadata": {**copy.deepcopy(capture), "tensors": identities},
        "source_execution_identity": execution, "streamed_source_execution_identity": copy.deepcopy(execution),
        "source_model_identity": copy.deepcopy(probe["source_model"]), "runtime": copy.deepcopy(capture["capture_runtime"]),
        "calibration_subset": {"artifact_sha256": "4" * 64, "full_shape": [1, 2], "row": 0,
            "shape": [1, 2], "dtype": "torch.int64", "subset_artifact_sha256": "4" * 64, "sha256": "1" * 64},
        "tensor_comparisons": {name: {"equal": True, "shape": value["shape"], "dtype": value["dtype"],
            "actual_sha256": value["content_sha256"], "captured_sha256": value["content_sha256"]}
            for name, value in identities.items()}}
    if change is not None:
        key, value = {"bias_bytes": ("actual_sha256", "0" * 64),
                      "bias_dtype": ("dtype", "torch.bfloat16"),
                      "bias_shape": ("shape", [shape["n_routed_experts"] - 1])}[change]
        proof["tensor_comparisons"]["expert_bias"][key] = value
    path = tmp_path / "glm-source.json"
    path.write_text(json.dumps({"schema": "prismaquant.packed_joint_screen.v1", "mode": "source",
                               "passed": True, "retained_boundary_qualification": proof}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if change is None:
        assert _qualified_source_execution(inputs, probe, path, digest) == execution
    else:
        with pytest.raises(ValueError):
            _qualified_source_execution(inputs, probe, path, digest)


@pytest.mark.parametrize("change", ["missing", "order", "format", "rows", "probe", "preclip", "wire", "source",
                                    "config", "route", "runtime", "capture", "transport", "workspace"])
def test_changed_or_partial_stack_cannot_freeze(joined, change):
    inputs, preflight, rows = joined
    first = inputs["members"][0]["unit"]
    if change == "missing":
        inputs["members"].pop()
    elif change == "order":
        inputs["members"][0], inputs["members"][1] = inputs["members"][1], inputs["members"][0]
    elif change == "format":
        inputs["members"][0]["format"] = "TESSERA_BF16_K1_R1792"
    elif change == "rows":
        rows.pop(first)
    elif change == "probe":
        inputs["probe_request"]["seed_base"] += 1
    elif change == "preclip":
        inputs["members"][0]["activation"]["clip_enabled"] = True
    elif change == "wire":
        preflight["operator"]["members"][0]["wire_sha256"] = "0" * 64
    elif change == "source":
        inputs["probe_request"]["source_shards"] = {"fixture.safetensors": "0" * 64}
    elif change == "config":
        inputs["routing_capture"]["runtime_config"] = {"model_type": "different"}
        inputs["routing_capture_sha256"] = identity_sha256(inputs["routing_capture"])
    elif change == "route":
        preflight["operator"]["declared_route"]["kind"] = "dense"
    elif change == "runtime":
        preflight["runtime"]["execution"]["expert_parallel"] = 2
        preflight["runtime_sha256"] = identity_sha256(preflight["runtime"])
    elif change == "capture":
        inputs["routing_capture_sha256"] = "0" * 64
    elif change == "transport":
        inputs["phases"]["prefill"]["transport"]["topk_weights"]["operation"] = "renormalized"
    else:
        preflight["workspace"]["locked"] = False
        preflight["workspace_sha256"] = identity_sha256(preflight["workspace"])
    with pytest.raises((ValueError, RuntimeError)):
        freeze_moe_panel(inputs, preflight, rows, cost_sha256="4" * 64)


def test_lossless_transport_keeps_real_bf16_rounding():
    values, transport = phase_tensors()
    validate_transport(values, transport)
    _validate_phase_tensors(values["input"], values["topk_ids"], values["topk_weights"],
                           {"hidden_size": 4, "top_k": 2, "experts": 32}, cuda=False)
    assert not torch.equal(values["topk_weights"].sum(-1), torch.ones(2))


@pytest.mark.parametrize("change", ["renormalize", "overflow", "reorder", "source_hash"])
def test_transport_cannot_repair_or_replace_routing(change):
    values, transport = phase_tensors()
    if change == "renormalize":
        values["topk_weights"] /= values["topk_weights"].sum(-1, keepdim=True)
        transport["topk_weights"]["supplied"] = tensor_id(values["topk_weights"])
    elif change == "overflow":
        values["source_topk_ids"][0, 0] = 2**32
        transport["topk_ids"]["source"] = tensor_id(values["source_topk_ids"])
    elif change == "reorder":
        values["topk_ids"] = values["topk_ids"].flip(-1)
        transport["topk_ids"]["supplied"] = tensor_id(values["topk_ids"])
    else:
        transport["topk_weights"]["source"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_transport(values, transport)


def test_route_rejects_softmax_label_and_input_weighting():
    for key, value in (("scoring_func", "softmax"), ("apply_router_weight_on_input", True), ("routed_scaling_factor", 2.)):
        changed = routing() | {key: value}
        with pytest.raises(ValueError):
            validate_routing(changed)


def test_reference_preserves_external_weights_without_renormalizing(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_PROD_ACT_SCALES", "0")
    from prismaquant import format_registry
    from prismaquant.production_weight_cache import ProductionWeightCache
    values, _ = phase_tensors()
    experts = SimpleNamespace(num_experts=4, act_fn=torch.nn.functional.silu)
    gate_up = torch.arange(4 * 6 * 4, dtype=torch.float32).reshape(4, 6, 4).div(32).to(torch.bfloat16)
    down = torch.ones(4, 4, 3, dtype=torch.bfloat16)
    # Tessera E4M3's registry synthesis delegates its activation QDQ to this
    # existing owner. This reference test exercises that owner without encoding.
    spec, cache = format_registry.get_format("FP8_E4M3"), ProductionWeightCache(weights={}, levers={})
    left = packed_reference(experts, values["input"], values["topk_ids"], values["topk_weights"], gate_up, down, spec=spec, cache=cache)
    right = packed_reference(experts, values["input"], values["topk_ids"], values["topk_weights"] / 2, gate_up, down, spec=spec, cache=cache)
    assert torch.equal(left / 2, right)
    monkeypatch.setenv("PRISMAQUANT_PROD_ACT_SCALES", "1")
    with pytest.raises(ValueError, match="PRISMAQUANT_PROD_ACT_SCALES"):
        packed_reference(experts, values["input"], values["topk_ids"], values["topk_weights"], gate_up, down, spec=spec, cache=cache)


def receipt_fixture(joined, complete=True):
    inputs, preflight, _ = joined
    panel = freeze_moe_panel(*joined, cost_sha256="4" * 64)
    # ``max_abs_error`` is what the activation gate decides on since #574, and
    # the receipt harness has always emitted it (bench_native_operator's
    # compare_tensors); an agreeing runtime reports exactly zero.
    error = {"status": "passed", "finite": True, "max_normalized_error": .25,
             "max_abs_error": 0.0, **panel["numerics"]}
    phases = {phase: {**inputs["phases"][phase], "numerics": dict(error), "qdq_numerics": dict(error),
        "route": {**preflight["operator"]["declared_route"], "state": "served", "reason": None, "shape": "M2:N6:K4"},
        "measurement": {"method": "cuda_events", "sample_unit": "single_apply", "samples_ms": [3., 1., 2.],
                        "warmup_iterations": 4}} for phase in ("prefill", "decode")}
    trace = {"fixture": "trace", "capture": {"collector_library_sha256": "5" * 64}}
    bound = {"status": "complete_operator_bound", "composition": "sum_of_independent_peaks_including_output",
             "full_model_fixed_resources_complete": False, "peak_scratch_bytes": 128,
             "external_native_peak_bytes": 64, "torch_peak_increment_bytes": 64}
    receipt = {"schema": "tessera.native_moe_operator_receipt.v1", "status": "timing_admissible",
        "panel": panel, "panel_sha256": identity_sha256(panel), "operator": preflight["operator"],
        "runtime": preflight["runtime"], "runtime_sha256": preflight["runtime_sha256"],
        "phases": phases,
        "resources": {"status": "complete_operator_bound" if complete else "incomplete", "resident_bytes": 100,
            "workspace_resident_bytes": 64, "workspace_sha256": preflight["workspace_sha256"],
            "trace_sha256": identity_sha256(trace), "phases": {
                phase: {"bound": copy.deepcopy(bound)} if complete else {} for phase in ("prefill", "decode")}}}
    return panel, receipt, trace


def write(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("complete", [False, True])
def test_whole_apply_evidence_preserves_workspace_and_full_model_unknown(joined, tmp_path, complete):
    panel, receipt, trace = receipt_fixture(joined, complete)
    path, trace_path = tmp_path / "receipt.json", tmp_path / "trace.json"
    digest = write(path, receipt)
    write(trace_path, trace)
    observed = consume_moe_receipt(path, expected_sha256=digest, expected_panel=panel,
                                   memory_trace_path=trace_path if complete else None)
    assert observed["resident_bytes"] == 100
    assert observed["workspace_resident_bytes"] == 64
    assert observed["phases"]["prefill"]["median_ms"] == 2.
    assert observed["phases"]["prefill"]["peak_scratch_bytes"] == (128 if complete else None)
    assert observed["runtime_table_admissible"] is False and observed["full_model_resources"] is None
    assert "cross_operator_workspace_composition" in observed["unknown"]


@pytest.mark.parametrize("change", ["missing_member", "weights", "workspace", "workspace_digest", "workspace_bytes", "resource_trace",
                                    "scalar_leaf_sum", "tolerance", "scratch", "route_shape", "config", "raw_identity"])
def test_whole_receipt_cannot_replace_frozen_inputs_or_resources(joined, tmp_path, change):
    panel, receipt, trace = receipt_fixture(joined)
    if change == "missing_member":
        receipt["operator"]["members"].pop()
    elif change == "weights":
        receipt["phases"]["prefill"]["topk_weights"] = dict(receipt["phases"]["prefill"]["topk_weights"], content_sha256="0" * 64)
    elif change == "workspace":
        receipt["panel"] = copy.deepcopy(receipt["panel"])
        receipt["panel"]["workspace"]["slots"][0]["shape"] = [32]
        receipt["panel"]["workspace_sha256"] = identity_sha256(receipt["panel"]["workspace"])
        receipt["panel_sha256"] = identity_sha256(receipt["panel"])
    elif change == "workspace_digest":
        receipt["resources"]["workspace_sha256"] = "0" * 64
    elif change == "workspace_bytes":
        receipt["resources"]["workspace_resident_bytes"] = 0
    elif change == "resource_trace":
        trace["capture"]["collector_library_sha256"] = "0" * 64
        receipt["resources"]["trace_sha256"] = identity_sha256(trace)
    elif change == "scalar_leaf_sum":
        receipt["phases"]["decode"]["measurement"]["sample_unit"] = "sum_of_leaf_medians"
    elif change == "tolerance":
        receipt["phases"]["prefill"]["numerics"]["atol"] *= 2
    elif change == "scratch":
        receipt["resources"]["phases"]["decode"]["bound"]["peak_scratch_bytes"] = 64
    elif change == "route_shape":
        receipt["phases"]["decode"]["route"]["shape"] = "M2:N3:K4"
    elif change == "config":
        receipt["operator"]["config"] = {"different": "MoE config"}
        receipt["operator"]["config_sha256"] = identity_sha256(receipt["operator"]["config"])
    else:
        receipt["phases"]["decode"]["transport"] = copy.deepcopy(receipt["phases"]["decode"]["transport"])
        receipt["phases"]["decode"]["transport"]["topk_ids"]["source"]["content_sha256"] = "0" * 64
    path, trace_path = tmp_path / "receipt.json", tmp_path / "trace.json"
    digest = write(path, receipt)
    write(trace_path, trace)
    with pytest.raises((ValueError, RuntimeError)):
        consume_moe_receipt(path, expected_sha256=digest, expected_panel=panel, memory_trace_path=trace_path)


@pytest.mark.parametrize("change", ["missing_init", "noncanonical_init", "attention", "runtime_version"])
def test_quarantined_capture_cannot_qualify_native_panel(joined, change):
    inputs, preflight, rows = joined
    capture = inputs["routing_capture"]
    if change == "missing_init":
        capture["model_load_contract"] = None
    elif change == "noncanonical_init":
        capture["model_load_contract"]["status"] = "operator_asserted"
    elif change == "attention":
        capture["attention_implementation"] = "sdpa"
    else:
        capture["capture_runtime"]["transformers"] = "different-transformers"
    inputs["routing_capture_sha256"] = identity_sha256(capture)
    with pytest.raises(ValueError):
        freeze_moe_panel(inputs, preflight, rows, cost_sha256="4" * 64)


@pytest.fixture
def raw_boundary(joined):
    inputs, _, _ = joined
    values, _ = phase_tensors()
    tensors = {"inputs": values["input"], "top_k_index": values["source_topk_ids"],
        "top_k_weights": values["source_topk_weights"], "coordinates": torch.tensor([[0, 0], [0, 1]]),
        "expert_bias": torch.zeros(32)}
    meta = {key: copy.deepcopy(value) for key, value in inputs["routing_capture"].items() if key != "phases"}
    meta.update(schema="prismaquant.native_moe_raw_boundary.v1", profile_role_order=list(ROLES),
                scope="first calibration sequence; decode uses its first row, not autoregressive generation")
    meta["routing"].update(topk_ids_dtype="torch.int64", topk_weights_dtype="torch.bfloat16")
    meta["tensors"] = {key: tensor_id(value) for key, value in tensors.items()}
    manifest = {"schema": "prismaquant.tessera_calibration_cache.v2", "status": "complete", "identity": {
        **{key: copy.deepcopy(meta[key]) for key in ("model_load_contract", "capture_runtime", "attention_implementation")},
        "source_files": {**meta["producer_source"]["files"], **meta["producer_source"]["auxiliary_sha256"]}}}
    return {"source": "routed_boundary_capture", "boundary_metadata": meta, **tensors}, inputs["calibration"], manifest


def test_raw_boundary_transport_preserves_original_bf16_weights(raw_boundary):
    from prismaquant.native_moe_panel import routed_boundary_inputs
    raw, calibration, manifest = raw_boundary
    capture, phases, bias = routed_boundary_inputs(raw, calibration_receipt=calibration,
                                                  capture_manifest=manifest, device="cpu")
    assert phases["decode"]["input"].shape == (1, 4)
    assert torch.equal(phases["prefill"]["topk_weights"], raw["top_k_weights"].float())
    assert float(phases["prefill"]["topk_weights"][0].sum()) != 1.0
    assert capture["phases"]["prefill"]["transport"]["topk_weights"]["source"] == tensor_id(raw["top_k_weights"])
    assert bias.dtype == torch.float32


@pytest.mark.parametrize("change", ["historical", "coordinates", "weights", "dtype", "bias", "initialization", "source"])
def test_raw_boundary_changes_cannot_enter_native_protocol(raw_boundary, change):
    from prismaquant.native_moe_panel import routed_boundary_inputs
    raw, calibration, manifest = raw_boundary
    if change == "historical":
        manifest["schema"] = "prismaquant.tessera_calibration_cache.v1"
    elif change == "coordinates":
        raw["coordinates"] = raw["coordinates"].flip(0)
        raw["boundary_metadata"]["tensors"]["coordinates"] = tensor_id(raw["coordinates"])
    elif change == "weights":
        raw["top_k_weights"][0, 0] = .5
    elif change == "dtype":
        raw["boundary_metadata"]["routing"]["topk_weights_dtype"] = "torch.float32"
    elif change == "bias":
        raw["expert_bias"] = raw["expert_bias"].bfloat16()
        raw["boundary_metadata"]["tensors"]["expert_bias"] = tensor_id(raw["expert_bias"])
    elif change == "initialization":
        manifest["identity"]["model_load_contract"]["status"] = "skipped"
    else:
        manifest["identity"]["source_files"]["fixture.safetensors"] = "0" * 64
    with pytest.raises(ValueError):
        routed_boundary_inputs(raw, calibration_receipt=calibration, capture_manifest=manifest, device="cpu")


def token_receipt(tokens):
    identity = tensor_id(tokens)
    return {"schema": "prismaquant.calibration_input.v1", "shape": identity["shape"],
            "dtype": identity["dtype"], "calibration_sha256": identity["content_sha256"]}


def test_subset_probe_verifies_actual_ids_and_keeps_full_parent_distinct():
    from prismaquant.native_moe_panel import verified_probe_subset, _probe_calibration
    parent = torch.tensor([[1, 2], [3, 4]])
    subset = parent[:1].clone()
    full_receipt, subset_receipt = token_receipt(parent), token_receipt(subset)
    scope = verified_probe_subset(parent, subset, parent_calibration=full_receipt, subset_calibration=subset_receipt)
    assert scope["parent_calibration_sha256"] != scope["subset_calibration_sha256"]
    assert _probe_calibration({"calibration": full_receipt, "probe_calibration": subset_receipt, "probe_scope": scope}) == subset_receipt
    with pytest.raises(ValueError, match="actual first calibration sequence"):
        verified_probe_subset(parent, parent[1:], parent_calibration=full_receipt, subset_calibration=token_receipt(parent[1:]))
    with pytest.raises(ValueError, match="actual subset calibration_sha256"):
        verified_probe_subset(parent, subset, parent_calibration=full_receipt, subset_calibration=subset_receipt | {"calibration_sha256": "0" * 64})


def test_subset_joint_panel_retains_both_calibration_scopes(joined):
    from prismaquant.native_moe_panel import verified_probe_subset
    inputs, preflight, old_rows = joined
    parent = torch.tensor([[1, 2], [3, 4]])
    inputs["calibration"] = token_receipt(parent)
    inputs["probe_calibration"] = token_receipt(parent[:1])
    inputs["probe_scope"] = verified_probe_subset(parent, parent[:1], parent_calibration=inputs["calibration"],
                                                  subset_calibration=inputs["probe_calibration"])
    for key, value in (("calibration_shape", [2, 2]), ("calibration_sha256", inputs["calibration"]["calibration_sha256"])):
        inputs["routing_capture"][key] = value
    inputs["routing_capture_sha256"] = identity_sha256(inputs["routing_capture"])
    preflight["operator"]["routing_capture_sha256"] = inputs["routing_capture_sha256"]
    rows = {}
    for name, row in old_rows.items():
        probe = copy.deepcopy(row["probe_identity"])
        probe.update(calibration_shape=[1, 2], calibration_sha256=inputs["probe_calibration"]["calibration_sha256"])
        operator = copy.deepcopy(row["joint_operator_identity"])
        operator["probe_identity_sha256"] = identity_sha256(probe)
        rows[name] = make_joint_aura_entry(operator_identity=operator, probe_identity=probe,
            signed_components=[{"weight": v, "activation": 0., "mixed": 0., "total": v} for v in (.1, -.2, .3)])
    panel = freeze_moe_panel(inputs, preflight, rows, cost_sha256="4" * 64)
    assert panel["calibration_sha256"] == inputs["probe_calibration"]["calibration_sha256"]
    assert panel["probe_scope"]["parent_calibration_sha256"] == inputs["calibration"]["calibration_sha256"]
    assert panel["probe_scope"]["scope"] == "first_sequence_integration_screen"
    inputs["probe_scope"]["sample_indices"] = [1]
    with pytest.raises(ValueError, match="first-sequence screen"):
        freeze_moe_panel(inputs, preflight, rows, cost_sha256="4" * 64)


def test_a_routed_activation_that_differs_at_all_is_refused(joined, tmp_path):
    """The measured fp4 value that USED to pass, on the routed panel.

    0.0078125 sat inside the old 0.015625 tolerance and was recorded as
    "passed" (#567). With a bit-identical stored scale it is at least one code
    flipped, so there is no tolerance that separates it from the 0.0957 and
    0.1436 that failed -- the activation gate has none.
    """
    panel, receipt, trace = receipt_fixture(joined)
    receipt["phases"]["prefill"]["qdq_numerics"]["max_abs_error"] = 0.0078125
    path, trace_path = tmp_path / "receipt.json", tmp_path / "trace.json"
    digest = write(path, receipt)
    write(trace_path, trace)
    with pytest.raises(ValueError, match="E2M1 code flipped"):
        consume_moe_receipt(path, expected_sha256=digest, expected_panel=panel,
                            memory_trace_path=trace_path)


def test_a_routed_gemm_output_keeps_its_tolerance(joined, tmp_path):
    """The OTHER gate is unchanged: it compares an accumulation, not a lookup."""
    panel, receipt, trace = receipt_fixture(joined)
    receipt["phases"]["prefill"]["numerics"]["max_abs_error"] = 0.0078125
    path, trace_path = tmp_path / "receipt.json", tmp_path / "trace.json"
    digest = write(path, receipt)
    write(trace_path, trace)
    consume_moe_receipt(path, expected_sha256=digest, expected_panel=panel,
                        memory_trace_path=trace_path)


def test_a_world_of_one_binds_no_separate_full_quality_preparation(joined):
    """TP1 is unchanged: the container IS the cut, so there is one render.

    The rank-local quality gate exists because a rank holds a CUT of the
    module's render. At a world of one there is nothing to cut, so the panel
    carries no second preparation and the joint quality row names the same
    render the native member roster does.
    """
    inputs, preflight, rows = joined
    assert inputs["shape"].get("tensor_parallel", 1) == 1
    panel = freeze_moe_panel(inputs, preflight, rows, cost_sha256="4" * 64)
    assert "quality_preparation" not in panel
    member = panel["members"][0]
    assert "quality_rendered_weight" not in member and "rank_render_proof" not in member
    joint = rows[member["unit"]]["joint_operator_identity"]
    assert joint["rendered_weight"] == member["rendered_weight"]


def test_late_whole_owner_binding_preserves_router_and_member_execution(joined):
    from prismaquant.native_moe_execution_binding import (
        execution_panel_from_joint,bind_execution_receipt,resolve_execution_binding,RAW_RECEIPT_SCHEMA)
    final=freeze_moe_panel(*joined,cost_sha256='4'*64)
    execution=execution_panel_from_joint(final)
    raw={'schema':RAW_RECEIPT_SCHEMA,'status':'timing_admissible','panel':execution,
         'panel_sha256':identity_sha256(execution),'phases':{'fixture':'not GPU evidence'}}
    bound=bind_execution_receipt(raw,final)
    assert bound['raw_receipt']==raw
    assert resolve_execution_binding(bound,final)['panel']==final
    for field in ('routing_capture_sha256','calibration_sha256','source_sha256'):
        changed=copy.deepcopy(final);changed[field]='0'*64
        with pytest.raises(ValueError,match='changes measured execution'):
            bind_execution_receipt(raw,changed)
    changed=copy.deepcopy(final);changed['members'][0]['rendered_weight']['content_sha256']='0'*64
    with pytest.raises(ValueError,match='changes measured execution'):
        bind_execution_receipt(raw,changed)


def test_packed_reference_uses_glm_apply_gate_without_inventing_act_fn():
    from prismaquant.measure_quant_cost import _packed_experts_forward_with_weights
    class GlmLike(torch.nn.Module):
        num_experts=1
        def _apply_gate(self,value):
            gate,up=value.chunk(2,dim=-1)
            return torch.nn.functional.silu(gate.clamp(max=.5))*up.clamp(-1,1)
    x=torch.ones(1,4)
    gate_up=torch.cat((torch.eye(4),2*torch.eye(4))).unsqueeze(0)
    out=_packed_experts_forward_with_weights(GlmLike(),x,torch.zeros(1,1,dtype=torch.long),
        torch.ones(1,1),gate_up,torch.eye(4).unsqueeze(0))
    assert torch.equal(out,torch.full((1,4),torch.nn.functional.silu(torch.tensor(.5)).item()))


# ---------------------------------------------------------------------------
# #1565: Tessera-published preflight, workspace and receipt fields are read
# under the #1548 rule -- an additive field is accepted, a consumed field is
# required, and a producer's must_understand mark refuses.
# ---------------------------------------------------------------------------
MU = "must_understand"
EMPTY_SLOT = {"index": 1, "allocation": None}


def _restamp(preflight):
    preflight["runtime_sha256"] = identity_sha256(preflight["runtime"])
    preflight["workspace_sha256"] = identity_sha256(preflight["workspace"])
    return preflight


def test_additive_preflight_and_workspace_fields_freeze_the_same_panel_facts(joined):
    base = freeze_moe_panel(*joined, cost_sha256="4" * 64)
    inputs, preflight, rows = copy.deepcopy(joined)
    preflight["producer_note"] = "prose"
    preflight["operator"]["compile_note"] = "prose"
    preflight["runtime"]["execution"]["cuda_graphs"] = False
    preflight["workspace"]["allocator"] = "caching"
    preflight["workspace"]["slots"][0]["alignment"] = 256
    preflight["workspace"]["slots"].append({**EMPTY_SLOT, "reserved_for": "prefill"})
    panel = freeze_moe_panel(inputs, _restamp(preflight), rows, cost_sha256="4" * 64)
    assert panel["execution"] == base["execution"]
    assert panel["runtime_binding"] == base["runtime_binding"]
    assert panel["workspace"]["resident_bytes"] == base["workspace"]["resident_bytes"]


def test_an_empty_slot_that_carries_an_allocation_geometry_is_still_refused(joined):
    inputs, preflight, rows = copy.deepcopy(joined)
    preflight["workspace"]["slots"].append({**EMPTY_SLOT, "storage_bytes": 64})
    with pytest.raises(ValueError, match="empty workspace slot"):
        freeze_moe_panel(inputs, _restamp(preflight), rows, cost_sha256="4" * 64)


@pytest.mark.parametrize("where", ["preflight", "operator", "execution", "workspace", "slot", "empty_slot"])
def test_a_must_understand_preflight_field_is_refused(joined, where):
    inputs, preflight, rows = copy.deepcopy(joined)
    preflight["workspace"]["slots"].append(dict(EMPTY_SLOT))
    target = {"preflight": preflight, "operator": preflight["operator"],
              "execution": preflight["runtime"]["execution"], "workspace": preflight["workspace"],
              "slot": preflight["workspace"]["slots"][0],
              "empty_slot": preflight["workspace"]["slots"][1]}[where]
    target["new_axis"] = 1
    target[MU] = ["new_axis"]
    with pytest.raises(ValueError, match="must-understand"):
        freeze_moe_panel(inputs, _restamp(preflight), rows, cost_sha256="4" * 64)


def test_a_missing_consumed_workspace_field_is_a_refusal_not_a_key_error(joined):
    inputs, preflight, rows = copy.deepcopy(joined)
    del preflight["workspace"]["resident_bytes"]
    with pytest.raises(ValueError, match="missing field"):
        freeze_moe_panel(inputs, _restamp(preflight), rows, cost_sha256="4" * 64)


def _moe_receipt_targets(receipt):
    phase = receipt["phases"]["decode"]
    return {"receipt": receipt, "operator": receipt["operator"], "resources": receipt["resources"],
            "phase": phase, "route": phase["route"]}


def test_an_additive_receipt_field_gives_the_same_observation(joined, tmp_path):
    panel, receipt, trace = receipt_fixture(joined)
    trace_path = tmp_path / "trace.json"
    write(trace_path, trace)
    path = tmp_path / "receipt.json"
    base_sha = write(path, receipt)
    base = consume_moe_receipt(path, expected_sha256=base_sha,
                               expected_panel=panel, memory_trace_path=trace_path)
    receipt = copy.deepcopy(receipt)
    for name, target in _moe_receipt_targets(receipt).items():
        target[f"added_{name}"] = {"schema": "tessera.future.v1"}
    new_sha = write(path, receipt)
    observed = consume_moe_receipt(path, expected_sha256=new_sha,
                                   expected_panel=panel, memory_trace_path=trace_path)
    # Same path, so the only receipt-derived difference is the file's digest.
    assert json.loads(json.dumps(observed).replace(new_sha, base_sha)) == json.loads(json.dumps(base))


@pytest.mark.parametrize("where", ["receipt", "operator", "resources", "phase", "route"])
def test_a_must_understand_receipt_field_is_refused(joined, tmp_path, where):
    panel, receipt, trace = receipt_fixture(joined)
    receipt = copy.deepcopy(receipt)
    target = _moe_receipt_targets(receipt)[where]
    target["new_axis"] = 1
    target[MU] = ["new_axis"]
    path, trace_path = tmp_path / "receipt.json", tmp_path / "trace.json"
    write(trace_path, trace)
    with pytest.raises(ValueError, match="must-understand"):
        consume_moe_receipt(path, expected_sha256=write(path, receipt),
                            expected_panel=panel, memory_trace_path=trace_path)


def original_raw_entry_metadata(layer=3):
    """CPU tensor protocol fixture, not original source or CUDA qualification."""
    tensors = {
        "inputs": torch.ones(512, 4096, dtype=torch.bfloat16),
        "top_k_index": torch.arange(8).expand(512, 8).contiguous(),
        "top_k_weights": torch.full((512, 8), 2.5 / 8, dtype=torch.bfloat16),
        "coordinates": torch.stack((torch.zeros(512, dtype=torch.int64), torch.arange(512)), dim=1),
        "expert_bias": torch.zeros(288, dtype=torch.float32),
    }
    return {"unit": f"model.language_model.layers.{layer}.mlp.experts",
            "profile_role_order": list(ROLES),
            "tensors": {name: tensor_id(value) for name, value in tensors.items()}}, tensors


def test_original_entry_retains_raw_source_dtypes_and_frozen_tensor_identity():
    from prismaquant.native_moe_panel import original_capture_entry, _validate_original_capture_entry

    metadata, tensors = original_raw_entry_metadata()
    entry = original_capture_entry(metadata)
    assert _validate_original_capture_entry(entry, metadata) == entry
    assert entry["sample"] == 0 and entry["positions"] == list(range(512))
    assert entry["tensors"]["top_k_index"] == tensor_id(tensors["top_k_index"])
    assert entry["tensors"]["top_k_weights"] == tensor_id(tensors["top_k_weights"])
    metadata["tensors"]["inputs"]["content_sha256"] = "0" * 64
    assert entry["tensors"]["inputs"]["content_sha256"] != "0" * 64
    assert "file_sha256" not in entry and "complete_checkpoint" not in entry


@pytest.mark.parametrize("change", ["layer", "boolean_layer", "sample", "boolean_sample", "positions",
                                    "boolean_position", "role", "raw_schema", "entry_hash", "extra"])
def test_original_entry_refuses_coordinate_role_hash_or_schema_drift(change):
    from prismaquant.native_moe_panel import original_capture_entry, _validate_original_capture_entry

    metadata, _ = original_raw_entry_metadata()
    entry = original_capture_entry(metadata)
    if change == "layer":
        entry["layer"] = 8
    elif change == "boolean_layer":
        entry["layer"] = True
    elif change == "sample":
        entry["sample"] = 1
    elif change == "boolean_sample":
        entry["sample"] = False
    elif change == "positions":
        entry["positions"][8] = 9
    elif change == "boolean_position":
        entry["positions"][0] = False
    elif change == "role":
        entry["profile_role_order"] = ["w3", "w1", "w2"]
    elif change == "raw_schema":
        entry["raw_boundary_schema"] = "prismaquant.routed_boundary_capture.v1"
    elif change == "entry_hash":
        entry["tensors"]["expert_bias"]["content_sha256"] = "0" * 64
    else:
        entry["complete_capture"] = True
    with pytest.raises(ValueError):
        _validate_original_capture_entry(entry, metadata)


@pytest.mark.parametrize("name", ["inputs", "top_k_index", "top_k_weights", "coordinates", "expert_bias"])
@pytest.mark.parametrize("change", ["shape", "dtype", "bytes", "hash", "extra"])
def test_original_entry_requires_closed_original_tensor_identities(name, change):
    from prismaquant.native_moe_panel import original_capture_entry

    metadata, _ = original_raw_entry_metadata()
    record = metadata["tensors"][name]
    if change == "shape":
        record["shape"] = [1]
    elif change == "dtype":
        record["dtype"] = "torch.float16"
    elif change == "bytes":
        record["logical_bytes"] = 0
    elif change == "hash":
        record["content_sha256"] = "bad"
    else:
        record["converted"] = True
    with pytest.raises(ValueError):
        original_capture_entry(metadata)


def bound_protocol_document(tmp_path, name, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    path = tmp_path / name
    path.write_bytes(raw)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def original_protocol_case(tmp_path, layer=3):
    """Isolated SYNTHETIC metadata/completion fixture, never an actual receipt.

    Native descriptors, claims and CUDA completion rows below are deliberately
    synthetic parser inputs. No real owner receipt is altered or presented as
    CUDA proof. Qualification/root admission are absent, so this record cannot
    authorize a launch. Actual CPU owner receipts are tested separately.
    """
    from prismaquant.io_engine import _SEALS
    from prismaquant.native_moe_panel import original_capture_entry
    from prismaquant.residency_map import residency_map_key
    from test_native_prefix_intake import fixture

    metadata, tensors = original_raw_entry_metadata(layer)
    prefix, _ = fixture(layer)
    names = ['lm_head.weight', 'model.language_model.embed_tokens.weight', 'model.language_model.norm.weight']
    names += [f'model.language_model.layers.{i}.norm.weight' for i in range(45)]
    checkpoint = {name: ('lookahead.safetensors' if '.layers.44.' in name else 'current.safetensors') for name in names}
    mapping = {name: name for name in names}
    config = {'model_type': 'glm5_next', 'synthetic_protocol_fixture': True}
    producer = {'files': {'current.safetensors': '5' * 64, 'lookahead.safetensors': '6' * 64},
        'auxiliary_sha256': {'model.safetensors.index.json': '8' * 64},
        'config_sha256': '7' * 64, 'tensors': checkpoint}
    model_root = tmp_path / 'synthetic-source'
    paths = {name: str(model_root / name) for name in
             ('config.json', 'model.safetensors.index.json', 'current.safetensors', 'lookahead.safetensors')}
    digests = {**producer['files'], **producer['auxiliary_sha256'], 'config.json': producer['config_sha256']}
    readset = {'schema': 'prismaquant.prismabuild.data_manifest.v1',
        'produced_by': {'synthetic_protocol_fixture': True}, 'mount_prefix': str(tmp_path),
        'entries': [{'path': paths[name], 'offset': 0, 'bytes': 128, 'sha256': digest}
                    for name, digest in sorted(digests.items())],
        'entry_count': len(digests), 'total_bytes': 128 * len(digests), 'annotations': {}}
    publisher = {'id': 'SYNTHETIC/parser-control', 'sha': 'a' * 40,
        'siblings': [{'rfilename': name, 'size': 128, 'blobId': 'b' * 40,
                      **({'lfs': {'sha256': digest, 'size': 128}} if name.endswith('.safetensors') else {})}
                     for name, digest in sorted(digests.items())]}
    source_value = {'config': config, 'weight_map': mapping, 'checkpoint_weight_map': checkpoint,
        'shards': [{'path': paths[name], 'size': 128, 'sha256': digest} for name, digest in sorted(producer['files'].items())]}
    source = {'schema': 'prismaquant.streamed_model.identity.v1', 'source': str(model_root),
              'resolved_commit': None, 'content_sha256': identity_sha256(source_value), **source_value}
    contract = prefix['model_load_contract']
    contract['source_map_sha256'] = identity_sha256({name: {'tensor': name, 'file': checkpoint[name]} for name in names})
    calibration_ids = torch.zeros(512, 512, dtype=torch.int64)
    calibration = {'schema': 'prismaquant.calibration_input.v1', 'artifact_sha256': '9' * 64,
        'calibration_sha256': tensor_id(calibration_ids)['content_sha256'], 'shape': [512, 512], 'dtype': 'torch.int64',
        'provenance': {'nsamples': 512, 'seqlen': 512, 'fit_tokens': 262144, 'fit_tokens_min': 1,
            'fit_ids_sha256': hashlib.sha256(calibration_ids.int().numpy().tobytes()).hexdigest(),
            'text_sha256': 'a' * 64, 'model': str(model_root), 'seed': 0,
            'source': 'SYNTHETIC/parser-control', 'split_role': 'calibration'}}
    runtime = {'schema': 'prismaquant.original_source_runtime.v1',
        'prismaquant_source_sha256': 'b' * 64, 'tessera_source_sha256': 'c' * 64,
        'modeling_source': {'path': '/synthetic/modeling_glm5_next.py', 'sha256': 'd' * 64},
        'model_class': contract['model_class'], 'profile': 'glm5_next', 'config': config,
        'versions': {'python': 'synthetic-python', 'torch': 'synthetic-torch', 'torch_git': None,
                     'cuda': 'synthetic-cuda', 'transformers': contract['transformers_version']},
        'container_content_sha256': None,
        'arithmetic': {'matmul_precision': 'highest', 'allow_tf32': False,
                       'allow_bf16_reduced_precision_reduction': True},
        'material_pipeline': {'decoder': 'safetensors.safe_open', 'framework': 'pt', 'decoder_device': 'cpu',
                             'cast_owner': 'prismaquant.layer_streaming', 'direct_gpu_decode': False,
                             'target_dtype': 'torch.bfloat16', 'tensor_dtypes': {}, 'scale_inv_map': {}},
        'prismabuild': {'sdk_version': 4, 'helper_root': '/synthetic-sdk4', 'runtime_generation': 'synthetic-sdk4',
                       'source_tree': {'package_sha256': 'e' * 64, 'helper_tree_sha256': None}}}
    resources = {'schema': 'prismaquant.original_source_resources.v1', 'material_bytes': 1024,
        'cpu_bytes': 128 * 1024**2, 'source_cache_bytes': 1024, 'copy_bytes': 1024, 'gpu_bytes': 0,
        'native_bytes': 0, 'serialization_bytes': 8 * 1024**2, 'artifact_bytes': 16 * 1024**2,
        'deadline_seconds': 60, 'stall_seconds': 10, 'host_floor_bytes': 16 * 1024**3,
        'margin_bytes': 0, 'claim_demand': {'cpu': 1, 'mem_gb': 1},
        'source_prefetch': {'max_cache_slots': 2, 'prefetch_workers': 1, 'prefetch_lookahead': 1,
                            'cache_headroom_gb': 1, 'prefetch_min_available_gb': 1,
                            'require_prefetched_residency': True}}
    execution = {'schema': 'prismaquant.joint_aura.source_execution.v1',
                 'modules': {'': {'attention': 'eager'}, metadata['unit']: {'experts': 'grouped_mm'}}}
    session = {'generation': 'f' * 32, 'run_identity_sha256': '0' * 64}
    authority = {'schema': 'prismaquant.original_source_authority.v1',
        'scope': 'original_text_source_diagnostic_and_first_sequence_routed_capture',
        'publisher': {'id': 'SYNTHETIC/parser-control', 'revision': 'a' * 40,
                      'input': bound_protocol_document(tmp_path, 'publisher.json', publisher)},
        'producer': bound_protocol_document(tmp_path, 'producer.json', producer),
        'source_paths': bound_protocol_document(tmp_path, 'paths.json', paths),
        'readset': bound_protocol_document(tmp_path, 'readset.json', readset),
        'runtime': bound_protocol_document(tmp_path, 'runtime.json', runtime),
        'resources': bound_protocol_document(tmp_path, 'resources.json', resources),
        'qualification': None, 'root_admission': None, 'calibration': calibration,
        'source_model_identity': source, 'source_execution': execution, 'session': session}
    authority_input = bound_protocol_document(tmp_path, 'authority.json', authority)
    route = glm_routing()
    route['topk_ids_dtype'] = 'torch.int64'
    route['topk_weights_dtype'] = 'torch.bfloat16'
    namespace = runtime['model_class'].rsplit('.', 1)[0]
    route['source_protocol']['router_class'] = namespace + '.Glm5NextTextTopkRouter'
    route['source_protocol']['correction_bias'] = {key: tensor_id(tensors['expert_bias'])[key]
                                                  for key in ('content_sha256', 'dtype')}
    metadata.update(schema='prismaquant.native_moe_raw_boundary.v1', shape=glm_shape(), routing=route,
        producer_source=producer, runtime_config=config, model_load_contract=contract,
        replay=prefix['replay'], source_execution=execution, capture_source_sha256='1' * 64,
        attention_implementation='eager',
        capture_runtime={name: runtime['versions'][name] for name in ('torch', 'cuda', 'transformers')},
        calibration_sha256=calibration['calibration_sha256'], calibration_shape=calibration['shape'],
        calibration_dtype=calibration['dtype'],
        scope='first calibration sequence; decode uses its first row, not autoregressive generation')
    claim = {'queue_root': '/synthetic-queue', 'action_key': '2' * 64, 'nonce': 'synthetic-producer-nonce',
        'scope_id': 'synthetic-scope', 'worker': 'synthetic-worker', 'host': 'synthetic-host',
        'incarnation': 'synthetic-worker', 'attempt_source': 'launch-env', 'map_path': '/synthetic-map',
        'helper_root': '/synthetic-sdk4'}
    deliveries = []
    for index, (name, digest) in enumerate(sorted(digests.items()), 1):
        key = residency_map_key(paths[name], 0)
        deliveries.append({'name': name, 'sha256': digest, 'bytes_hashed': 128, 'delivery_index': 1,
            'held': name == 'lookahead.safetensors', 'material_windows': int(name == 'lookahead.safetensors'),
            'readers': 0, 'storage_aliases': int(name == 'lookahead.safetensors'), 'payload_reads': 1,
            'native_delivery': {'claim': copy.deepcopy(claim), 'ref_id': 'synthetic-ref-' + name,
                'serving': {'tier_id': 'synthetic-tier', 'epoch': '', 'pin_id': 'synthetic-pin-' + name, 'range_ref': key},
                'entry': {'key': key, 'stage_path': paths[name], 'bytes': 128, 'sha256': digest,
                    'file_id': {'ino': index, 'size': 128, 'mtime_ns': 1, 'ctime_ns': 1},
                    'mover_action_key': '3' * 64, 'generation': 'synthetic-delivery'},
                'source_fd_stat': [1, index, 128, 1, 1], 'sealed_fd_stat': [2, index, 128, 1, 1],
                'kernel_seals': _SEALS, 'descriptors_closed': True, 'lease_released': True}})
    generations = {row['name']: row['delivery_index'] for row in deliveries}
    completed = {'device': 'cuda:0', 'stream_id': 1, 'files': {'current.safetensors': generations['current.safetensors']},
        'fence': 'cuda_event_synchronize', 'failed': False, 'retained_host_aliases': 0, 'host_aliases_at_fence': 1}
    pending = {**completed, 'stream_id': 2, 'files': {'lookahead.safetensors': generations['lookahead.safetensors']},
               'fence': None, 'retained_host_aliases': 1}
    if layer == 44:
        completed['files']['lookahead.safetensors'] = generations['lookahead.safetensors']
    material = {'schema': 'prismaquant.original_source_material.v1',
        'publisher_control_sha256': authority['publisher']['input']['sha256'],
        'publisher_id': authority['publisher']['id'], 'publisher_revision': authority['publisher']['revision'],
        'readset_sha256': authority['readset']['sha256'], 'automatic_capture_qualified': False,
        'authentication': 'independent publisher/readset SHA256 and native Git auxiliary objects; kernel-sealed whole-file delivery',
        'verified_files': [{'name': name, 'sha256': producer['files'][name], 'bytes_hashed': 128} for name in sorted(producer['files'])],
        'auxiliary_verified': sorted(producer['auxiliary_sha256']) + ['config.json'],
        'material_live_bytes': 128, 'material_limit_bytes': 1024, 'deliveries': deliveries,
        'copy_completions': [completed], 'pending_copy_completions': [] if layer == 44 else [pending]}
    material['auxiliary_verified'].sort()
    metadata['source_acquisition'] = {'schema': 'prismaquant.original_source_acquisition.v1',
        'authority': authority_input, 'session': session, 'source_material': material,
        'source_initialization': copy.deepcopy(contract), 'source_execution': execution,
        'runtime': {'schema': 'prismaquant.original_routed_capture_runtime.v1', 'source_runtime': runtime,
            'device': 'cuda:0', 'source_tensor_dtypes': {name: record['dtype'] for name, record in metadata['tensors'].items()},
            'source_tensor_devices': {name: ('cpu' if name == 'coordinates' else 'cuda:0') for name in metadata['tensors']},
            'expert_class': namespace + '.Glm5NextTextExperts', 'router_class': route['source_protocol']['router_class'],
            'router_source_sha256': route['source_protocol']['router_source_sha256']},
        'entry': original_capture_entry(metadata)}
    manifest = {'schema': 'prismaquant.first_sequence_original_capture.v1', 'scope': 'first_sequence_original_capture',
                'authority': authority_input, 'session': session, 'calibration': calibration}
    return {'payload': {'source': 'routed_boundary_capture', 'boundary_metadata': metadata, **tensors},
            'calibration': calibration, 'manifest': manifest, 'source': source, 'authority': authority,
            'authority_input': authority_input, 'session': session}


def original_protocol_intake(case):
    from prismaquant.native_moe_panel import routed_boundary_inputs

    return routed_boundary_inputs(case['payload'], calibration_receipt=case['calibration'],
        capture_manifest=case['manifest'], device='cpu', source_model_identity=case['source'],
        expected_original_authority=case['authority_input'], expected_original_session=case['session'],
        original_source_authority=case['authority'])


@pytest.mark.parametrize('layer', [3, 43, 44])
def test_synthetic_original_scoped_transport_retains_all_original_identity_and_lookahead_debt(tmp_path, layer):
    case = original_protocol_case(tmp_path, layer)
    original = copy.deepcopy(case['payload']['boundary_metadata']['source_acquisition'])
    routed, phases, bias = original_protocol_intake(case)
    assert routed['source_acquisition'] == original
    assert routed['tensors'] == original['entry']['tensors']
    assert phases['prefill']['source_topk_ids'].dtype == torch.int64
    assert phases['prefill']['topk_ids'].dtype == torch.int32
    assert phases['prefill']['source_topk_weights'].dtype == torch.bfloat16
    assert phases['prefill']['topk_weights'].dtype == torch.float32
    assert torch.equal(phases['prefill']['topk_weights'], case['payload']['top_k_weights'].float())
    assert phases['decode']['input'].shape == (1, 4096) and bias.shape == (288,)
    assert case['manifest']['schema'] != 'prismaquant.tessera_calibration_cache.v2'
    assert 'status' not in case['manifest'] and 'source_cache_reuse' not in routed
    assert case['authority']['qualification'] is None and case['authority']['root_admission'] is None
    assert routed['source_acquisition']['source_material']['pending_copy_completions'] == original['source_material']['pending_copy_completions']


@pytest.mark.parametrize('damage', ['missing-checkpoint', 'extra-checkpoint', 'same-cardinality-replacement'])
def test_original_prefix_requires_exact_checkpoint_roster_after_valid_initialization(tmp_path, damage):
    from prismaquant.native_moe_panel import original_capture_entry
    from prismaquant.streaming_initialization import validate_streaming_prefix_initialization_contract

    case = original_protocol_case(tmp_path, 3)
    metadata = case['payload']['boundary_metadata']
    contract = metadata['model_load_contract']
    state = contract['state']
    name = 'model.language_model.layers.3.norm.weight'
    if damage == 'missing-checkpoint':
        state[name] = {'shape': [4], 'dtype': 'torch.bfloat16',
                       'kind': 'derived_buffer', 'sha256': 'e' * 64}
    elif damage == 'extra-checkpoint':
        state['model.language_model.layers.3.extra.weight'] = dict(state[name])
    else:
        state['model.language_model.layers.3.replacement.weight'] = state.pop(name)
    contract['persistent_tensors'] = sum(row['kind'] == 'checkpoint' for row in state.values())
    contract['derived_buffers'] = sum(row['kind'] == 'derived_buffer' for row in state.values())
    contract['state_sha256'] = identity_sha256(state)
    assert validate_streaming_prefix_initialization_contract(contract) == contract
    metadata['source_acquisition']['source_initialization'] = copy.deepcopy(contract)
    metadata['source_acquisition']['entry'] = original_capture_entry(metadata)
    with pytest.raises(ValueError, match='actual original prefix checkpoint coverage'):
        original_protocol_intake(case)


@pytest.mark.parametrize('change', ['authority', 'session', 'source', 'map', 'calibration', 'text', 'device',
    'coordinates_device', 'bias_device', 'class', 'cast', 'runtime', 'entry_hash', 'raw_hash', 'bias',
    'head', 'prefix', 'layer44_prefix', 'mixed_dev', 'canonical', 'full_scope', 'missing_completion',
    'completion_generation', 'completion_fence', 'completion_failure', 'header_only'])
def test_synthetic_original_intake_rejects_wrong_independent_provenance_before_transport(tmp_path, change):
    case = original_protocol_case(tmp_path, 44 if change == 'layer44_prefix' else 3)
    metadata = case['payload']['boundary_metadata']
    acquisition = metadata['source_acquisition']
    if change == 'authority':
        case['authority_input'] = {'path': '/different-authority', 'sha256': '0' * 64}
    elif change == 'session':
        case['session'] = {**case['session'], 'generation': '0' * 32}
    elif change == 'source':
        case['source']['content_sha256'] = '0' * 64
    elif change == 'map':
        metadata['model_load_contract']['source_map_sha256'] = '0' * 64
    elif change in ('calibration', 'text'):
        case['calibration'] = copy.deepcopy(case['calibration'])
        if change == 'calibration':
            case['calibration']['shape'] = [1, 512]
        else:
            case['calibration']['provenance']['text_sha256'] = '0' * 64
    elif change == 'device':
        acquisition['runtime']['device'] = 'cuda'
    elif change == 'coordinates_device':
        acquisition['runtime']['source_tensor_devices']['coordinates'] = 'cuda:0'
    elif change == 'bias_device':
        acquisition['runtime']['source_tensor_devices']['expert_bias'] = 'cpu'
    elif change == 'class':
        acquisition['runtime']['expert_class'] = 'fixture.OtherExperts'
    elif change == 'cast':
        acquisition['runtime']['source_runtime']['material_pipeline']['decoder_device'] = 'cuda:0'
    elif change == 'runtime':
        acquisition['runtime']['source_runtime']['arithmetic']['allow_tf32'] = True
    elif change == 'entry_hash':
        acquisition['entry']['tensors']['inputs']['content_sha256'] = '0' * 64
    elif change == 'raw_hash':
        case['payload']['inputs'][0, 0] = 7
    elif change == 'bias':
        metadata['routing']['source_protocol']['correction_bias']['content_sha256'] = '0' * 64
    elif change == 'head':
        metadata['model_load_contract']['head_state_names'].remove('lm_head.weight')
    elif change == 'prefix':
        metadata['model_load_contract']['observed_layers'].pop()
    elif change == 'layer44_prefix':
        from test_native_prefix_intake import fixture
        metadata['model_load_contract'] = fixture(43)[0]['model_load_contract']
    elif change == 'mixed_dev':
        metadata['dev_uncertified'] = True
    elif change == 'canonical':
        case['manifest'] = {'schema': 'prismaquant.tessera_calibration_cache.v2', 'status': 'complete', 'identity': {}}
    elif change == 'full_scope':
        metadata['scope'] = 'complete 512-row draw, H and prices'
    elif change == 'missing_completion':
        acquisition['source_material']['copy_completions'] = []
    elif change == 'completion_generation':
        acquisition['source_material']['copy_completions'][0]['files']['current.safetensors'] = 999
    elif change == 'completion_fence':
        acquisition['source_material']['copy_completions'][0]['fence'] = None
    elif change == 'completion_failure':
        acquisition['source_material']['copy_completions'][0]['failed'] = True
    else:
        row = next(row for row in acquisition['source_material']['deliveries'] if row['name'] == 'current.safetensors')
        row['payload_reads'] = 0
    with pytest.raises((ValueError, RuntimeError, OSError)):
        original_protocol_intake(case)


@pytest.mark.parametrize('missing', ['authority', 'session', 'source', 'normalized_authority'])
def test_original_payload_cannot_supply_its_own_independent_expectations(tmp_path, missing):
    from prismaquant.native_moe_panel import routed_boundary_inputs

    case = original_protocol_case(tmp_path)
    kwargs = {'source_model_identity': case['source'], 'expected_original_authority': case['authority_input'],
              'expected_original_session': case['session'], 'original_source_authority': case['authority']}
    key = {'authority': 'expected_original_authority', 'session': 'expected_original_session',
           'source': 'source_model_identity', 'normalized_authority': 'original_source_authority'}[missing]
    kwargs[key] = None
    with pytest.raises(ValueError):
        routed_boundary_inputs(case['payload'], calibration_receipt=case['calibration'],
                               capture_manifest=case['manifest'], device='cpu', **kwargs)


@pytest.mark.parametrize('where', ['acquisition', 'entry', 'runtime', 'manifest'])
def test_original_nested_records_are_closed_not_legacy_additive_metadata(tmp_path, where):
    case = original_protocol_case(tmp_path)
    acquisition = case['payload']['boundary_metadata']['source_acquisition']
    record = {'acquisition': acquisition, 'entry': acquisition['entry'],
              'runtime': acquisition['runtime'], 'manifest': case['manifest']}[where]
    record['allow_partial'] = True
    with pytest.raises((ValueError, RuntimeError)):
        original_protocol_intake(case)


def test_original_prior_completed_delivery_is_not_overwritten_by_new_pending_same_file(tmp_path):
    case = original_protocol_case(tmp_path)
    material = case['payload']['boundary_metadata']['source_acquisition']['source_material']
    old = next(row for row in material['deliveries'] if row['name'] == 'current.safetensors')
    current = copy.deepcopy(old)
    current.update(delivery_index=old['delivery_index'] + 1, held=True,
                   material_windows=1, storage_aliases=1, payload_reads=1)
    current['native_delivery']['ref_id'] += '-new'
    current['native_delivery']['serving']['pin_id'] += '-new'
    material['deliveries'].append(current)
    material['deliveries'].sort(key=lambda row: (row['name'], row['delivery_index']))
    material['material_live_bytes'] += current['bytes_hashed']
    material['pending_copy_completions'].append({'device': 'cuda:0', 'stream_id': 7,
        'files': {'current.safetensors': current['delivery_index']}, 'fence': None, 'failed': False,
        'retained_host_aliases': 1, 'host_aliases_at_fence': 0})
    frozen = copy.deepcopy(material)
    capture, _, _ = original_protocol_intake(case)
    assert capture['source_acquisition']['source_material'] == frozen
    assert len(capture['source_acquisition']['source_material']['deliveries']) == len(frozen['deliveries'])
    assert capture['source_acquisition']['source_material']['copy_completions'][0]['files']['current.safetensors'] == old['delivery_index']


@pytest.mark.parametrize('change', ['schema', 'attention', 'experts'])
def test_original_source_execution_cannot_inherit_derivative_or_different_dispatch(tmp_path, change):
    case = original_protocol_case(tmp_path)
    metadata = case['payload']['boundary_metadata']
    metadata['source_acquisition']['source_execution'] = copy.deepcopy(metadata['source_execution'])
    execution = metadata['source_acquisition']['source_execution']
    if change == 'schema':
        execution['schema'] = 'prismaquant.joint_aura.source_execution.v2'
        execution['source_derivative'] = {'synthetic': True}
    elif change == 'attention':
        execution['modules']['']['attention'] = 'sdpa'
    else:
        execution['modules'][metadata['unit']]['experts'] = 'different_dispatch'
    with pytest.raises((ValueError, RuntimeError)):
        original_protocol_intake(case)


def test_native_source_execution_owner_keeps_isinstance_and_unsorted_json_policy():
    from prismaquant import joint_aura, native_moe_panel

    class Selector(str):
        pass

    value = {'schema': 'prismaquant.joint_aura.source_execution.v1', 'modules': {
        '': {'attention': Selector('eager'), 'experts': Selector('grouped_mm')},
        Selector('u'): {'attention': Selector('eager'), 'experts': Selector('grouped_mm')},
        'extra': {'attention': {2: 'café', 'z': '\ud800'}}}}
    assert native_moe_panel.require_native_source_execution is joint_aura.require_native_source_execution
    assert joint_aura.require_native_source_execution(value, unit='u') is value
    # Mixed leaf keys retain the native reader's unsorted JSON acceptance; an
    # Original control separately refuses this selector's non-string leaf key.
    assert json.dumps(value, allow_nan=False).encode('utf-8')


@pytest.mark.parametrize(('value', 'message'), [
    (None, 'native MoE requires explicit source execution identity'),
    ({'schema': 'prismaquant.joint_aura.source_execution.v2', 'modules': {}},
     'native MoE requires explicit source execution identity'),
    ({'schema': 'prismaquant.joint_aura.source_execution.v1', 'modules': {}},
     'native MoE requires explicit source execution identity'),
    ({'schema': 'prismaquant.joint_aura.source_execution.v1', 'modules': {2: {'attention': 'eager'}}},
     'native MoE source execution selectors are malformed'),
    ({'schema': 'prismaquant.joint_aura.source_execution.v1', 'modules': {'u': {}}},
     'native MoE source execution selectors are malformed'),
    ({'schema': 'prismaquant.joint_aura.source_execution.v1', 'modules': {'u': {'unknown': 'eager'}}},
     'native MoE source execution selectors are malformed'),
    ({'schema': 'prismaquant.joint_aura.source_execution.v1', 'modules': {
        '': {'attention': 'eager', 'experts': 'grouped_mm'}}},
     'native MoE source execution lacks resolved root/target backends'),
    ({'schema': 'prismaquant.joint_aura.source_execution.v1', 'modules': {
        '': {'attention': 'sdpa', 'experts': 'grouped_mm'},
        'u': {'attention': 'eager', 'experts': 'grouped_mm'}}},
     'native MoE source execution lacks resolved root/target backends'),
    ({'schema': 'prismaquant.joint_aura.source_execution.v1', 'modules': {
        '': {'attention': 'eager', 'experts': 'grouped_mm'},
        'u': {'attention': 'eager', 'experts': None}}},
     'native MoE source execution lacks resolved root/target backends'),
])
def test_native_source_execution_owner_keeps_exact_envelope_and_backend_refusals(value, message):
    from prismaquant.joint_aura import require_native_source_execution

    with pytest.raises(ValueError) as caught:
        require_native_source_execution(value, unit='u')
    assert str(caught.value) == message
    assert caught.value.__cause__ is None


def test_native_source_execution_owner_still_requires_dictionary_envelope():
    from types import MappingProxyType
    from prismaquant.joint_aura import require_native_source_execution

    value = MappingProxyType({'schema': 'prismaquant.joint_aura.source_execution.v1',
        'modules': {'': {'attention': 'eager', 'experts': 'grouped_mm'},
                    'u': {'attention': 'eager', 'experts': 'grouped_mm'}}})
    with pytest.raises(ValueError) as caught:
        require_native_source_execution(value, unit='u')
    assert str(caught.value) == 'native MoE requires explicit source execution identity'


@pytest.mark.parametrize('leaf', [float('nan'), float('inf'), float('-inf'), object()])
def test_native_source_execution_owner_keeps_strict_json_error(leaf):
    from prismaquant.joint_aura import require_native_source_execution

    value = {'schema': 'prismaquant.joint_aura.source_execution.v1', 'modules': {
        '': {'attention': 'eager', 'experts': 'grouped_mm'},
        'u': {'attention': 'eager', 'experts': 'grouped_mm'},
        'extra': {'attention': leaf}}}
    with pytest.raises((TypeError, ValueError)) as previous:
        json.dumps(value, allow_nan=False)
    with pytest.raises(type(previous.value)) as current:
        require_native_source_execution(value, unit='u')
    assert str(current.value) == str(previous.value)
    assert current.value.__cause__ is None
