import json
import pickle
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import torch

from prismaquant import allocator, digests, native_row_admission
from prismaquant.joint_aura import activation_identity, arithmetic_identity
from prismaquant.joint_aura_rows import joint_identity_sha256, make_joint_aura_entry
from prismaquant.layer_config import LAYER_CONFIG_META_KEY
from prismaquant.measured_runtime_prices import CONTEXT_SCHEMA, SCHEMA, identity_sha256
from prismaquant.transient_charge_boundary import BOUNDARY_V1


def _allocator_inputs(tmp_path, monkeypatch):
    unit, fmt, shape = "model.layers.0.self_attn.o_proj", "FP8_E4M3", [64, 64]
    spec = allocator.fr.get_format(fmt)
    source = {"config": {}, "weight_map": {"fixture.weight": "fixture.weight"},
              "shards": [{"path": "/fixture/fixture.weight", "size": 1, "sha256": "a" * 64}]}
    source_sha = identity_sha256(source)
    arithmetic = arithmetic_identity(torch.float32)
    probe = {"schema": "prismaquant.joint_aura.probes.v2", "seed_base": 0,
             "n_probes": 3, "calibration_sha256": "c" * 64,
             "producer_source_sha256": "d" * 64,
             "source_model": {"schema": "prismaquant.streamed_model.identity.v1",
                              "source": "synthetic", "resolved_commit": None,
                              "content_sha256": source_sha, **source},
             "distribution": "rademacher", "normalization": "global_kl_fisher",
             "temperature": 1.0, "arithmetic": arithmetic}
    operator = {"schema": "prismaquant.joint_aura.operator.v2", "qname": unit,
                "format": fmt, "probe_identity_sha256": joint_identity_sha256(probe),
                "source_weight": {"content_sha256": "a" * 64, "shape": shape,
                                  "dtype": "torch.float32", "logical_bytes": 16384},
                "rendered_weight": {"content_sha256": "b" * 64, "shape": shape,
                                    "dtype": "torch.float32", "logical_bytes": 16384},
                "activation": activation_identity(spec, {}, unit), "arithmetic": arithmetic}
    row = make_joint_aura_entry(
        operator_identity=operator, probe_identity=probe,
        signed_components=[dict(weight=1.0, activation=0.0, mixed=0.0, total=1.0)] * 3)
    probe_path, cost_path = tmp_path / "probe.pkl", tmp_path / "costs.pkl"
    probe_path.write_bytes(pickle.dumps({"stats": {unit: {"h_trace": 1.0,
        "n_params": 4096, "in_features": 64, "out_features": 64}}, "meta": {"model": None}}))
    cost_path.write_bytes(pickle.dumps({"costs": {unit: {fmt: row}}, "meta": {"formats": [fmt]},
        "provenance": {"cost_mode": "aura", "joint_activation": True,
                       "cost_currency": "joint_aura_predicted_dloss"}}))

    context = {"schema": CONTEXT_SCHEMA, "transient_charge_boundary": BOUNDARY_V1,
               "serving_context": {"platform": "sm_121", "structure": "dense",
                                   "residency": "resident", "execution_mode": "eager",
                                   "runtime_image": "fixture@sha256:" + "a" * 64},
               "gpu_identity": "synthetic", "runtime_sha256": "a" * 64,
               "source_sha256": source_sha, "calibration_sha256": "c" * 64,
               "prompt_tokens": 16, "batch_size": 1, "tensor_parallel": 1,
               "graph_mode": "eager", "operator_routes": {unit: {fmt: "synthetic"}}}
    receipt = tmp_path / "receipt.txt"
    receipt.write_text("The fixture has no GPU measurements.\n")
    receipt_sha = digests.bytes_sha256hex(receipt.read_bytes())
    fixed = dict(prefill_ms=0, decode_ms=0, serialized_bytes=0, resident_bytes=0,
                 activation_bytes=0, peak_scratch_bytes=0, kv_bytes=0)
    serialized, _, _ = allocator.serialized_candidate_payload(spec, shape, qname=unit)
    runtime_row = {"unit": unit, "format": fmt, "decode": None,
                   "binding": {"member_formats": {unit: fmt}, "member_shapes": {unit: shape},
                               "member_operator_identity_sha256": {
                                   unit: row["joint_operator_identity_sha256"]},
                               "operator_route": "synthetic"},
                   "resources": dict(fixed, prefill_ms=1, decode_ms=None,
                                     serialized_bytes=serialized, resident_bytes=16384),
                   "prefill": {"method": "cuda_events", "samples_ms": [1.0] * 3,
                               "warmup_iterations": 3, "receipt_path": receipt.name,
                               "receipt_sha256": receipt_sha}}
    now = datetime.now(timezone.utc)
    table = {"schema": SCHEMA, "table_id": "synthetic", "status": "proposal_data",
             "composition": "sequential_operator_sum", "context": context,
             "cost_sha256": digests.bytes_sha256hex(cost_path.read_bytes()),
             "measured_at": (now - timedelta(days=1)).isoformat(),
             "valid_until": (now + timedelta(days=1)).isoformat(), "fixed_assignment": {},
             "fixed_resources": fixed, "fixed_resources_receipt_path": receipt.name,
             "fixed_resources_receipt_sha256": receipt_sha, "rows": [runtime_row]}
    table_path, context_path = tmp_path / "runtime.json", tmp_path / "context.json"
    table_path.write_text(json.dumps(table))
    context_path.write_text(json.dumps(context))
    load = native_row_admission.load_measured_runtime_table

    def admit(*args, **kwargs):
        return replace(load(*args, **kwargs), native_rows_admitted=True,
                       fixed_resources_admitted=True, fixed_resources_refusal=None,
                       runtime_provenance={"path": "synthetic", "sha256": "0" * 64})

    monkeypatch.setattr(native_row_admission, "load_measured_runtime_table", admit)
    return ["--probe", str(probe_path), "--costs", str(cost_path), "--formats", fmt,
            "--allow-default-profile", "--target-bits", "9", "--pareto-targets", "9",
            "--pareto-csv", str(tmp_path / "pareto.csv"),
            "--pareto-output-dir", str(tmp_path / "seeds")], [
        "--measured-runtime-table", str(table_path), "--measured-runtime-context", str(context_path)]


def test_allocator_json_writes_use_insertion_ascii_indent2_lax(tmp_path, monkeypatch):
    argv, runtime = _allocator_inputs(tmp_path, monkeypatch)
    real_text = digests.JsonProfile.text
    expected = {}

    def observe(self, value):
        text = real_text(self, value)
        if self is digests.INSERTION_ASCII_INDENT2_LAX:
            if LAYER_CONFIG_META_KEY in value:
                name = "replay" if "prefill_frontier_replay" in value[LAYER_CONFIG_META_KEY] else "ordinary"
                lf = b""
            else:
                name, lf = "report", b"\n"
            expected[name] = marker.encode("ascii") + json.dumps(value, indent=2).encode("utf-8") + lf
            return marker + text
        return text

    def emit(sweep):
        assignment = sweep.solve(5.0, 9.0)["assignment"]
        sweep.emit_replay(5.0, 9.0, assignment, {"assignment_sha256": identity_sha256(assignment)})

    monkeypatch.setattr(digests.JsonProfile, "text", observe)
    # Add a space on the second run. Check that it reaches each file.
    for index, marker in enumerate(("", " ")):
        expected.clear()
        paths = {name: tmp_path / str(index) / f"{name}.json"
                 for name in ("ordinary", "report", "replay")}
        allocator.main([*argv, "--layer-config", str(paths["ordinary"]),
                        "--bit-attribution-json", str(paths["report"])])
        allocator.main([*argv, *runtime, "--layer-config", str(paths["replay"])],
                       measured_runtime_sweep=emit)
        assert expected.keys() == paths.keys(), "The profile must supply all three outputs."
        for name, path in paths.items():
            assert path.read_bytes() == expected[name], name
