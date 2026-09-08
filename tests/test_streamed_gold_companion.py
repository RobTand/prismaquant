"""Opt-in full-vocabulary companion; the existing top-K gold draw stays fixed."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from tools import build_streamed_full_kl_teacher as builder
from tools import full_kl_teacher_payload as contract
from tools import measure_vllm_full_kl as gold


V2 = "prismaquant.full_kl_teacher_payload/2"


def test_builder_public_cli_accepts_explicit_companion(tmp_path, monkeypatch):
    class ReachedBuild(Exception):
        pass

    def capture(args):
        assert args.include_final_logprobs is True
        raise ReachedBuild

    monkeypatch.setattr(builder, "_build_payload", capture)
    monkeypatch.setattr(sys, "argv", [
        "build_streamed_full_kl_teacher", "--model", str(tmp_path),
        "--identity-cache", "identity.json", "--output", str(tmp_path / "out.pt"),
        "--meta-output", str(tmp_path / "out.json"), "--offload-folder", "offload",
        "--wikitext-inputs", "inputs.json", "--include-final-logprobs",
    ])
    with pytest.raises(ReachedBuild):
        builder.main()


def test_final_companion_uses_last_causal_row_and_full_vocabulary():
    logits = torch.arange(2 * 4 * 7, dtype=torch.float32).reshape(2, 4, 7)
    logits[0, -1] = torch.tensor([8., 2., -4., 0., 1., 3., 7.])
    actual = builder._final_logprobs(logits)
    assert actual.dtype == torch.float32 and actual.device.type == "cpu"
    assert list(actual.shape) == [2, 7]
    torch.testing.assert_close(actual, torch.log_softmax(logits[:, -1, :], dim=-1))


def _student_args(tmp_path, **overrides):
    args = argparse.Namespace(
        teacher_payload=str(tmp_path / "teacher.pt"), teacher_meta="meta.json",
        model="candidate", output=str(tmp_path / "student.json"),
        score_positions="final", quantization=None,
    )
    Path(args.teacher_payload).write_bytes(b"teacher")
    vars(args).update(overrides)
    return args


def test_v2_student_final_uses_companion_and_keeps_evidence(tmp_path, monkeypatch):
    lp = torch.log_softmax(torch.tensor([[2., 0., -1.], [0., 3., 1.]]), dim=-1)
    payload = dict(schema=V2, score_positions="all", final_logprobs=lp,
                   calib_ids=torch.ones(2, 4, dtype=torch.long), seqlen=4,
                   vocab_size=3, model="source")
    evidence = {"schema": "prismaquant.full_kl_teacher_evidence/2", "bound": True}
    monkeypatch.setattr(gold, "load_teacher_evidence", lambda *a: (payload, evidence))
    monkeypatch.setattr(gold, "_require_v2_candidate_identity", lambda *a: None)
    monkeypatch.setattr(gold, "_student_all_positions", lambda *a: pytest.fail("wrong scoring mode"))
    monkeypatch.setattr(gold, "_load_llm", lambda *a, **k: object())
    monkeypatch.setattr(gold, "_measure_logprobs", lambda *a, **k: lp.clone())
    monkeypatch.setattr(gold, "_provenance", lambda *a: {})
    args = _student_args(tmp_path)
    assert gold._student(args) == 0
    result = json.loads(Path(args.output).read_text())
    assert result["score_positions"] == "final" and result["n_positions"] == 2
    assert result["teacher_evidence"] == evidence and result["kl_mean"] == 0


def test_v2_final_without_companion_refuses_before_engine(tmp_path, monkeypatch):
    payload = dict(schema=V2, score_positions="all", final_logprobs=None)
    monkeypatch.setattr(gold, "load_teacher_evidence", lambda *a: (payload, {}))
    monkeypatch.setattr(gold, "_require_v2_candidate_identity", lambda *a: None)
    monkeypatch.setattr(gold, "_student_all_positions", lambda *a: pytest.fail("wrong scoring mode"))
    monkeypatch.setattr(gold, "_load_llm", lambda *a, **k: pytest.fail("loaded engine"))
    with pytest.raises(RuntimeError, match="final.*companion"):
        gold._student(_student_args(tmp_path))


def _producer():
    from tools import serve_fingerprint as fingerprint
    names = fingerprint._GOLD_PRODUCER_COMMON_FILES + fingerprint._GOLD_PRODUCER_TOOL_FILES[
        "build_streamed_full_kl_teacher"]
    files = {name: {"bytes": 1, "sha256": "a" * 64} for name in sorted(set(names))}
    return {"tools": {"schema": fingerprint.GOLD_PRODUCER_IDENTITY_SCHEMA,
                     "measurement_tool": "build_streamed_full_kl_teacher",
                     "git_commit": "b" * 40, "git_tree": "c" * 40, "git_dirty": False,
                     "source_files": files, "source_files_sha256": contract.canonical_sha256(files)},
            "prismaquant_source_sha256": "d" * 64}


@pytest.fixture(scope="module")
def v1_payload(tmp_path_factory):
    """Adapted live-contract fixture from archived test_full_kl_teacher_payload.

    The full historical 8x512x8192 shape is retained; expanded views avoid
    allocating duplicate constant rows until the real descriptor hashes them.
    """
    root = tmp_path_factory.mktemp("teacher-source")
    shard = root / "model.safetensors"
    shard.write_bytes(b"source weights")
    bearing = {"config": {"model_type": "deepseek_v4", "vocab_size": contract.PROMPT_TOP_K + 1},
               "weight_map": {"model.layers.0.weight": "layers.0.weight"},
               "checkpoint_weight_map": {"layers.0.weight": shard.name},
               "shards": [{"path": str(shard), "size": shard.stat().st_size,
                           "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}]}
    (root / "config.json").write_text(json.dumps(bearing["config"]))
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    identity = {"schema": "prismaquant.streamed_model.identity.v1", "source": str(root),
                "resolved_commit": None, "content_sha256": canonical_json_sha256(bearing, where="fixture"), **bearing}
    calib = torch.zeros(contract.N_SAMPLES, contract.SEQLEN, dtype=torch.long)
    starts = list(range(contract.N_SAMPLES))
    calibration = contract.build_calibration_contract(
        dataset_fingerprint="fixture", corpus_sha256="e" * 64,
        tokenizer={"content_sha256": "f" * 64}, starts=starts, total_tokens=100_000,
        calib_ids=calib)
    ids = torch.arange(contract.PROMPT_TOP_K, dtype=torch.int32).view(1, 1, -1).expand(
        contract.N_SAMPLES, contract.SEQLEN - 1, -1)
    lp = (torch.log_softmax(torch.linspace(4., -4., contract.PROMPT_TOP_K).double(), dim=0)
          + math.log(0.99)).float().view(1, 1, -1).expand_as(ids)
    payload = {"schema": contract.TEACHER_PAYLOAD_SCHEMA, "score_positions": "all",
               "prompt_top_k": contract.PROMPT_TOP_K, "topk_ids": ids, "topk_lps": lp,
               "calib_ids": calib, "starts": starts, "model": str(root),
               "n_samples": contract.N_SAMPLES, "seqlen": contract.SEQLEN,
               "vocab_size": contract.PROMPT_TOP_K + 1,
               "source_model_identity": identity,
               "source_model": contract.compact_source_model_identity(identity),
               "source_model_identity_sha256": contract.canonical_sha256(identity),
               "calibration_contract": calibration,
               "calibration_contract_sha256": contract.canonical_sha256(calibration)}
    payload["payload_semantic_sha256"] = contract.payload_semantic_sha256(payload)
    return payload


@pytest.fixture
def v2_payload(v1_payload):
    from tools.dsv4_wikitext_inputs import normalize_wikitext_model_identity
    payload = dict(v1_payload, schema=V2,
                   final_logprobs=torch.full((contract.N_SAMPLES, v1_payload["vocab_size"]),
                                            -math.log(v1_payload["vocab_size"])),
                   source_execution={"schema": "prismaquant.joint_aura.source_execution.v1",
                                     "modules": {"": {"attention": "eager"}}},
                   producer_identity=_producer(), fit_overlap_status="unverified",
                   wikitext_inputs_sha256="e" * 64,
                   model_identity=normalize_wikitext_model_identity(v1_payload["source_model_identity"]["config"]))
    payload["payload_semantic_sha256"] = contract.payload_semantic_sha256(payload)
    return payload


@pytest.mark.parametrize("version", [1, 2])
def test_payload_roundtrip_and_v1_exact_semantic_projection(
        version, v1_payload, v2_payload, tmp_path):
    payload = v1_payload if version == 1 else v2_payload
    contract.validate_teacher_payload(payload)
    if version == 1:
        projection = {key: contract.tensor_descriptor(value) if key in
                      ("calib_ids", "topk_ids", "topk_lps") else value
                      for key, value in payload.items() if key != "payload_semantic_sha256"}
        assert contract.canonical_sha256(projection) == payload["payload_semantic_sha256"]
    path, meta_path = tmp_path / "teacher.pt", tmp_path / "meta.json"
    contract.atomic_torch_save(payload, path)
    meta = contract.teacher_meta(payload_path=path, elapsed_s=1.)
    contract.atomic_json_write(meta, meta_path)
    loaded, evidence = contract.load_teacher_evidence(path, meta_path)
    assert evidence["schema"] == f"prismaquant.full_kl_teacher_evidence/{version}"
    assert loaded["payload_semantic_sha256"] == payload["payload_semantic_sha256"]
    if version == 2:
        assert evidence["final_logprobs_descriptor"] == contract.tensor_descriptor(payload["final_logprobs"])
        assert evidence["source_execution"] == payload["source_execution"]
        assert evidence["fit_overlap_status"] == "unverified"
    else:
        assert "final_logprobs_descriptor" not in meta and "source_execution" not in evidence


@pytest.mark.parametrize("mutation, error", [
    (lambda p: p.update(final_logprobs=p["final_logprobs"].half()), "shape/dtype"),
    (lambda p: p.update(final_logprobs=p["final_logprobs"][:, :-1]), "shape/dtype"),
    (lambda p: p.update(final_logprobs=p["final_logprobs"] + 0.1), "normalized"),
    (lambda p: p.update(final_logprobs=p["final_logprobs"] * float("nan")), "finite"),
    (lambda p: p.update(final_logprobs=torch.zeros_like(p["final_logprobs"]) + 1), "nonpositive"),
    (lambda p: p.update(fit_overlap_status="held_out"), "overlap"),
    (lambda p: p.update(wikitext_inputs_sha256="bad"), "SHA256"),
    (lambda p: p.update(source_execution={"schema": "invented", "modules": {}}), "schema"),
    (lambda p: p.update(producer_identity={}), "closed"),
    (lambda p: p.update(extra="unpriced"), "closed"),
])
def test_v2_closed_fields_and_final_contract(v2_payload, mutation, error):
    mutation(v2_payload)
    with pytest.raises(contract.TeacherPayloadError, match=error):
        contract.validate_teacher_payload(v2_payload)


@pytest.mark.parametrize("field", ["final_logprobs", "source_execution", "source_model_identity"])
def test_v2_tensor_or_source_tamper_refuses(v2_payload, field):
    if field == "final_logprobs":
        # Remains normalized; the semantic byte seal must detect the change.
        v2_payload[field] = torch.log_softmax(torch.randn_like(v2_payload[field]), dim=-1)
    elif field == "source_execution":
        v2_payload[field] = {"schema": "prismaquant.joint_aura.source_execution.v1",
                             "modules": {"": {"attention": "sdpa"}}}
    else:
        v2_payload[field] = dict(v2_payload[field], content_sha256="a" * 64)
    with pytest.raises(contract.TeacherPayloadError):
        contract.validate_teacher_payload(v2_payload)


def test_v2_meta_tamper_refuses_before_student_engine(v2_payload, tmp_path, monkeypatch):
    payload_path, meta_path = tmp_path / "teacher.pt", tmp_path / "meta.json"
    contract.atomic_torch_save(v2_payload, payload_path)
    meta = contract.teacher_meta(payload_path=payload_path, elapsed_s=1.)
    meta["final_logprobs_descriptor"]["sha256"] = "a" * 64
    contract.atomic_json_write(meta, meta_path)
    monkeypatch.setattr(gold, "_load_llm", lambda *a, **k: pytest.fail("loaded engine"))
    args = argparse.Namespace(teacher_payload=str(payload_path), teacher_meta=str(meta_path))
    with pytest.raises(contract.TeacherPayloadError, match="metadata differs"):
        gold._student(args)


def test_v2_requires_meta_before_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(gold, "safe_load_torch_payload", lambda *a: {"schema": V2})
    monkeypatch.setattr(gold, "_load_llm", lambda *a, **k: pytest.fail("loaded engine"))
    with pytest.raises(RuntimeError, match="authenticated.*teacher-meta"):
        gold._student(_student_args(tmp_path, teacher_meta=None))


@pytest.mark.parametrize("version, requested_mode", [(1, "final"), (1, "all"), (2, "all")])
def test_existing_v1_all_routing_and_v2_explicit_all_preserved(
        tmp_path, monkeypatch, version, requested_mode):
    payload = {"schema": f"prismaquant.full_kl_teacher_payload/{version}", "score_positions": "all"}
    monkeypatch.setattr(gold, "load_teacher_evidence", lambda *a: (payload, {}))
    monkeypatch.setattr(gold, "_require_v2_candidate_identity", lambda *a: None)
    monkeypatch.setattr(gold, "_student_all_positions", lambda *a: 27)
    assert gold._student(_student_args(tmp_path, score_positions=requested_mode)) == 27


def test_v2_null_companion_is_valid_for_all_position_use(v2_payload):
    v2_payload["final_logprobs"] = None
    v2_payload["payload_semantic_sha256"] = contract.payload_semantic_sha256(v2_payload)
    assert contract.validate_teacher_payload(v2_payload)["final_logprobs"] is None


def test_builder_source_policy_requires_both_cli_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["builder", "--model", "model", "--identity-cache", "identity",
        "--output", str(tmp_path / "out"), "--meta-output", str(tmp_path / "meta"),
        "--offload-folder", "offload", "--wikitext-inputs", "inputs",
        "--source-derivative-json", "policy.json"])
    monkeypatch.setattr(builder, "_build_payload", lambda *a: pytest.fail("built model"))
    with pytest.raises(SystemExit) as error:
        builder.main()
    assert error.value.code == 2


def test_source_policy_null_and_digest_tamper(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text("null")
    args = argparse.Namespace(source_derivative_json=str(path),
                              source_derivative_sha256=contract.file_sha256(path))
    assert builder._source_derivative_policy(args) is None
    path.write_text("{}")
    with pytest.raises(ValueError, match="bytes changed"):
        builder._source_derivative_policy(args)


def _declared_derivative():
    from prismaquant.glm_source_derivative import SCHEMA, VERSION
    return {"schema": SCHEMA, "version": VERSION,
            "image_build": {"path": "image-build.json", "sha256": "a" * 64}}


def _observed_derivative():
    from prismaquant.glm_source_derivative import (
        declaration, CORRECTED_MODELING_SHA256, CORRECTED_IMAGE_CONTENT_SHA256,
        ORIGINAL_HUB_KERNELS_SHA256, ORIGINAL_ACCELERATE_INTEGRATION_SHA256,
    )
    return {"declaration": declaration(), "modeling_sha256": CORRECTED_MODELING_SHA256,
            "image_content_sha256": CORRECTED_IMAGE_CONTENT_SHA256,
            "hub_kernels_sha256": ORIGINAL_HUB_KERNELS_SHA256,
            "accelerate_integration_sha256": ORIGINAL_ACCELERATE_INTEGRATION_SHA256,
            "image_build_sha256": "a" * 64,
            "gates": {"layer": {"branch": "safe_lower_bound_times_sigmoid", "lower_bound": -5.,
                                "heads": 64, "head_dim": 128}}}


def _stub_builder(v1_payload, tmp_path, monkeypatch, *, include_final=True,
                  drift=None, explicit_derivative=False):
    from prismaquant import cost_streaming, gpu_guard, joint_aura, model_profiles
    from transformers import AutoTokenizer
    source = v1_payload["source_model_identity"]
    input_path = tmp_path / "inputs.json"
    input_path.write_text("{}")
    args = argparse.Namespace(model=source["source"], identity_cache="identity.json",
        wikitext_inputs=str(input_path), offload_folder=str(tmp_path / "offload"),
        cache_headroom_gb=100., logits_chunk_rows=32, include_final_logprobs=include_final,
        source_derivative_json="declared.json" if explicit_derivative else None)
    calls = {"forward": 0, "shutdown": 0, "source": 0, "execution": 0, "policy": 0, "tokenizer": 0, "input_model": 0,
             "producer": 0, "kwargs": None}
    vocab = v1_payload["vocab_size"]
    last = torch.arange(vocab).float() / vocab
    logits = last.view(1, 1, vocab).expand(contract.N_SAMPLES, contract.SEQLEN, vocab)

    class Runner:
        context = SimpleNamespace(max_cache_slots=1)
        prefetch_lookahead = 0
        model = object()

        def __call__(self, ids):
            calls["forward"] += 1
            torch.testing.assert_close(ids, v1_payload["calib_ids"])
            if drift == "input":
                input_path.write_text('{"changed":true}')
            return SimpleNamespace(logits=logits)

        def shutdown(self):
            calls["shutdown"] += 1

    def build(*a, **kwargs):
        calls["kwargs"] = kwargs
        return Runner()

    def identity(*a, **kwargs):
        calls["source"] += 1
        return dict(source, content_sha256="a" * 64) if drift == "source" and calls["source"] > 1 else source

    execution = {"schema": "prismaquant.joint_aura.source_execution.v1", "modules": {"": {"attention": "eager"}}}
    if explicit_derivative:
        execution.update(schema="prismaquant.joint_aura.source_execution.v2",
                         source_derivative=_observed_derivative())

    def observe(*a):
        calls["execution"] += 1
        return dict(execution, modules={}) if drift == "execution" and calls["execution"] > 1 else execution

    def policy(*a):
        calls["policy"] += 1
        if drift == "derivative" and calls["policy"] > 1:
            raise ValueError("derivative bytes changed")
        return _declared_derivative() if explicit_derivative else None

    def tokenizer_identity(*a):
        calls["tokenizer"] += 1
        return {"content_sha256": "a" * 64 if drift == "tokenizer" and calls["tokenizer"] > 1 else "f" * 64}

    def input_model_identity(*a):
        from tools.dsv4_wikitext_inputs import normalize_wikitext_model_identity
        calls["input_model"] += 1
        config = source["config"]
        if drift == "input_model" and calls["input_model"] > 1:
            config = dict(config, model_type="changed")
        return normalize_wikitext_model_identity(config)

    def producer():
        calls["producer"] += 1
        value = _producer()
        if drift == "producer" and calls["producer"] > 1:
            value["prismaquant_source_sha256"] = "a" * 64
        return value

    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *a: torch.device("cpu"))
    monkeypatch.setattr(model_profiles, "detect_profile", lambda *a: object())
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: range(vocab))
    monkeypatch.setattr(builder, "tokenizer_identity", tokenizer_identity)
    monkeypatch.setattr(builder, "_input_model_identity", input_model_identity)
    monkeypatch.setattr(builder, "_load_gold_inputs", lambda *a: {"full_kl": {
        "token_ids": v1_payload["calib_ids"].tolist(), "selection": {"starts": v1_payload["starts"]},
        "dataset": dict(v1_payload["calibration_contract"]["dataset"], total_tokens=100_000)}})
    monkeypatch.setattr(cost_streaming, "validate_cached_streamed_model_identity", identity)
    monkeypatch.setattr(cost_streaming, "build_streamed_causal_lm", build)
    monkeypatch.setattr(joint_aura, "source_execution_identity", observe)
    monkeypatch.setattr(builder, "_source_derivative_policy", policy)
    monkeypatch.setattr(builder, "_teacher_producer_identity", producer)
    monkeypatch.setattr(builder, "_topk_all_positions", lambda *a, **k: (v1_payload["topk_ids"], v1_payload["topk_lps"]))
    return args, calls, logits


def test_builder_one_forward_binds_final_rows_and_source_policy(v1_payload, tmp_path, monkeypatch):
    args, calls, logits = _stub_builder(v1_payload, tmp_path, monkeypatch, explicit_derivative=True)
    payload = builder._build_payload(args)
    assert calls["forward"] == calls["shutdown"] == 1
    assert calls["source"] == calls["execution"] == calls["producer"] == 2
    assert calls["kwargs"]["source_derivative"] == _declared_derivative()
    assert calls["kwargs"]["attn_implementation"] == "eager"
    assert calls["kwargs"]["max_cache_slots"] == 1 and calls["kwargs"]["prefetch_lookahead"] == 0
    torch.testing.assert_close(payload["final_logprobs"], torch.log_softmax(logits[:, -1, :], dim=-1))
    assert payload["topk_lps"] is v1_payload["topk_lps"]
    assert payload["wikitext_inputs_sha256"] == contract.file_sha256(args.wikitext_inputs)


def test_builder_default_v1_payload_and_runner_kwargs_preserved(v1_payload, tmp_path, monkeypatch):
    args, calls, _ = _stub_builder(v1_payload, tmp_path, monkeypatch, include_final=False)
    payload = builder._build_payload(args)
    assert payload["schema"] == contract.TEACHER_PAYLOAD_SCHEMA
    assert set(payload) == set(v1_payload)
    assert calls["execution"] == calls["producer"] == 0 and calls["source"] == 1
    assert "source_derivative" not in calls["kwargs"] and "attn_implementation" not in calls["kwargs"]


@pytest.mark.parametrize("drift", ["source", "execution", "producer", "derivative", "input", "tokenizer", "input_model"])
def test_builder_drift_refuses_and_releases_runner(v1_payload, tmp_path, monkeypatch, drift):
    args, calls, _ = _stub_builder(v1_payload, tmp_path, monkeypatch, drift=drift)
    with pytest.raises((RuntimeError, ValueError), match="changed"):
        builder._build_payload(args)
    assert calls["forward"] == calls["shutdown"] == 1


def test_failed_existing_fidelity_gate_does_not_publish_companion(v2_payload, tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise contract.TeacherPayloadError("existing forward fidelity refusal")
    monkeypatch.setattr(contract, "teacher_forward_fidelity_summary", refuse)
    monkeypatch.setattr(builder, "_build_payload", lambda *a: contract.validate_teacher_payload(v2_payload))
    monkeypatch.setattr(builder, "atomic_torch_save", lambda *a: pytest.fail("published invalid teacher"))
    output, meta = tmp_path / "out.pt", tmp_path / "meta.json"
    monkeypatch.setattr(sys, "argv", ["builder", "--model", "model", "--identity-cache", "identity",
        "--output", str(output), "--meta-output", str(meta), "--offload-folder", "offload",
        "--wikitext-inputs", "inputs", "--include-final-logprobs"])
    with pytest.raises(contract.TeacherPayloadError, match="fidelity"):
        builder.main()
    assert not output.exists() and not meta.exists()


def test_producer_closure_has_builder_and_shared_input_tools():
    from tools import serve_fingerprint as fingerprint
    names = set(fingerprint._GOLD_PRODUCER_COMMON_FILES + fingerprint._GOLD_PRODUCER_TOOL_FILES[
        "build_streamed_full_kl_teacher"])
    assert {"tools/build_streamed_full_kl_teacher.py", "tools/full_kl_teacher_payload.py",
            "tools/dsv4_wikitext_inputs.py", "tools/prepare_dsv4_wikitext_inputs.py",
            "tools/container_runtime_identity.py"} <= names


@pytest.mark.parametrize("policy,execution", [
    (_declared_derivative(), {"schema": "prismaquant.joint_aura.source_execution.v1", "modules": {}}),
    (_declared_derivative(), {"schema": "prismaquant.joint_aura.source_execution.v2", "modules": {},
                              "source_derivative": {"image_build_sha256": "b" * 64}}),
    (None, {"schema": "prismaquant.joint_aura.source_execution.v2", "modules": {},
            "source_derivative": _observed_derivative()}),
])
def test_declared_source_policy_must_match_observed_execution(
        policy, execution, v1_payload, tmp_path, monkeypatch):
    from prismaquant import joint_aura
    args, calls, _ = _stub_builder(v1_payload, tmp_path, monkeypatch)
    monkeypatch.setattr(builder, "_source_derivative_policy", lambda *a: policy)
    monkeypatch.setattr(joint_aura, "source_execution_identity", lambda *a: execution)
    with pytest.raises(RuntimeError, match="observed"):
        builder._build_payload(args)
    assert calls["forward"] == 0 and calls["shutdown"] == 1


@pytest.mark.parametrize("mismatch", ["tokenizer", "family", "vocabulary", "teacher_vocabulary"])
def test_v2_candidate_identity_mismatch_refuses_before_engine(
        v2_payload, tmp_path, monkeypatch, mismatch):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    config = dict(v2_payload["source_model_identity"]["config"],
                  quantization_config={"quant_method": "tessera"})
    if mismatch == "family":
        config["model_type"] = "different_family"
    if mismatch == "vocabulary":
        config["vocab_size"] += 1
    if mismatch == "teacher_vocabulary":
        v2_payload["vocab_size"] += 1
    (candidate / "config.json").write_text(json.dumps(config))
    digest = v2_payload["calibration_contract"]["tokenizer"]["identity_sha256"]
    monkeypatch.setattr(gold, "tokenizer_identity", lambda *a: {
        "content_sha256": "a" * 64 if mismatch == "tokenizer" else digest})
    monkeypatch.setattr(gold, "load_teacher_evidence", lambda *a: (v2_payload, {}))
    monkeypatch.setattr(gold, "_load_llm", lambda *a, **k: pytest.fail("loaded engine"))
    with pytest.raises(RuntimeError, match="candidate.*differs"):
        gold._student(_student_args(tmp_path, model=str(candidate)))


def test_v2_candidate_allows_quantization_config_but_binds_tokenizer(v2_payload, tmp_path, monkeypatch):
    config = dict(v2_payload["source_model_identity"]["config"],
                  quantization_config={"quant_method": "tessera"}, architectures=["QuantizedModel"])
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(gold, "tokenizer_identity", lambda *a: {
        "content_sha256": v2_payload["calibration_contract"]["tokenizer"]["identity_sha256"]})
    gold._require_v2_candidate_identity(argparse.Namespace(model=str(tmp_path)), v2_payload)


def test_builder_generic_input_reader_requires_independent_sha(tmp_path, monkeypatch):
    from tools import dsv4_wikitext_inputs as inputs
    called = {}
    def read(path, **kwargs):
        called.update(path=path, **kwargs)
        return {"full_kl": "verified"}
    monkeypatch.setattr(inputs, "load_wikitext_inputs", read)
    monkeypatch.setattr(inputs, "wikitext_model_identity", lambda *a: {"model": "bound"})
    args = argparse.Namespace(wikitext_inputs="input.json", wikitext_inputs_sha256="a" * 64)
    assert builder._load_gold_inputs(args, tmp_path, {"tokenizer": "bound"}) == {"full_kl": "verified"}
    assert called == {"path": "input.json", "expected_sha256": "a" * 64,
                      "expected_tokenizer_identity": {"tokenizer": "bound"},
                      "expected_model_identity": {"model": "bound"}}


def test_v2_full_glm_candidate_pairs_to_original_domain_despite_staged_text_config(
        v2_payload, tmp_path, monkeypatch):
    from tools.dsv4_wikitext_inputs import normalize_wikitext_model_identity
    full = {"model_type": "glm5_next", "text_config": {
        "model_type": "glm5_next_text", "vocab_size": v2_payload["vocab_size"]},
        "vision_config": {"model_type": "glm5_next_vision"},
        "quantization_config": {"quant_method": "tessera"}}
    staged = dict(full["text_config"])
    v2_payload["source_model_identity"] = dict(v2_payload["source_model_identity"], config=staged)
    v2_payload["model_identity"] = normalize_wikitext_model_identity(full)
    assert normalize_wikitext_model_identity(staged) != v2_payload["model_identity"]
    (tmp_path / "config.json").write_text(json.dumps(full))
    monkeypatch.setattr(gold, "tokenizer_identity", lambda *a: {
        "content_sha256": v2_payload["calibration_contract"]["tokenizer"]["identity_sha256"]})
    gold._require_v2_candidate_identity(argparse.Namespace(model=str(tmp_path)), v2_payload)
