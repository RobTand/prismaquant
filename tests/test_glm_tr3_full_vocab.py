"""CPU contracts only; native stock-vLLM hook qualification is separate."""
import copy
import enum
import hashlib
import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import glm_tr3_full_vocab as exp
from experiments import build_glm_tr3_teacher as teacher
from experiments import measure_glm_tr3_vllm as served
from experiments import authenticate_glm_tr3_source as authentication


def test_native_restart_kv_capacity_does_not_invalidate_qualification():
    fixtures = Path(__file__).parent / "fixtures" / "glm_tr3_runtime"
    qualified = json.loads((fixtures / "qualified-runtime.json").read_text())["runtime_binding"]
    restarted = json.loads((fixtures / "restarted-runtime.json").read_text())["runtime_binding"]
    assert qualified != restarted  # Actual attempt 07 versus attempt 08 allocations.
    original = copy.deepcopy((qualified, restarted))
    assert served.qualification_runtime_matches(qualified, restarted)
    assert (qualified, restarted) == original


def test_runtime_mismatch_diagnostics_are_normalized_bounded_and_specific():
    fixtures = Path(__file__).parent / "fixtures" / "glm_tr3_runtime"
    qualified = json.loads((fixtures / "qualified-runtime.json").read_text())["runtime_binding"]
    restarted = json.loads((fixtures / "restarted-runtime.json").read_text())["runtime_binding"]
    assert served.qualification_runtime_differences(qualified, restarted) == []
    restarted["teacher_sha256"] = "different"
    restarted["worker_runtime"][0]["attention_runtime"][3]["backend"] = "renamed.backend"
    differences = served.qualification_runtime_differences(qualified, restarted, limit=2)
    assert differences == [
        '$["teacher_sha256"]',
        '$["worker_runtime"][0]["attention_runtime"][3]["allocated_kv_cache"]["shape"][0]',
    ]
    assert any(path.endswith('["backend"]') for path in
               served.qualification_runtime_differences(qualified, restarted))
    with pytest.raises(ValueError, match=r'teacher_sha256'):
        served.require_native_qualification({
            "schema": "prismaquant.glm_tr3_hook_qualification/1",
            "passed": True, "runtime_binding": qualified,
        }, restarted)


@pytest.mark.parametrize("field,value", [("schema", "wrong"), ("passed", False)])
def test_qualification_envelope_diagnostic(field, value):
    qualification = {"schema": "prismaquant.glm_tr3_hook_qualification/1",
                     "passed": True, "runtime_binding": {"worker_runtime": []}}
    qualification[field] = value
    with pytest.raises(ValueError, match=field):
        served.require_native_qualification(qualification, {"worker_runtime": []})


def test_invalid_runtime_shape_diagnostic_identifies_side_and_path():
    binding = {"worker_runtime": [{"attention_runtime": [{
        "backend": "vllm.v1.attention.backends.mla.indexer.DeepseekV32IndexerBackend",
        "allocated_kv_cache": {"shape": [0, 4, 8]},
    }]}]}
    differences = served.qualification_runtime_differences(binding, binding)
    assert len(differences) == 1
    assert 'qualified' in differences[0]
    assert '["worker_runtime"][0]["attention_runtime"][0]["allocated_kv_cache"]["shape"]' in differences[0]


@pytest.mark.parametrize("field", [
    "dtype", "device", "layout", "zero_blocks", "negative_blocks", "boolean_blocks",
    "rank", "backend", "backend_source", "unknown_backend", "module", "missing_cache",
    "teacher", "candidate", "producer", "context", "logits_layout", "tp", "missing_binding",
])
def test_native_restart_still_refuses_noncapacity_changes(field):
    fixtures = Path(__file__).parent / "fixtures" / "glm_tr3_runtime"
    qualified = json.loads((fixtures / "qualified-runtime.json").read_text())["runtime_binding"]
    restarted = json.loads((fixtures / "restarted-runtime.json").read_text())["runtime_binding"]
    attention = restarted["worker_runtime"][0]["attention_runtime"][3]
    cache = attention["allocated_kv_cache"]
    if field in ("dtype", "device"):
        cache[field] = "different"
    elif field == "layout":
        cache["shape"][1] += 1
    elif field in ("zero_blocks", "negative_blocks", "boolean_blocks"):
        cache["shape"][0] = {"zero_blocks": 0, "negative_blocks": -1, "boolean_blocks": True}[field]
    elif field == "rank":
        cache["shape"].append(1)
    elif field in ("backend", "backend_source", "module"):
        attention[{"backend": "backend", "backend_source": "backend_source_sha256", "module": "module"}[field]] = "different"
    elif field == "unknown_backend":
        # Matching but unknown backend names cannot waive an axis comparison.
        attention["backend"] = qualified["worker_runtime"][0]["attention_runtime"][3]["backend"] = "unknown"
    elif field == "missing_cache":
        attention["allocated_kv_cache"] = None
    elif field == "teacher":
        restarted["teacher_sha256"] = "different"
    elif field == "candidate":
        restarted["candidate_identity"]["content_sha256"] = "different"
    elif field == "producer":
        restarted["producer_identity"]["gold_source"]["tools"]["git_commit"] = "different"
    elif field == "context":
        restarted["engine_kwargs"]["max_model_len"] += 1
    elif field == "logits_layout":
        restarted["logits_layout"] = "legacy_single"
    elif field == "tp":
        restarted["worker_runtime"][1]["rank"] = 0
    elif field == "missing_binding":
        restarted = None
    assert not served.qualification_runtime_matches(qualified, restarted)


def upstream_oracle(teacher, student):
    # Exact upstream token_kld_chunk calculation, copied for the small CPU
    # oracle only. Source: pinned TR3 runtime quant_pipeline/evaluation/glm53_logits.py.
    teacher64 = np.asarray(teacher, dtype=np.float64).copy()
    student64 = np.asarray(student, dtype=np.float64).copy()
    if not np.isfinite(teacher64).all() or not np.isfinite(student64).all():
        raise ValueError("teacher/student logits must be finite")
    teacher64 -= np.max(teacher64, axis=-1, keepdims=True)
    student64 -= np.max(student64, axis=-1, keepdims=True)
    teacher64 -= np.logaddexp.reduce(teacher64, axis=-1, keepdims=True)
    student64 -= np.logaddexp.reduce(student64, axis=-1, keepdims=True)
    return np.sum(np.exp(teacher64) * (teacher64 - student64), axis=-1)


@pytest.mark.parametrize("tile_rows", [1, 3, 32])
@pytest.mark.parametrize("scale", [0., 1., 1000.])
def test_full_vocabulary_fp64_matches_upstream(tile_rows, scale):
    rng = np.random.default_rng(42)
    t = (rng.normal(size=(7, 19)) * scale).astype(np.float32)
    c = (rng.normal(size=(7, 19)) * scale).astype(np.float32)
    original = t.copy()
    actual = exp.token_kl(torch.from_numpy(t), torch.from_numpy(c),
                         tile_rows=tile_rows, require_cuda=False)
    assert actual.dtype == torch.float64
    np.testing.assert_allclose(actual.numpy(), upstream_oracle(t, c), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(t, original)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", [0, 1])
def test_nonfinite_refuses_before_number(bad, side):
    tensors = [torch.zeros(4, 5), torch.ones(4, 5)]
    tensors[side][2, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        exp.token_kl(*tensors, tile_rows=2, require_cuda=False)


def test_actual_scoring_has_no_cpu_fallback():
    with pytest.raises(ValueError, match="CUDA"):
        exp.token_kl(torch.zeros(2, 3), torch.zeros(2, 3))


def capture(rank=0, world=1):
    h = exp.PromptLogitsCapture(rank=rank, world_size=world, rows=3, vocab_size=5,
                              require_cuda=False)
    h.arm(0, "final-0000", torch.zeros(3, 5) if rank == 0 else None)
    return h


@pytest.mark.parametrize("reverse", [False, True])
def test_hook_does_not_modify_output_and_accepts_stock_call_order(reverse):
    h = capture()
    calls = [torch.ones(1, 5), torch.arange(15).reshape(3, 5).float()]
    for value in reversed(calls) if reverse else calls:
        original = value.clone()
        assert h(None, (), value) is None
        torch.testing.assert_close(value, original)
    result = h.finish("final-0000")
    values = exp.collect_tp_result([result], window_id="final-0000", world_size=1,
                                   rows=3, vocab_size=5)
    assert len(values) == 3
    with pytest.raises(ValueError, match="order"):
        h.arm(0, "final-0000", torch.zeros(3, 5))
    h.arm(1, "final-0001", torch.zeros(3, 5))


@pytest.mark.parametrize("mode", ["full", "none"])
def test_tp_exactly_one_score_owner(mode):
    owner, other = capture(0, 2), capture(1, 2)
    for h in (owner, other):
        for value in (torch.zeros(1, 5), torch.ones(3, 5)):
            h(None, (), None if h.rank == 1 and mode == "none" else value)
    results = [h.finish("final-0000") for h in (owner, other)]
    assert len(exp.collect_tp_result(results, window_id="final-0000", world_size=2,
                                     rows=3, vocab_size=5)) == 3
    results[1]["values"] = [0.] * 3
    with pytest.raises(ValueError, match="ownership"):
        exp.collect_tp_result(results, window_id="final-0000", world_size=2, rows=3, vocab_size=5)


@pytest.mark.parametrize("shape", [(2, 5), (3, 4), (4, 5), (1, 3, 5)])
def test_hook_refuses_chunking_and_partial_vocab(shape):
    with pytest.raises(ValueError):
        capture()(None, (), torch.zeros(shape))


def test_hook_missing_duplicate_unarmed_wrong_window_and_owner_none():
    h = capture()
    h(None, (), torch.zeros(1, 5))
    with pytest.raises(ValueError, match="missing"):
        h.finish("final-0000")
    with pytest.raises(ValueError, match="duplicate"):
        h(None, (), torch.zeros(1, 5))
    with pytest.raises(ValueError, match="owner"):
        capture()(None, (), None)
    h = capture()
    for x in (torch.zeros(1, 5), torch.zeros(3, 5)):
        h(None, (), x)
    with pytest.raises(ValueError, match="wrong window"):
        h.finish("final-0001")
    h.finish("final-0000")
    with pytest.raises(ValueError, match="unarmed"):
        h(None, (), torch.zeros(1, 5))


@pytest.mark.parametrize("mutation", ["duplicate-rank", "missing-rank", "wrong-window", "wrong-world", "missing-owner-vector"])
def test_tp_report_refuses_ambiguous_ownership(mutation):
    results = []
    for rank in (0, 1):
        h = capture(rank, 2)
        for x in (torch.zeros(1, 5), torch.zeros(3, 5)):
            h(None, (), x)
        results.append(h.finish("final-0000"))
    if mutation == "duplicate-rank": results[1]["rank"] = 0
    elif mutation == "missing-rank": results.pop()
    elif mutation == "wrong-window": results[1]["window_id"] = "final-0001"
    elif mutation == "wrong-world": results[1]["world_size"] = 3
    else: results[0]["values"] = None
    with pytest.raises(ValueError):
        exp.collect_tp_result(results, window_id="final-0000", world_size=2, rows=3, vocab_size=5)


def test_summary_preserves_domain_document_and_position_vectors():
    panel = {"windows": [{"window_id": f"final-{i:04d}", "domain": f"d{i % 2}",
                           "document_id": f"doc{i % 2}", "prediction_positions": 3}
                          for i in range(4)]}
    vectors = [[float(i)] * 3 for i in range(4)]
    result = exp.summarize_panel(panel, vectors)
    assert result["mean"] == 1.5
    assert result["documents"]["doc0"] == {"positions": 6, "mean": 1.}
    assert "correlated" in result["interpretation"]
    assert "pvalue" not in result
    with pytest.raises(ValueError):
        exp.summarize_panel(panel, vectors[:-1])


def test_native_target_alignment_checks_causal_order():
    from types import SimpleNamespace
    tokens = [4, 2, 3, 1]
    h = exp.PromptLogitsCapture(rank=0, world_size=1, rows=3, vocab_size=5, require_cuda=False)
    targets = torch.tensor(tokens[1:])
    h.arm(0, "final-0000", torch.zeros(3, 5), targets)
    logits = torch.arange(15).reshape(3, 5).float()
    h(None, (), torch.zeros(1, 5))
    h(None, (), logits)
    report = h.finish("final-0000")
    lps = torch.log_softmax(logits, -1)
    output = SimpleNamespace(prompt_token_ids=tokens,
        prompt_logprobs=[None] + [{t: SimpleNamespace(logprob=float(lps[i, t]))}
                                  for i, t in enumerate(tokens[1:])])
    assert served.verify_prompt_alignment(output, tokens, [report])["passed"]
    output.prompt_logprobs[1][2].logprob += .01
    with pytest.raises(ValueError, match="alignment"):
        served.verify_prompt_alignment(output, tokens, [report])


def v2_capture(rank=0, world=1):
    h = exp.PromptLogitsCapture(rank=rank, world_size=world, rows=2047,
        vocab_size=11, require_cuda=False, logits_layout="vllm_v2_chunk1024")
    generator = torch.Generator().manual_seed(42)
    teacher_logits = torch.randn(2047, 11, generator=generator)
    candidate = torch.randn(2048, 11, generator=generator)
    tokens = (torch.arange(2048) % 11).tolist()
    h.arm(0, "final-0000", teacher_logits if rank == 0 else None,
          torch.tensor(tokens[1:]) if rank == 0 else None)
    return h, teacher_logits, candidate, tokens


@pytest.mark.parametrize("other_gathers", [True, False])
def test_native_v2_logits_chunks_cover_exact_causal_rows(other_gathers):
    from types import SimpleNamespace
    owner, teacher_logits, candidate, tokens = v2_capture(0, 2)
    other, _, _, _ = v2_capture(1, 2)
    candidate[-1] = 10000.  # This final prompt row is outside the teacher.
    original = candidate.clone()
    for h in (owner, other):
        for output in (candidate[-1:], candidate[:1024], candidate[1024:]):
            assert h(None, (), output if h.rank == 0 or other_gathers else None) is None
    torch.testing.assert_close(candidate, original)
    reports = [h.finish("final-0000") for h in (owner, other)]
    values = exp.collect_tp_result(reports, window_id="final-0000", world_size=2,
        rows=2047, vocab_size=11, logits_layout="vllm_v2_chunk1024")
    np.testing.assert_allclose(values, upstream_oracle(teacher_logits.numpy(), candidate[:2047].numpy()),
                               rtol=1e-12, atol=1e-12)
    lp = torch.log_softmax(candidate[:2047], -1)
    output = SimpleNamespace(prompt_token_ids=tokens, prompt_logprobs=[None] + [
        {token: SimpleNamespace(logprob=float(lp[index, token]))}
        for index, token in enumerate(tokens[1:])])
    assert served.verify_prompt_alignment(output, tokens, reports)["positions"] == 2047
    assert reports[0]["calls"] == [(1, 11), (1024, 11), (1024, 11)]
    reports[1]["logits_layout"] = "legacy_single"
    with pytest.raises(ValueError, match="geometry"):
        exp.collect_tp_result(reports, window_id="final-0000", world_size=2,
            rows=2047, vocab_size=11, logits_layout="vllm_v2_chunk1024")


@pytest.mark.parametrize("mutation", ["missing", "extra", "sample-last", "partial-vocab", "partial-chunk"])
def test_native_v2_refuses_incomplete_or_undeclared_geometry(mutation):
    h, _, candidate, _ = v2_capture()
    calls = [candidate[-1:], candidate[:1024], candidate[1024:]]
    if mutation == "missing": calls.pop()
    elif mutation == "extra": calls.append(candidate[:1024])
    elif mutation == "sample-last": calls = calls[1:] + calls[:1]
    elif mutation == "partial-vocab": calls[1] = calls[1][:, :-1]
    else: calls[1] = calls[1][:-1]
    with pytest.raises(ValueError):
        for output in calls:
            h(None, (), output)
        h.finish("final-0000")


@pytest.mark.parametrize("mutation", ["duplicate-first", "reorder"])
def test_native_v2_alignment_refuses_wrong_chunk_content(mutation):
    from types import SimpleNamespace
    h, _, candidate, tokens = v2_capture()
    chunks = [candidate[:1024], candidate[1024:]]
    if mutation == "duplicate-first": chunks[1] = chunks[0]
    else: chunks.reverse()
    for output in [candidate[-1:], *chunks]:
        h(None, (), output)
    report = h.finish("final-0000")
    lp = torch.log_softmax(candidate[:2047], -1)
    output = SimpleNamespace(prompt_token_ids=tokens, prompt_logprobs=[None] + [
        {token: SimpleNamespace(logprob=float(lp[index, token]))}
        for index, token in enumerate(tokens[1:])])
    with pytest.raises(ValueError, match="alignment"):
        served.verify_prompt_alignment(output, tokens, [report])


def test_native_v2_layout_requires_the_observed_context():
    with pytest.raises(ValueError, match="layout"):
        exp.PromptLogitsCapture(rank=0, world_size=1, rows=3, vocab_size=11,
                               require_cuda=False, logits_layout="vllm_v2_chunk1024")


@pytest.fixture
def sealed_panel(tmp_path, monkeypatch):
    def save(name, array):
        stream = io.BytesIO()
        np.save(stream, array, allow_pickle=False)
        raw = stream.getvalue()
        path = tmp_path / name
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest(), len(raw)
    mask, mask_sha, _ = save("causal-mask-2048.npy", np.ones(2048, dtype=np.uint8))
    panel = {"schema": "prismaquant.exl3-final-panel-handoff.v1",
             "dataset_revision": exp.DATASET_REVISION,
             "reference_model": "zai-org/GLM-5.3-Flash-BF16", "reference_revision": exp.REFERENCE_REVISION,
             "tokenizer_sha256": exp.TOKENIZER_SHA256, "vocab_size": exp.VOCAB_SIZE,
             "context_length": 2048, "window_count": 25, "prediction_positions_per_window": 2047,
             "total_prediction_positions": 51175, "maximum_token_id_exclusive": 154856,
             "causal_mask_array": str(mask), "causal_mask_sha256": mask_sha, "windows": []}
    for i in range(25):
        name = f"final-{i:04d}"
        path, digest, size = save(name + ".tokens.npy", np.arange(2048, dtype=np.int32) + i)
        panel["windows"].append({"window_id": name, "role": "final", "prediction_positions": 2047,
            "attention_mask_sha256": mask_sha, "tokens_path": str(path), "tokens_sha256": digest,
            "panel_token_ids_sha256": digest, "tokens_bytes": size, "document_id": f"doc{i%4}",
            "domain": f"domain{i%4}"})
    path = tmp_path / "handoff.json"
    raw = json.dumps(panel).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(exp, "PANEL_SHA256", hashlib.sha256(raw).hexdigest())
    return path, panel


def test_panel_authenticates_before_loading_and_retains_order(sealed_panel):
    path, original = sealed_panel
    panel, inputs = exp.load_panel(path, arrays_root=path.parent)
    assert panel == original and len(inputs) == 25
    assert all(tuple(t.shape) == (1, 2048) and t.dtype == torch.long for t in inputs)
    assert inputs[24][0, 0].item() == 24
    array = path.parent / "final-0001.tokens.npy"
    raw = bytearray(array.read_bytes()); raw[-1] ^= 1; array.write_bytes(raw)
    with pytest.raises(ValueError, match="digest"):
        exp.load_panel(path)


def test_panel_handoff_tamper_refuses(sealed_panel):
    path, panel = sealed_panel
    panel["windows"][0]["role"] = "fit"
    path.write_text(json.dumps(panel))
    with pytest.raises(ValueError, match="digest"):
        exp.load_panel(path)


@pytest.mark.parametrize("bad", ["reorder", "omit", "repeat"])
def test_teacher_output_consumer_refuses_changed_order(bad):
    inputs = [torch.tensor([[i]]) for i in range(3)]
    class Runner:
        def visit_layer_batches(self, batches, visitor, output_consumer):
            for layer in range(2):
                visited = []
                visitor(layer, lambda t: visited.append(t.item()))
                assert visited == [0, 1, 2]
            order = {"reorder": [1, 0, 2], "omit": [0, 1], "repeat": [0, 0, 1]}[bad]
            for i in order:
                output_consumer(i, torch.zeros(1))
    with pytest.raises(ValueError):
        teacher.visit_panel(Runner(), inputs, lambda *a: None)


def test_teacher_visits_each_layer_once_over_every_ordered_window():
    inputs = [torch.tensor([[i]]) for i in range(25)]
    visits, outputs = [], []
    class Runner:
        def visit_layer_batches(self, batches, visitor, output_consumer):
            assert batches is inputs
            for layer in range(4):
                visitor(layer, lambda t: visits.append((layer, t.item())))
            for i in range(25):
                output_consumer(i, torch.tensor(i))
    teacher.visit_panel(Runner(), inputs, lambda i, logits: outputs.append((i, logits.item())))
    assert visits == [(layer, i) for layer in range(4) for i in range(25)]
    assert outputs == [(i, i) for i in range(25)]


def test_source_binding_compares_every_real_shard_and_small_file(tmp_path):
    roster, shards = [], []
    for i in range(126):
        name = f"model-{i:05d}.safetensors" if i < 120 else f"metadata-{i}.json"
        raw = f"file{i}".encode(); digest = hashlib.sha256(raw).hexdigest()
        (tmp_path / name).write_bytes(raw)
        roster.append({"name": name, "capture_sha256": digest, "upstream_sha256": digest,
                       "size_bytes": len(raw)})
        if i < 120: shards.append({"name": name, "size": len(raw), "sha256": digest})
    binding = {"schema": "root-source-revision-binding-v1", "repo": "zai-org/GLM-5.3-Flash-BF16",
               "revision": exp.REFERENCE_REVISION, "all_matched": True,
               "safetensors_count": 120, "source_files": roster}
    identity = {"shards": shards}
    teacher.require_reference_binding(binding, identity, tmp_path)
    bad = copy.deepcopy(identity); bad["shards"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="shard roster"):
        teacher.require_reference_binding(binding, bad, tmp_path)
    (tmp_path / "metadata-125.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="metadata"):
        teacher.require_reference_binding(binding, identity, tmp_path)


def test_declared_language_model_accessor_resolves_only_supported_owner():
    owner = type("Glm5NextForCausalLM", (), {})()
    wrapper_type = type("Glm5NextForConditionalGeneration", (), {
        "get_language_model": lambda self: self.language_model})
    wrapper = wrapper_type(); wrapper.language_model = owner
    assert served.language_model_owner(wrapper)[0] is owner
    assert served.language_model_owner(owner)[0] is owner
    wrapper.language_model = object()
    with pytest.raises(ValueError, match="declared language"):
        served.language_model_owner(wrapper)
    with pytest.raises(ValueError, match="unsupported"):
        served.language_model_owner(type("SomeWrapper", (), {"language_model": owner})())


def test_observed_engine_contract_refuses_silent_kv_promotion_and_prefix_cache():
    from types import SimpleNamespace as S
    config = S(model_config=S(enforce_eager=True, max_model_len=2049, logprobs_mode="raw_logprobs",
                             multimodal_config=S(language_model_only=True)),
               cache_config=S(enable_prefix_caching=False, cache_dtype="fp8_ds_mla"),
               scheduler_config=S(enable_chunked_prefill=False, max_num_seqs=1, max_num_batched_tokens=2049),
               parallel_config=S(pipeline_parallel_size=1, data_parallel_size=1), speculative_config=None)
    llm = S(llm_engine=S(vllm_config=config))
    with pytest.raises(ValueError, match="cache_config"):
        served.observed_engine_configuration(llm, expected_kv_cache_dtype="bfloat16")
    assert served.observed_engine_configuration(llm, expected_kv_cache_dtype="fp8_ds_mla")["language_model_only"]
    config.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match="cache_config"):
        served.observed_engine_configuration(llm, expected_kv_cache_dtype="fp8_ds_mla")
    config.cache_config.enable_prefix_caching = False
    config.model_config.multimodal_config.language_model_only = False
    with pytest.raises(ValueError, match="language-model-only"):
        served.observed_engine_configuration(llm, expected_kv_cache_dtype="fp8_ds_mla")


def test_scorer_engine_kwargs_carry_glm53_nope_runtime_selection():
    from types import SimpleNamespace as S
    args = S(kv_cache_dtype="fp8_ds_mla", gpu_memory_utilization=.9,
             attention_backend="CUSTOM",
             kernel_config='{"enable_flashinfer_autotune": false}', quantization=None)
    kwargs = served.scorer_engine_kwargs(args, model="candidate", topology={
        "tensor_parallel_size": 2, "moe_backend": "triton"})
    assert kwargs["attention_backend"] == "CUSTOM"
    assert kwargs["kernel_config"] == {"enable_flashinfer_autotune": False}
    assert kwargs["enforce_eager"] is True
    assert kwargs["moe_backend"] == "triton"


def test_scorer_engine_kwargs_carry_the_explicit_kv_bound_and_moe_backend():
    from types import SimpleNamespace as S
    args = S(kv_cache_dtype="fp8_ds_mla", gpu_memory_utilization=.25,
             attention_backend="CUSTOM",
             kernel_config='{"enable_flashinfer_autotune": false}', quantization=None)
    kwargs = served.scorer_engine_kwargs(args, model="candidate", topology={
        "tensor_parallel_size": 2, "nnodes": 2,
        "moe_backend": "flashinfer_cutlass", "kv_cache_memory_bytes": 4294967296})
    assert kwargs["moe_backend"] == "flashinfer_cutlass"
    assert kwargs["kv_cache_memory_bytes"] == 4294967296


def test_scorer_cli_accepts_the_explicit_kv_bound_and_cutlass(monkeypatch):
    """The tr3 scorer takes the same shared options as the two gold runners:
    run.sh passes --kv-cache-memory-bytes and MOE_BACKEND=flashinfer_cutlass
    straight through to its parser, and they must reach the engine kwargs."""
    from tools.gold_engine_options import gold_engine_kwargs
    parsed = {}
    monkeypatch.setattr(served, "measure", lambda args: parsed.update(vars(args)))
    argv = ["measure_glm_tr3_vllm.py", "--model", "candidate",
            "--candidate-digest-cache", "cache", "--panel", "panel",
            "--arrays-root", "arrays", "--teacher", "teacher.json",
            "--teacher-sha256", "a" * 64,
            "--serve-image", "image@sha256:" + "b" * 64,
            "--output", "result.json",
            "--kv-cache-dtype", "fp8_ds_mla",
            "--expected-kv-cache-dtype", "fp8_ds_mla",
            "--qualify-hook",
            "--moe-backend", "flashinfer_cutlass",
            "--kv-cache-memory-bytes", "4294967296"]
    monkeypatch.setattr(sys, "argv", argv)
    served.main()
    topology = gold_engine_kwargs(types.SimpleNamespace(**parsed))
    assert topology["moe_backend"] == "flashinfer_cutlass"
    assert topology["kv_cache_memory_bytes"] == 4294967296


def test_worker_observation_refuses_promotion_hidden_from_coordinator():
    from types import SimpleNamespace as S
    config = S(model_config=S(enforce_eager=True, max_model_len=2049, logprobs_mode="raw_logprobs",
                             multimodal_config=S(language_model_only=True)),
               cache_config=S(enable_prefix_caching=False, cache_dtype="auto"),
               scheduler_config=S(enable_chunked_prefill=False, max_num_seqs=1, max_num_batched_tokens=2049),
               parallel_config=S(pipeline_parallel_size=1, data_parallel_size=1), speculative_config=None)
    # A valid coordinator snapshot says nothing about worker-local mutation.
    served.observed_engine_configuration(S(llm_engine=S(vllm_config=config)), expected_kv_cache_dtype="auto")
    split = served.observed_engine_configuration(
        S(llm_engine=S(vllm_config=config)), expected_kv_cache_dtype="fp8_ds_mla",
        requested_kv_cache_dtype="auto")
    assert split["cache_config"]["cache_dtype"] == "auto"
    with pytest.raises(ValueError, match="neither requested nor declared"):
        served.observed_engine_configuration(S(llm_engine=S(vllm_config=config)),
                                             expected_kv_cache_dtype="fp8_ds_mla",
                                             requested_kv_cache_dtype="bfloat16")
    worker = S(vllm_config=copy.deepcopy(config), cache_config=S(cache_dtype="auto"),
               model_runner=S(cache_config=S(cache_dtype="fp8_ds_mla"), kv_cache_dtype=torch.uint8,
                              model=S(_tr3_capture=S(rank=1))))
    with pytest.raises(ValueError, match="worker/runner KV"):
        served.observed_worker_configuration(worker, expected_kv_cache_dtype="auto")
    worker.model_runner.cache_config.cache_dtype = "auto"
    worker.model_runner.kv_cache_dtype = torch.bfloat16
    assert served.observed_worker_configuration(worker, expected_kv_cache_dtype="auto")["runner_initial_kv_dtype"] == "torch.bfloat16"
    with pytest.raises(ValueError, match="native V2 prompt worker"):
        served.observed_worker_configuration(worker, expected_kv_cache_dtype="auto",
                                             logits_layout="vllm_v2_chunk1024")
    worker.vllm_config.cache_config.cache_dtype = "fp8_ds_mla"
    with pytest.raises(ValueError, match="cache_config"):
        served.observed_worker_configuration(worker, expected_kv_cache_dtype="auto")


def test_attention_receipt_reports_allocated_storage_separately_from_cache_policy():
    class Backend:
        pass
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kv_cache_dtype = "fp8_ds_mla"
            self.kv_cache = torch.empty((2, 4), dtype=torch.uint8)
        def get_attn_backend(self):
            return Backend
    row, = served.attention_runtime(Attention())
    assert row["kv_cache_dtype"] == "fp8_ds_mla"
    assert row["allocated_kv_cache"] == {"dtype": "torch.uint8", "shape": [2, 4], "device": "cpu"}


def test_initialized_runtime_observation_preserves_raw_state_without_claiming_score(tmp_path):
    binding = {"worker_runtime": [{"rank": 0, "allocated_kv_cache": {
        "dtype": "torch.uint8", "shape": [100, 64, 512], "device": "cuda:0"}}]}
    output = tmp_path / "hook-qualification.json"
    first = served.write_runtime_observation(output, binding)
    observed = json.loads(first.read_text())
    assert observed["runtime_binding"] == binding
    assert observed["stage"] == "initialized_before_scoring" and observed["scored_windows"] == 0
    assert "passed" not in observed and not output.exists()
    binding["worker_runtime"][0]["allocated_kv_cache"]["shape"][0] = 101
    second = served.write_runtime_observation(output, binding)
    assert second != first and first.exists() and second.exists()
    assert json.loads(first.read_text()) == observed


@pytest.mark.parametrize("observer_fails", [False, True])
def test_teacher_manifest_waits_for_both_box_observer_completion(tmp_path, monkeypatch, observer_fails):
    import sys
    from experiments import glm_full_capture_profile, workspace_netdata
    events = []
    class Observer:
        def __init__(self, out, **kwargs):
            self.out = out; out.mkdir(); self.result = {"status": "complete"}
        def __enter__(self):
            events.append("start"); return self
        def __exit__(self, *args):
            events.append("stop")
            assert not (tmp_path / "artifact/teacher.json").exists()
            for name in ("netdata.jsonl", "python_sampler.jsonl", "result.json"):
                (self.out / name).write_text("{}\n")
            if observer_fails:
                raise RuntimeError("observer incomplete")
    def build(args):
        events.append("build"); (tmp_path / "artifact").mkdir(); return {"teacher": "complete"}
    monkeypatch.setattr(glm_full_capture_profile, "CaptureObserver", Observer)
    monkeypatch.setattr(workspace_netdata, "sample_netdata", lambda host: {"host": host})
    monkeypatch.setattr(teacher, "build", build)
    argv = ["teacher"]
    for flag in ("model", "identity-cache", "panel", "reference-binding", "reference-binding-sha256",
                 "source-derivative-json", "source-derivative-sha256", "offload-folder"):
        argv += ["--" + flag, "unused"]
    argv += ["--output-dir", str(tmp_path / "artifact")]
    monkeypatch.setattr(sys, "argv", argv)
    if observer_fails:
        with pytest.raises(RuntimeError, match="observer incomplete"):
            teacher.main()
        assert not (tmp_path / "artifact/teacher.json").exists()
    else:
        teacher.main()
        result = json.loads((tmp_path / "artifact/teacher.json").read_text())
        assert result["host_telemetry"]["hosts"] == ["sparky", "sparklina"]
        assert len(result["host_telemetry"]["files"]) == 5
    assert events == ["start", "build", "stop"]


def test_exl3_receipt_requires_postscore_calls_on_every_rank():
    before = [{"rank": rank, "exl3_source_sha256": "a"*64,
               "exl3": {"tp_rank": rank, "tp_size": 2, "prefill_layer_calls": 0}}
              for rank in range(2)]
    with pytest.raises(ValueError, match="counter delta"):
        served.verify_exl3_route_delta(before, before, world_size=2)
    after = copy.deepcopy(before)
    for row in after: row["exl3"]["prefill_layer_calls"] = 45
    served.verify_exl3_route_delta(before, after, world_size=2)
    after[1]["rank"] = 0
    with pytest.raises(ValueError, match="every TP rank"):
        served.verify_exl3_route_delta(before, after, world_size=2)


def test_candidate_authentication_matches_real_manifest_bytes(tmp_path):
    rows = []
    shards = []
    for name in ["model-01.safetensors", "model-02.safetensors", "config.json", "tokenizer.json"]:
        raw = name.encode(); digest = hashlib.sha256(raw).hexdigest()
        (tmp_path / name).write_bytes(raw)
        rows.append(f"{digest}  {name}\n")
        if name.endswith(".safetensors"):
            shards.append({"name": name, "sha256": digest, "size": len(raw)})
    path = tmp_path / "SHA256SUMS"; path.write_text("".join(rows))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    authentication.require_checksum_binding(path, digest, {"shards": shards}, tmp_path)
    with pytest.raises(ValueError, match="shard set"):
        authentication.require_checksum_binding(path, digest, {"shards": shards[:1]}, tmp_path)
    (tmp_path / "config.json").write_bytes(b"edited")
    with pytest.raises(ValueError, match="metadata"):
        authentication.require_checksum_binding(path, digest, {"shards": shards}, tmp_path)


def test_missing_digest_cache_cannot_trigger_implicit_full_rehash(tmp_path, monkeypatch):
    from prismaquant import cost_streaming
    (tmp_path / "model.safetensors").write_bytes(b"not loaded")
    monkeypatch.setattr(cost_streaming, "build_source_checkpoint_identity",
                        lambda *a, **k: pytest.fail("must not hash missing/stale cache"))
    with pytest.raises(ValueError, match="incomplete/stale"):
        exp.cached_checkpoint_identity(tmp_path, tmp_path / "missing.json")


class _FakeOutput:
    def __init__(self, tokens):
        self.prompt_token_ids = list(tokens)


class _FakeLLM:
    """Two TP ranks behind the public LLM surface measure() drives."""
    instances = []

    def __init__(self, *, bad_window=None, **kwargs):
        self.kwargs, self.bad_window = kwargs, bad_window
        self.armed, self.generated, self.removed = [], 0, False
        _FakeLLM.instances.append(self)

    def apply_model(self, fn):
        name = getattr(fn, "func", fn).__name__
        keywords = getattr(fn, "keywords", {})
        if name == "install_capture":
            attention = [{"module": "layers.0.self_attn.indexer",
                          "backend": "vllm.v1.attention.backends.mla.indexer.DeepseekV32IndexerBackend",
                          "allocated_kv_cache": {"dtype": "torch.uint8", "shape": [7, 64, 132],
                                                 "device": "cuda:0"}}]
            return [{"rank": rank, "world_size": 2, "attention_runtime": copy.deepcopy(attention)}
                    for rank in (1, 0)]
        if name == "route_diagnostics":
            return [{"rank": 0, "exl3": None}, {"rank": 1, "exl3": None}]
        if name == "arm_capture":
            self.armed.append(keywords["index"])
            return [{"rank": 0, "teacher_resident": True}, {"rank": 1, "teacher_resident": False}]
        if name == "finish_capture":
            rows, vocab = exp.CONTEXT_LENGTH - 1, exp.VOCAB_SIZE
            fill = float("nan") if keywords["window_id"] == self.bad_window else 0.25
            owner = {"rank": 0, "world_size": 2, "window_id": keywords["window_id"],
                     "calls": [(1, vocab), (rows, vocab)], "values": [fill] * rows,
                     "logits_layout": "legacy_single"}
            other = {"rank": 1, "world_size": 2, "window_id": keywords["window_id"],
                     "calls": [None, None], "values": None, "logits_layout": "legacy_single"}
            return [owner, other]
        if name == "remove_capture":
            self.removed = True
            return [True, True]
        raise AssertionError(f"unexpected apply_model call {name}")

    def collective_rpc(self, fn, kwargs=None):
        return [{"rank": 1}, {"rank": 0}]

    def generate(self, prompts, params, use_tqdm=False):
        self.generated += 1
        return [_FakeOutput(prompts[0]["prompt_token_ids"])]


@pytest.fixture
def fake_measure(tmp_path, monkeypatch):
    """measure() with every identity source and the engine replaced by fakes."""
    windows = [{"window_id": f"w{i}"} for i in range(3)]
    tokens = [(np.arange(exp.CONTEXT_LENGTH) + i,) for i in range(3)]
    teacher_value = {"tokenizer_identity": {"tok": 1}, "source_execution": {"teacher": 1},
                     "windows": [{"window_id": w["window_id"]} for w in windows]}
    model = tmp_path / "model"
    model.mkdir()
    teacher_path = tmp_path / "teacher.json"
    real_bound_json = served.bound_json
    monkeypatch.setattr(served, "load_panel", lambda path, arrays_root=None: ({"windows": windows}, tokens))
    monkeypatch.setattr(served, "load_teacher", lambda path, digest, panel: teacher_value)
    monkeypatch.setattr(served, "bound_json", lambda path, digest: (
        teacher_value if Path(path) == teacher_path else real_bound_json(path, digest)))
    monkeypatch.setattr(served, "sha256", lambda path: served.TOKENIZER_SHA256)
    monkeypatch.setattr(served, "tokenizer_identity", lambda path: {"tok": 1})
    monkeypatch.setattr(served, "cached_checkpoint_identity", lambda path, cache: {"ckpt": 1})
    monkeypatch.setattr(served, "producer_identity", lambda: {"producer": 1})
    monkeypatch.setattr(served, "gold_engine_kwargs", lambda args: {"tensor_parallel_size": 2})
    monkeypatch.setattr(served, "refuse_if_spec_decode", lambda llm, context: False)
    monkeypatch.setattr(served, "observed_engine_configuration", lambda llm, **kw: {"observed": 1})
    monkeypatch.setattr(served, "verify_prompt_alignment", lambda output, tokens, reports: {"aligned": 1})
    monkeypatch.setattr(served, "summarize_panel", lambda panel, vectors: {"windows": len(vectors)})
    monkeypatch.setattr(served, "self_manifest", lambda image, extra: {"image": image})
    state = {"bad_window": None}
    _FakeLLM.instances = []
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(
        LLM=lambda **kwargs: _FakeLLM(bad_window=state["bad_window"], **kwargs),
        SamplingParams=lambda **kwargs: kwargs))
    args = types.SimpleNamespace(
        model=str(model), candidate_digest_cache="cache", panel="panel", arrays_root=None,
        teacher=str(teacher_path), teacher_sha256="a" * 64,
        serve_image="image@sha256:" + "b" * 64, output=str(tmp_path / "full.json"),
        kv_cache_dtype="fp8_ds_mla", expected_kv_cache_dtype="fp8_ds_mla",
        attention_backend=None, kernel_config=None, quantization=None, require_exl3_diag=False,
        gpu_memory_utilization=.5, tile_rows=32, logits_layout="legacy_single",
        qualify_hook=False, qualification=None, qualification_sha256=None,
        qualify_then_score=str(tmp_path / "qualification.json"))
    return args, state


def test_qualify_then_score_gates_the_panel_on_the_real_replay_check(fake_measure, monkeypatch):
    args, _ = fake_measure
    seen = []
    real_check = served.require_native_qualification

    def spy(qualification, runtime_binding):
        seen.append(qualification)
        return real_check(qualification, runtime_binding)

    monkeypatch.setattr(served, "require_native_qualification", spy)
    result = served.measure(args)
    (llm,) = _FakeLLM.instances
    assert llm.armed == [0, 1, 2] and llm.generated == 3 and llm.removed
    raw = Path(args.qualify_then_score).read_bytes()
    record = json.loads(raw)
    assert record["schema"] == "prismaquant.glm_tr3_hook_qualification/1" and record["passed"] is True
    assert len(record["per_position_kl"]) == 1 and seen == [record]
    written = json.loads(Path(args.output).read_bytes())
    assert written == json.loads(json.dumps(result))
    assert written["schema"] == "prismaquant.glm_tr3_full_vocabulary_kl/1"
    assert written["qualification"] == {"mode": "in_process", "path": args.qualify_then_score,
                                        "sha256": hashlib.sha256(raw).hexdigest(), "window_reused": True}
    assert written["per_position_kl"][0] == record["per_position_kl"][0]
    assert written["runtime_binding"] == record["runtime_binding"]
    assert len(written["per_position_kl"]) == 3


def test_qualify_then_score_refusal_writes_no_panel_result(fake_measure, monkeypatch):
    args, _ = fake_measure

    def refuse(qualification, runtime_binding):
        raise ValueError("native qualification differs from this candidate/runtime/teacher/topology")

    monkeypatch.setattr(served, "require_native_qualification", refuse)
    with pytest.raises(ValueError, match="native qualification differs"):
        served.measure(args)
    (llm,) = _FakeLLM.instances
    assert llm.armed == [0] and llm.generated == 1 and llm.removed
    assert Path(args.qualify_then_score).exists()
    assert not Path(args.output).exists()


def test_qualify_then_score_window_zero_failure_writes_neither_file(fake_measure):
    args, state = fake_measure
    state["bad_window"] = "w0"
    with pytest.raises(ValueError, match="nonfinite"):
        served.measure(args)
    assert not Path(args.qualify_then_score).exists()
    assert not Path(args.output).exists()


def test_two_process_modes_are_unchanged_by_the_in_process_mode(fake_measure):
    args, _ = fake_measure
    args.qualify_then_score, args.qualify_hook = None, True
    qualification = served.measure(args)
    assert "qualification" not in qualification
    raw = Path(args.output).read_bytes()
    assert json.loads(raw)["schema"] == "prismaquant.glm_tr3_hook_qualification/1"
    args.qualify_hook, args.output = False, str(Path(args.output).with_name("panel.json"))
    args.qualification, args.qualification_sha256 = str(Path(args.output).with_name("full.json")), \
        hashlib.sha256(raw).hexdigest()
    full = served.measure(args)
    assert full["schema"] == "prismaquant.glm_tr3_full_vocabulary_kl/1" and "qualification" not in full
    assert len(full["per_position_kl"]) == 3 and _FakeLLM.instances[-1].armed == [0, 1, 2]


@pytest.mark.parametrize("other", ["qualify_hook", "qualification", "qualification_sha256", "same_path"])
def test_qualify_then_score_refuses_a_second_qualification_mode(fake_measure, other):
    args, _ = fake_measure
    if other == "same_path":
        args.qualify_then_score = args.output
    else:
        setattr(args, other, True if other == "qualify_hook" else "c" * 64)
    with pytest.raises(ValueError, match="qualif|distinct paths"):
        served.measure(args)
    assert _FakeLLM.instances == []


def _npy_window(tmp_path, array, name="window.npy"):
    path = tmp_path / name
    np.save(path, array, allow_pickle=False)
    raw = path.read_bytes()
    return path, {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                  "shape": list(array.shape)}


@pytest.mark.parametrize("order", ["C", "F"])
def test_teacher_window_single_read_matches_np_load_bitwise(tmp_path, order):
    source = np.asarray(np.random.default_rng(3).standard_normal((33, 257)), dtype=np.float32, order=order)
    source[0, :4] = [np.inf, -np.inf, -0.0, np.float32(1e-45)]
    path, descriptor = _npy_window(tmp_path, source)
    got = served.load_teacher_window(path, descriptor)
    want = np.load(io.BytesIO(path.read_bytes()), allow_pickle=False)
    assert got.dtype == want.dtype and got.shape == want.shape
    assert got.tobytes(order="C") == want.tobytes(order="C")
    base = got
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    owner = base.obj if isinstance(base, memoryview) else base
    assert isinstance(owner, bytearray) and len(owner) == descriptor["bytes"]
    assert got.flags.writeable  # torch.from_numpy takes it without a copy or warning


def test_teacher_window_single_read_holds_one_window_on_the_host(tmp_path):
    import tracemalloc
    path, descriptor = _npy_window(tmp_path, np.ones((256, 4096), dtype=np.float32))
    size = descriptor["bytes"]

    def peak(load):
        tracemalloc.start()
        try:
            value = load()
            return tracemalloc.get_traced_memory()[1], value
        finally:
            tracemalloc.stop()

    single, _ = peak(lambda: served.load_teacher_window(path, descriptor))
    double, _ = peak(lambda: np.load(io.BytesIO(path.read_bytes()), allow_pickle=False))
    assert single < 1.1 * size < 1.9 * size < double


@pytest.mark.parametrize("mutation", ["flip", "truncate", "extend", "descriptor_bytes"])
def test_teacher_window_single_read_refuses_changed_bytes(tmp_path, mutation):
    path, descriptor = _npy_window(tmp_path, np.zeros((8, 16), dtype=np.float32))
    raw = bytearray(path.read_bytes())
    if mutation == "flip":
        raw[-1] ^= 1
    elif mutation == "truncate":
        raw = raw[:-4]
    elif mutation == "extend":
        raw += b"\0\0\0\0"
    else:
        descriptor = {**descriptor, "bytes": descriptor["bytes"] - 4}
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="teacher window bytes changed|geometry"):
        served.load_teacher_window(path, descriptor)


def test_arm_capture_teacher_tensor_matches_the_previous_two_copy_path(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("arm_capture preloads the teacher on CUDA; run on a GB10 worker")
    source = np.random.default_rng(7).standard_normal((31, 129)).astype(np.float32)
    path, descriptor = _npy_window(tmp_path, source)
    armed = {}

    class State:
        rank = 0

        def arm(self, index, window_id, teacher, targets):
            armed.update(teacher=teacher, targets=targets)

    model = types.SimpleNamespace(_tr3_capture=State())
    served.arm_capture(model, index=0, window_id="w0", descriptor=descriptor,
                       teacher_root=str(tmp_path), target_ids=list(range(31)))
    previous = torch.from_numpy(np.load(io.BytesIO(path.read_bytes()), allow_pickle=False)).to("cuda")
    assert armed["teacher"].device.type == "cuda" and armed["teacher"].dtype == torch.float32
    assert torch.equal(armed["teacher"], previous)


# --- compiled execution mode (PQ #1634) ---------------------------------------

class _CompilationMode(enum.IntEnum):
    """The member names of the pinned vLLM's CompilationMode."""
    NONE = 0
    STOCK_TORCH_COMPILE = 1
    DYNAMO_TRACE_ONCE = 2
    VLLM_COMPILE = 3


class _CUDAGraphMode(enum.Enum):
    """The member names of the pinned vLLM's CUDAGraphMode."""
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (2, 0)
    FULL_AND_PIECEWISE = (2, 1)


_FDO = {"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 3, 4]}


def _scorer_args(**overrides):
    from types import SimpleNamespace as S
    values = dict(kv_cache_dtype="fp8_ds_mla", gpu_memory_utilization=.9, attention_backend="CUSTOM",
                  kernel_config='{"enable_flashinfer_autotune": false}', quantization=None)
    values.update(overrides)
    return S(**values)


def _resolved_config(*, enforce_eager=False, mode=_CompilationMode.NONE,
                     cudagraph_mode=_CUDAGraphMode.FULL_DECODE_ONLY, sizes=(1, 2, 3, 4)):
    from types import SimpleNamespace as S
    return S(model_config=S(enforce_eager=enforce_eager, max_model_len=2049, logprobs_mode="raw_logprobs",
                            multimodal_config=S(language_model_only=True)),
             cache_config=S(enable_prefix_caching=False, cache_dtype="fp8_ds_mla"),
             scheduler_config=S(enable_chunked_prefill=False, max_num_seqs=1, max_num_batched_tokens=2049),
             parallel_config=S(pipeline_parallel_size=1, data_parallel_size=1), speculative_config=None,
             compilation_config=S(mode=mode, cudagraph_mode=cudagraph_mode,
                                  cudagraph_capture_sizes=list(sizes)))


def test_eager_engine_kwargs_are_unchanged_by_the_compiled_mode():
    """An eager run, stated or not, builds the engine it always built: eager on
    and no compilation config, so every eager qualification still replays."""
    topology = {"tensor_parallel_size": 2, "moe_backend": "triton"}
    legacy = served.scorer_engine_kwargs(_scorer_args(), model="candidate", topology=topology)
    stated = served.scorer_engine_kwargs(_scorer_args(execution_mode="eager", compilation_config=None),
                                         model="candidate", topology=topology)
    assert legacy == stated
    assert legacy["enforce_eager"] is True and "compilation_config" not in legacy


def test_compiled_engine_kwargs_turn_eager_off_and_carry_the_declared_config():
    kwargs = served.scorer_engine_kwargs(
        _scorer_args(execution_mode="compiled", compilation_config=json.dumps(_FDO)),
        model="candidate", topology={"tensor_parallel_size": 2})
    assert kwargs["enforce_eager"] is False
    assert kwargs["compilation_config"] == _FDO
    assert kwargs["max_num_seqs"] == 1 and kwargs["enable_chunked_prefill"] is False


@pytest.mark.parametrize("text", [
    "not json",
    "[1, 2]",
    json.dumps({k: v for k, v in _FDO.items() if k != "mode"}),
    json.dumps({**_FDO, "max_cudagraph_capture_size": 4}),
    json.dumps({**_FDO, "mode": "none"}),
    json.dumps({**_FDO, "mode": 0}),
    json.dumps({**_FDO, "cudagraph_mode": "NONE"}),
    json.dumps({**_FDO, "cudagraph_capture_sizes": []}),
    json.dumps({**_FDO, "cudagraph_capture_sizes": [2, 1]}),
    json.dumps({**_FDO, "cudagraph_capture_sizes": [1, 1, 2]}),
    json.dumps({**_FDO, "cudagraph_capture_sizes": [0, 1]}),
    json.dumps({**_FDO, "cudagraph_capture_sizes": [True, 2]}),
    json.dumps({**_FDO, "cudagraph_capture_sizes": [1, 2050]}),
    json.dumps({**_FDO, "cudagraph_capture_sizes": 4}),
])
def test_a_compilation_config_is_refused_unless_every_field_is_stated_canonically(text):
    """LLM() drops an unknown compilation key silently, and the engine may
    resolve a loose value to something else; both refuse before any load."""
    with pytest.raises(ValueError, match="compilation config|cudagraph"):
        served.parse_compilation_config(text)


@pytest.mark.parametrize("mode, text, match", [
    ("eager", json.dumps(_FDO), "applies only to --execution-mode compiled"),
    ("compiled", None, "requires --compilation-config"),
    ("graph", None, "execution mode must be one of"),
])
def test_the_declared_mode_and_the_config_must_agree(mode, text, match):
    with pytest.raises(ValueError, match=match):
        served.declared_compilation(_scorer_args(execution_mode=mode, compilation_config=text))


def test_compiled_observation_records_the_resolved_config():
    observed = served.observed_configuration(_resolved_config(), expected_kv_cache_dtype="fp8_ds_mla",
                                             compilation=_FDO)
    assert observed["model_config"]["enforce_eager"] is False
    assert observed["compilation_config"] == _FDO
    eager = served.observed_configuration(_resolved_config(enforce_eager=True),
                                          expected_kv_cache_dtype="fp8_ds_mla")
    assert "compilation_config" not in eager and eager["model_config"]["enforce_eager"] is True


@pytest.mark.parametrize("declared, resolved, match", [
    # Breakable CUDA graphs force mode NONE on GLM-5.3: a declared VLLM_COMPILE
    # would otherwise score an engine that never compiled.
    ({**_FDO, "mode": "VLLM_COMPILE"}, {}, "differs from the declared compiled contract: mode"),
    (_FDO, dict(cudagraph_mode=_CUDAGraphMode.PIECEWISE), "contract: cudagraph_mode"),
    (_FDO, dict(sizes=(1, 2, 4)), "contract: cudagraph_capture_sizes"),
    (_FDO, dict(enforce_eager=True), "model_config"),
])
def test_compiled_observation_refuses_a_config_the_engine_did_not_resolve(declared, resolved, match):
    with pytest.raises(ValueError, match=match):
        served.observed_configuration(_resolved_config(**resolved), expected_kv_cache_dtype="fp8_ds_mla",
                                      compilation=declared)


def test_the_eager_contract_still_refuses_an_engine_that_is_not_eager():
    with pytest.raises(ValueError, match="model_config"):
        served.observed_configuration(_resolved_config(), expected_kv_cache_dtype="fp8_ds_mla")


def test_compiled_observation_still_refuses_speculative_decoding():
    config = _resolved_config()
    config.speculative_config = object()
    with pytest.raises(ValueError, match="speculative"):
        served.observed_configuration(config, expected_kv_cache_dtype="fp8_ds_mla", compilation=_FDO)


def test_a_worker_that_resolved_another_graph_mode_refuses():
    """The model runner resolves the CUDA-graph mode against its attention
    backends after the coordinator's snapshot; the worker's own config is read."""
    from types import SimpleNamespace as S
    worker = S(vllm_config=_resolved_config(cudagraph_mode=_CUDAGraphMode.NONE),
               cache_config=S(cache_dtype="fp8_ds_mla"),
               model_runner=S(cache_config=S(cache_dtype="fp8_ds_mla"), kv_cache_dtype=torch.uint8,
                              model=S(_tr3_capture=S(rank=1))))
    with pytest.raises(ValueError, match="cudagraph_mode"):
        served.observed_worker_configuration(worker, expected_kv_cache_dtype="fp8_ds_mla", compilation=_FDO)
    worker.vllm_config.compilation_config.cudagraph_mode = _CUDAGraphMode.FULL_DECODE_ONLY
    row = served.observed_worker_configuration(worker, expected_kv_cache_dtype="fp8_ds_mla", compilation=_FDO)
    assert row["configuration"]["compilation_config"] == _FDO


def _scorer_argv(*extra):
    return ["measure_glm_tr3_vllm.py", "--model", "candidate",
            "--candidate-digest-cache", "cache", "--panel", "panel",
            "--teacher", "teacher.json", "--teacher-sha256", "a" * 64,
            "--serve-image", "image@sha256:" + "b" * 64, "--output", "result.json",
            "--kv-cache-dtype", "fp8_ds_mla", "--expected-kv-cache-dtype", "fp8_ds_mla",
            "--qualify-hook", *extra]


def test_scorer_cli_takes_the_compiled_mode_and_its_config(monkeypatch):
    parsed = {}
    monkeypatch.setattr(served, "measure", lambda args: parsed.update(vars(args)))
    monkeypatch.setattr(sys, "argv", _scorer_argv("--execution-mode", "compiled",
                                                  "--compilation-config", json.dumps(_FDO)))
    served.main()
    assert parsed["execution_mode"] == "compiled"
    assert served.declared_compilation(types.SimpleNamespace(**parsed)) == _FDO
    parsed.clear()
    monkeypatch.setattr(sys, "argv", _scorer_argv())
    served.main()
    assert parsed["execution_mode"] == "eager" and parsed["compilation_config"] is None


@pytest.mark.parametrize("extra", [
    ("--compilation-config", json.dumps(_FDO)),
    ("--execution-mode", "compiled"),
    ("--execution-mode", "compiled", "--compilation-config", json.dumps({**_FDO, "extra": 1})),
    ("--execution-mode", "graph"),
])
def test_scorer_cli_refuses_a_mode_and_config_that_disagree(monkeypatch, extra):
    monkeypatch.setattr(served, "measure", lambda args: pytest.fail("must refuse before measuring"))
    monkeypatch.setattr(sys, "argv", _scorer_argv(*extra))
    with pytest.raises(SystemExit):
        served.main()


def test_compiled_peer_argv_states_eager_off_and_the_declared_config():
    """The headless peer must build the same compiled engine as rank 0."""
    from tools.gold_engine_options import headless_peer_argv, parse_headless_peer_argv
    kwargs = served.scorer_engine_kwargs(
        _scorer_args(execution_mode="compiled", compilation_config=json.dumps(_FDO)),
        model="candidate", topology={"tensor_parallel_size": 2, "nnodes": 2, "node_rank": 0,
                                     "master_addr": "192.0.2.1", "master_port": 29531})
    argv = headless_peer_argv(kwargs, node_rank=1)
    assert "--no-enforce-eager" in argv and "--enforce-eager" not in argv
    assert json.loads(argv[argv.index("--compilation-config") + 1]) == _FDO
    _, _, back = parse_headless_peer_argv(argv)
    assert back["enforce_eager"] is False and back["compilation_config"] == _FDO


def _record_observations(monkeypatch):
    calls = {"engine": [], "worker": []}

    def engine(llm, **kw):
        calls["engine"].append(kw.get("compilation"))
        return {"observed": kw.get("compilation")}

    def rpc(self, fn, kwargs=None):
        calls["worker"].append(dict(kwargs or {}))
        return [{"rank": 1}, {"rank": 0}]

    monkeypatch.setattr(served, "observed_engine_configuration", engine)
    monkeypatch.setattr(_FakeLLM, "collective_rpc", rpc)
    return calls


def test_compiled_binding_is_stamped_and_threads_the_config_to_every_observation(fake_measure, monkeypatch):
    args, _ = fake_measure
    calls = _record_observations(monkeypatch)
    args.execution_mode, args.compilation_config = "compiled", json.dumps(_FDO)
    result = served.measure(args)
    binding = result["runtime_binding"]
    assert binding["execution_mode"] == "compiled"
    assert binding["engine_kwargs"]["enforce_eager"] is False
    assert binding["engine_kwargs"]["compilation_config"] == _FDO
    # The initial observation and every recheck see the declared config, so a
    # compiled run is not refused at the end as a "changed" configuration.
    assert len(calls["engine"]) >= 2 and all(c == _FDO for c in calls["engine"])
    assert [c.get("compilation") for c in calls["worker"]] == [_FDO]


def test_an_eager_run_binds_exactly_as_before(fake_measure, monkeypatch):
    args, _ = fake_measure
    calls = _record_observations(monkeypatch)
    result = served.measure(args)
    assert "execution_mode" not in result["runtime_binding"]
    assert "compilation_config" not in result["runtime_binding"]["engine_kwargs"]
    assert all(c is None for c in calls["engine"])
    assert all("compilation" not in c for c in calls["worker"])


@pytest.mark.parametrize("qualified_mode", ["eager", "compiled"])
def test_a_qualification_from_the_other_mode_refuses(fake_measure, qualified_mode):
    """A compiled KL receipt never replays an eager qualification, nor the reverse."""
    args, _ = fake_measure

    def set_mode(mode):
        args.execution_mode = mode
        args.compilation_config = json.dumps(_FDO) if mode == "compiled" else None

    args.qualify_then_score, args.qualify_hook = None, True
    set_mode(qualified_mode)
    served.measure(args)
    raw = Path(args.output).read_bytes()
    args.qualify_hook = False
    args.qualification, args.qualification_sha256 = args.output, hashlib.sha256(raw).hexdigest()
    args.output = str(Path(args.output).with_name("panel.json"))
    set_mode("compiled" if qualified_mode == "eager" else "eager")
    with pytest.raises(ValueError, match="native qualification differs"):
        served.measure(args)
    assert not Path(args.output).exists()
