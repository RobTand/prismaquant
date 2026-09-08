"""CPU contracts only; native stock-vLLM hook qualification is separate."""
import copy
import hashlib
import io
import json

import numpy as np
import pytest
import torch

from experiments import glm_tr3_full_vocab as exp
from experiments import build_glm_tr3_teacher as teacher
from experiments import measure_glm_tr3_vllm as served
from experiments import authenticate_glm_tr3_source as authentication


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
