"""The real probe caller must preserve pooled reuse across shard grouping."""
import pickle
import sys

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from prismaquant import genuine_weight_initialization, gpu_guard
from prismaquant import incremental_probe as probe
from test_probe_reuse_inputs_audit import _dataset, _source


@pytest.fixture
def pooled_run(tmp_path, monkeypatch):
    def prepare(packed):
        source = tmp_path / "source"
        source.mkdir()
        _source(source)
        if packed:
            config = Qwen3MoeConfig(
                vocab_size=35, hidden_size=16, intermediate_size=32,
                moe_intermediate_size=8, num_hidden_layers=2, num_attention_heads=2,
                num_key_value_heads=2, head_dim=8, num_experts=4, num_experts_per_tok=2,
                max_position_embeddings=64, tie_word_embeddings=False,
                attention_dropout=0.0, bos_token_id=1, eos_token_id=2, pad_token_id=0)
            model_class = Qwen3MoeForCausalLM
        else:
            config = LlamaConfig.from_pretrained(source)
            config.num_hidden_layers = 2
            model_class = LlamaForCausalLM
        with torch.random.fork_rng(), genuine_weight_initialization():
            torch.manual_seed(61)
            model = model_class(config)
        model.save_pretrained(source, safe_serialization=True)
        data = tmp_path / "calibration.jsonl"
        _dataset(data, 109)
        work = tmp_path / "work"
        output = tmp_path / "probe.pkl"
        for name, value in {
            "CACHE_HEADROOM_GB": "0", "PRISMAQUANT_LAYER_READ_THREADS": "1",
            "PRISMAQUANT_ACT_CACHE_WORKERS": "1", "PRISMAQUANT_PROBE_CTX_CACHE": "0",
        }.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *a, **k: torch.device("cpu"))

        def run(layers_per_shard):
            probe._W_STATS_CACHE.clear()
            monkeypatch.setattr(sys, "argv", [
                "incremental_probe", "--model", str(source), "--dataset", str(data),
                "--nsamples", "2", "--seqlen", "8", "--device", "cpu", "--dtype", "fp32",
                "--output", str(output), "--work-dir", str(work),
                "--activation-cache-dir", str(work / "act"), "--activation-rows-limit", "32",
                "--layers-per-shard", str(layers_per_shard), "--prefetch-lookahead", "0",
                "--prefetch-workers", "1", "--prefetch-min-available-gb", "0",
                "--no-include-mtp", "--no-include-visual", "--no-include-lm-head",
                "--no-minimax-fast-moe",
            ])
            probe.main()
            return pickle.loads(output.read_bytes())
        return run, work
    return prepare


def _set_coverage(work, version):
    for path in (work / "shards").glob("probe_shard_*.pkl"):
        data = pickle.loads(path.read_bytes())
        data["meta"]["router_coverage_version"] = version
        data["meta"]["incremental_shard"]["router_coverage_version"] = version
        path.write_bytes(pickle.dumps(data))


@pytest.mark.parametrize("packed,version", [(False, 3), (False, 2), (True, 3)])
def test_main_reuses_pooled_stats_after_grouping_change(pooled_run, monkeypatch, capsys, packed, version):
    run, work = pooled_run(packed)
    original = run(1)
    _set_coverage(work, version)
    capsys.readouterr()

    def recompute(*args, **kwargs):
        pytest.fail("valid pooled Linear stats must avoid probe recomputation")

    monkeypatch.setattr(probe, "_compute_global_precompute", recompute)
    monkeypatch.setattr(probe, "_run_body_streaming_shard", recompute)
    resumed = run(2)
    log = capsys.readouterr().out
    assert f"linear cache: {len(original['stats'])} stats pooled" in log
    assert "synthesize shard 0 (body): reused cached Linear stats" in log
    assert probe._cached_value_exact_equal(resumed["stats"], original["stats"])
    for field in probe._ROUTING_FIELDS:
        assert resumed[field] == original[field]
    data = pickle.loads((work / "shards" / "probe_shard_000.pkl").read_bytes())
    assert data["meta"]["synthesized_from_cache"]


def test_main_refuses_affected_packed_pool_after_grouping_change(pooled_run, capsys):
    run, work = pooled_run(True)
    original = run(1)
    assert original["router_counts"]
    _set_coverage(work, 2)
    capsys.readouterr()
    resumed = run(2)
    log = capsys.readouterr().out
    assert "linear cache:" not in log
    assert "synthesize shard" not in log
    assert "shard 0 (body): include=" in log
    assert resumed["router_counts"] == original["router_counts"]
