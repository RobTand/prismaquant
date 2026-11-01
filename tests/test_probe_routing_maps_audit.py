"""The public probe must keep route counts when it reuses phase 1."""
import pickle
import sys

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from prismaquant import genuine_weight_initialization, gpu_guard
from prismaquant import incremental_probe as owner
from test_probe_reuse_inputs_audit import _dataset, _source


@pytest.fixture
def probe_run(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _source(source)
    config = Qwen3MoeConfig(
        vocab_size=35, hidden_size=16, intermediate_size=32,
        moe_intermediate_size=8, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, head_dim=8, num_experts=4, num_experts_per_tok=2,
        max_position_embeddings=64, tie_word_embeddings=False,
        attention_dropout=0.0, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    with torch.random.fork_rng(), genuine_weight_initialization():
        torch.manual_seed(61)
        model = Qwen3MoeForCausalLM(config)
    model.save_pretrained(source, safe_serialization=True)
    data = tmp_path / "calibration.jsonl"
    _dataset(data, 109)
    monkeypatch.setenv("CACHE_HEADROOM_GB", "0")
    monkeypatch.setenv("PRISMAQUANT_LAYER_READ_THREADS", "1")
    monkeypatch.setenv("PRISMAQUANT_ACT_CACHE_WORKERS", "1")
    monkeypatch.setenv("PRISMAQUANT_PROBE_CTX_CACHE", "0")
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *a, **k: torch.device("cpu"))
    work = tmp_path / "work"
    output = tmp_path / "probe.pkl"
    monkeypatch.setattr(sys, "argv", [
        "incremental_probe", "--model", str(source), "--dataset", str(data),
        "--nsamples", "2", "--seqlen", "8", "--device", "cpu", "--dtype", "fp32",
        "--output", str(output), "--work-dir", str(work),
        "--activation-cache-dir", str(work / "act"), "--activation-rows-limit", "32",
        "--layers-per-shard", "1", "--prefetch-lookahead", "0",
        "--prefetch-workers", "1", "--prefetch-min-available-gb", "0",
        "--no-include-mtp", "--no-include-visual", "--no-include-lm-head",
        "--no-minimax-fast-moe",
    ])
    def run():
        owner.main()
        return pickle.loads(output.read_bytes())
    return run, work


def test_main_resume_keeps_router_population(probe_run, capsys):
    run, work = probe_run
    fresh = run()
    fields = ("router_counts", "router_totals", "router_active_counts", "expert_route_stats")
    assert all(fresh[field] for field in fields)
    assert sum(fresh["router_totals"].values()) == 16
    for shard in (work / "shards").glob("*.pkl"):
        shard.unlink()
    capsys.readouterr()
    resumed = run()
    assert "reused precompute cache" in capsys.readouterr().out
    assert {field: resumed[field] for field in fields} == {field: fresh[field] for field in fields}


def test_main_refuses_legacy_router_coverage(probe_run, capsys):
    run, work = probe_run
    fresh = run()
    shard = work / "shards" / "probe_shard_000.pkl"
    payload = pickle.loads(shard.read_bytes())
    for meta in (payload["meta"], payload["meta"].get("incremental_shard", {})):
        meta["router_coverage_version"] = 2
    for field in ("router_counts", "router_totals", "router_active_counts", "expert_route_stats"):
        payload[field] = {}
    shard.write_bytes(pickle.dumps(payload))
    capsys.readouterr()
    resumed = run()
    assert "reuse shard 0:" not in capsys.readouterr().out
    assert resumed["router_counts"] == fresh["router_counts"]
