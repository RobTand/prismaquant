"""The public probe records the router population of its forward."""
import json
import pickle
import random
import sys

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from prismaquant import genuine_weight_initialization, gpu_guard
from prismaquant import incremental_probe as owner


def _source(root):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2,
             **{f"w{index}": index + 3 for index in range(32)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                           bos_token="[BOS]", eos_token="[EOS]", pad_token="[UNK]").save_pretrained(root)


def _dataset(path, seed):
    rng = random.Random(seed)
    records = [{"text": " ".join(f"w{rng.randrange(32)}" for _ in range(128))}
               for _ in range(12)]
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


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
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *a, **k: torch.device("cpu"))
    work = tmp_path / "work"
    output = tmp_path / "probe.pkl"
    monkeypatch.setattr(sys, "argv", [
        "incremental_probe", "--model", str(source), "--dataset", str(data),
        "--nsamples", "2", "--seqlen", "8", "--device", "cpu", "--dtype", "fp32",
        "--output", str(output), "--work-dir", str(work),
        "--activation-cache-dir", str(work / "act"), "--activation-rows-limit", "32",
        "--prefetch-lookahead", "0",
        "--prefetch-workers", "1", "--prefetch-min-available-gb", "0",
        "--no-include-mtp", "--no-include-visual", "--no-include-lm-head",
    ])
    def run():
        owner.main()
        return pickle.loads(output.read_bytes())
    return run, work


def test_main_records_router_population_and_default_act_rows(probe_run, monkeypatch):
    run, work = probe_run
    monkeypatch.delenv("ACTIVATION_ROWS_LIMIT", raising=False)
    row_arg = sys.argv.index("--activation-rows-limit")
    monkeypatch.setattr(sys, "argv", sys.argv[:row_arg] + sys.argv[row_arg + 2:] +
                        ["--nsamples", "12", "--seqlen", "64"])
    fresh = run()
    fields = ("router_counts", "router_totals", "router_active_counts", "expert_route_stats")
    assert all(fresh[field] for field in fields)
    assert sum(fresh["router_totals"].values()) == 768
    act = torch.load(work / "act/model__layers__0__self_attn__o_proj.pt", weights_only=False)
    assert fresh["meta"]["activation_rows_limit"] == act["inputs"].shape[0] == 512
