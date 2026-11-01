"""Real incremental probe reuse must preserve source and calibration inputs."""
from __future__ import annotations

import hashlib
import json
import pickle
import random
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from prismaquant import genuine_weight_initialization, gpu_guard
from prismaquant import incremental_probe as owner


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(root):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    config = LlamaConfig(
        vocab_size=35, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=False, attention_dropout=0.0,
        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    with torch.random.fork_rng(), genuine_weight_initialization():
        torch.manual_seed(61)
        model = LlamaForCausalLM(config)
    model.save_pretrained(root, safe_serialization=True)
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


def _mutate_source(path):
    from safetensors.torch import load_file, save_file

    values = {key: value.clone() for key, value in load_file(str(path)).items()}
    values["model.embed_tokens.weight"].mul_(1.3)
    values["model.layers.0.self_attn.q_proj.weight"].mul_(1.7)
    save_file(values, str(path), metadata={"format": "pt"})


@pytest.mark.parametrize("change", [
    "unchanged", "calib_seed", "source_bytes", "dataset_bytes",
    "tokenizer_bytes", "chat_template_file", "chat_template_directory",
    "persistent_tokenizer_bytes", "persistent_chat_template_file",
    "persistent_chat_template_directory",
])
def test_real_probe_main_reuses_only_matching_inputs(tmp_path, monkeypatch, capsys, change):
    source = tmp_path / "source"
    source.mkdir()
    _source(source)
    data = tmp_path / "calibration.jsonl"
    _dataset(data, 109)
    monkeypatch.setenv("CACHE_HEADROOM_GB", "0")
    monkeypatch.setenv("PRISMAQUANT_LAYER_READ_THREADS", "1")
    monkeypatch.setenv("PRISMAQUANT_PROBE_CTX_CACHE", "1" if change.startswith("persistent_") else "0")
    monkeypatch.setattr(owner, "_PROBE_CTX_CACHE", {})
    input_change = change.removeprefix("persistent_")
    template = None
    if input_change.startswith("chat_template_"):
        template = source / ("chat_template.jinja" if input_change == "chat_template_file"
                             else "additional_chat_templates/default.jinja")
        template.parent.mkdir(exist_ok=True)
        template.write_text("{{ messages[0]['content'] }}")
        records = [json.loads(line) for line in data.read_text().splitlines()]
        data.write_text("".join(json.dumps({"messages": [
            {"role": "user", "content": row["text"]}]}) + "\n" for row in records))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *args, **kwargs: torch.device("cpu"))

    def run(work, seed):
        output = work / "probe.pkl"
        owner._W_STATS_CACHE.clear()
        monkeypatch.setattr(sys, "argv", [
            "incremental_probe", "--model", str(source), "--dataset", str(data),
            "--nsamples", "2", "--seqlen", "8", "--calib-seed", str(seed),
            "--device", "cpu", "--dtype", "fp32", "--output", str(output),
            "--activation-cache-dir", str(work / "act"), "--work-dir", str(work),
            "--layers-per-shard", "1", "--prefetch-lookahead", "0",
            "--prefetch-workers", "1", "--prefetch-min-available-gb", "0",
            "--activation-rows-limit", "8", "--no-include-mtp",
            "--no-include-visual", "--no-include-lm-head", "--no-minimax-fast-moe",
        ])
        owner.main()
        with output.open("rb") as file:
            payload = pickle.load(file)
        assert payload["stats"]
        measured = {name: float(row["h_trace"]) for name, row in payload["stats"].items()}
        assert all(torch.isfinite(torch.tensor(value)) for value in measured.values())
        assert any(value > 0 for value in measured.values())
        assert payload["meta"]["calib_hash"]
        return measured, payload["meta"]["calib_hash"]

    original_source = _sha(source / "model.safetensors")
    original_data = _sha(data)
    initial, initial_draw = run(tmp_path / "shared", 42)
    shard = tmp_path / "shared" / "shards" / "probe_shard_000.pkl"
    precompute = tmp_path / "shared" / "work" / "precomputed.pt"
    original_times = (shard.stat().st_mtime_ns, precompute.stat().st_mtime_ns)
    capsys.readouterr()
    if change == "source_bytes":
        _mutate_source(source / "model.safetensors")
        assert _sha(source / "model.safetensors") != original_source
    elif change == "dataset_bytes":
        _dataset(data, 211)
        assert _sha(data) != original_data
    elif input_change == "tokenizer_bytes":
        path = source / "tokenizer.json"
        value = json.loads(path.read_text())
        vocab = value["model"]["vocab"]
        for index in range(32):
            vocab[f"w{index}"] = 34 - index
        path.write_text(json.dumps(value))
    elif template is not None:
        template.write_text("{{ 'w1 ' * 128 }}")
    seed = 43 if change == "calib_seed" else 42
    resumed, resumed_draw = run(tmp_path / "shared", seed)
    resumed_log = capsys.readouterr().out
    if change == "unchanged":
        assert "reuse shard 0:" in resumed_log
        assert (shard.stat().st_mtime_ns, precompute.stat().st_mtime_ns) == original_times
    else:
        assert resumed != initial, f"{change}: stale probe statistics were reused"
        assert "reuse shard 0:" not in resumed_log
        assert "reused precompute cache" not in resumed_log
        reason_field = {
            "calib_seed": "calib_seed",
            "source_bytes": "source_checkpoint_sha256",
            "dataset_bytes": "dataset_sha256",
            "tokenizer_bytes": "tokenizer_sha256",
            "chat_template_file": "tokenizer_sha256",
            "chat_template_directory": "tokenizer_sha256",
        }[input_change]
        assert f"probe input {reason_field} differs" in resumed_log
    owner._PROBE_CTX_CACHE.clear()
    fresh, fresh_draw = run(tmp_path / "fresh", seed)
    if input_change in ("calib_seed", "dataset_bytes", "tokenizer_bytes",
                        "chat_template_file", "chat_template_directory"):
        assert fresh_draw != initial_draw, "fixture did not change the actual token draw"
    else:
        assert fresh_draw == initial_draw
    if change != "unchanged":
        assert fresh != initial, "fixture did not change actual measured body Fisher"
    assert resumed_draw == fresh_draw, (change, initial_draw, resumed_draw, fresh_draw)
    assert resumed == fresh, (change, initial, resumed, fresh)


@pytest.mark.parametrize("change", [
    "unchanged", "calib_seed", "source_checkpoint_sha256", "dataset_sha256",
    "tokenizer_sha256", "missing_identity", "missing_tokenizer", "unidentified_dataset",
])
def test_shard_and_linear_pool_require_matching_input_identity(tmp_path, change):
    inputs = {
        "calib_seed": 42,
        "source_checkpoint_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "tokenizer_sha256": "d" * 64,
    }
    anchor = {"probe_reuse_inputs": dict(inputs)}
    recorded = dict(inputs)
    if change in inputs:
        recorded[change] = 43 if change == "calib_seed" else "c" * 64
    elif change == "missing_tokenizer":
        del recorded["tokenizer_sha256"]
    elif change == "unidentified_dataset":
        recorded["dataset_sha256"] = None
        anchor["probe_reuse_inputs"]["dataset_sha256"] = None
    meta = {} if change == "missing_identity" else {"probe_reuse_inputs": recorded}
    path = tmp_path / "probe_shard_000.pkl"
    stats = {"model.layers.0.mlp.down_proj": {"h_trace": 1.0}}
    path.write_bytes(pickle.dumps({"stats": stats, "meta": meta}))
    assert owner.probe_shard_is_reusable(path, anchor) == (change == "unchanged")
    expected = stats if change == "unchanged" else {}
    assert owner.scan_cached_linear_stats(tmp_path, anchor) == expected
