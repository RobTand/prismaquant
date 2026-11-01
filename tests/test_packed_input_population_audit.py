"""One module call contributes its input rows once to the packed cache."""
import pytest
import torch
from torch import nn
from types import SimpleNamespace
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from prismaquant.sensitivity_probe import FisherAccumulator
from test_probe_routing_maps_audit import probe_run


@pytest.mark.parametrize("row_limit", [6, 12])
def test_fisher_cache_keeps_rows_from_both_forwards(tmp_path, row_limit):
    torch.manual_seed(31)
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].anchor = nn.Linear(8, 8, bias=False)
    model.layers[0].experts = Qwen3MoeExperts(SimpleNamespace(
        num_experts=2, hidden_size=8, moe_intermediate_size=4, hidden_act="silu", _experts_implementation="eager"))
    with torch.no_grad():
        for parameter in model.layers[0].experts.parameters():
            parameter.normal_(0, .1)
    collector = FisherAccumulator(model, ["layers.0.anchor"], {}, tmp_path, input_rows=row_limit)
    first = torch.arange(48, dtype=torch.float32).reshape(6, 8) / 64
    second = first + 10
    indices = torch.tensor([[0, 1]] * 6)
    weights = torch.full((6, 2), .5)
    try:
        for rows in (first, second):
            model.layers[0].experts(rows.clone().requires_grad_(True), indices, weights).sum().backward()
        collector.finalize(tracker=None, global_tokens=12)
    finally:
        collector.remove_hooks()
    saved = torch.load(tmp_path / "layers__0__experts.pt", weights_only=True)["inputs"]
    expected = torch.cat((first, second))[:row_limit].to(torch.bfloat16)
    torch.testing.assert_close(saved, expected, rtol=0, atol=0)
    for param in ("gate_up_proj", "down_proj"):
        assert collector.stats[f"layers.0.experts.{param}"]["n_tokens_seen"] == 12


def test_incremental_main_captures_one_input_per_module_call(probe_run):
    run, work = probe_run
    run()
    saved = torch.load(work / "act" / "model__layers__0__mlp__experts.pt", weights_only=True)
    assert saved["inputs"].shape == (16, 16)


def test_incremental_main_refuses_legacy_packed_capture(probe_run, capsys):
    import pickle

    run, work = probe_run
    run()
    shard = work / "shards" / "probe_shard_000.pkl"
    payload = pickle.loads(shard.read_bytes())
    for meta in (payload["meta"], payload["meta"].get("incremental_shard", {})):
        meta.pop("packed_input_capture", None)
    shard.write_bytes(pickle.dumps(payload))
    cached_input = work / "act" / "model__layers__0__mlp__experts.pt"
    legacy = torch.load(cached_input, weights_only=True)
    legacy["inputs"] = legacy["inputs"][:16].repeat(2, 1)
    torch.save(legacy, cached_input)
    capsys.readouterr()
    run()
    assert "reuse shard 0:" not in capsys.readouterr().out
    saved = torch.load(work / "act" / "model__layers__0__mlp__experts.pt", weights_only=True)
    assert saved["inputs"].shape == (16, 16)
