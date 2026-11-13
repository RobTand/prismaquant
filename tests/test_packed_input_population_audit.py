"""One module call contributes its input rows once to the packed cache."""
import pytest
import torch
from torch import nn
from types import SimpleNamespace
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from prismaquant import incremental_probe as probe
from test_probe_routing_maps_audit import probe_run


@pytest.mark.parametrize("row_limit", [6, 12])
def test_act_cache_keeps_rows_from_both_forwards(tmp_path, monkeypatch, row_limit):
    monkeypatch.setenv("PRISMAQUANT_ACT_CACHE_WORKERS", "1")
    torch.manual_seed(31)
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].experts = Qwen3MoeExperts(SimpleNamespace(
        num_experts=2, hidden_size=8, moe_intermediate_size=4, hidden_act="silu", _experts_implementation="eager"))
    with torch.no_grad():
        for parameter in model.layers[0].experts.parameters():
            parameter.normal_(0, .1)
    act_rows = probe._ActRowCache(str(tmp_path), row_limit)
    first = torch.arange(48, dtype=torch.float32).reshape(6, 8) / 64
    second = first + 10
    indices = torch.tensor([[0, 1]] * 6)
    weights = torch.full((6, 2), .5)
    handle = model.layers[0].experts.register_forward_hook(
        probe._input_hook(act_rows, "layers.0.experts"))
    try:
        with torch.no_grad():
            for rows in (first, second):
                model.layers[0].experts(rows.clone(), indices, weights)
    finally:
        handle.remove()
    act_rows.flush()
    act_rows.drain()
    saved = torch.load(tmp_path / "layers__0__experts.pt", weights_only=True)
    rows = saved["row_indices"]
    # Both calls number their rows in one stream (0..11), then keep row_limit.
    assert rows.numel() == row_limit and torch.equal(rows, rows.unique())
    assert 0 <= int(rows.min()) and int(rows.max()) < 12
    expected = torch.cat((first, second))[rows]
    torch.testing.assert_close(saved["inputs"], expected, rtol=0, atol=0)
    assert act_rows.offsets["layers.0.experts"] == 12


def test_incremental_main_captures_one_input_per_module_call(probe_run):
    run, work = probe_run
    payload = run()
    saved = torch.load(work / "act" / "model__layers__0__mlp__experts.pt", weights_only=True)
    assert saved["inputs"].shape == (16, 16)
    for param in ("gate_up_proj", "down_proj"):
        assert payload["stats"][f"model.layers.0.mlp.experts.{param}"]["n_tokens_seen"] == 16
