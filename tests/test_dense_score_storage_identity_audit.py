"""Retained decision scores must describe the canonical stored candidate."""
from __future__ import annotations

import pytest
import torch

from prismaquant import production_render_cost as cost
from prismaquant import production_weight_cache as pwc
from test_production_weight_cache import _TinyChain


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "resume"])
def test_dense_retained_weight_cost_prices_stored_candidate(tmp_path, dtype, resume):
    model = _TinyChain().eval()
    with torch.no_grad():
        values = (torch.arange(32 * 32).reshape(32, 32) % 37 - 18).to(torch.float32) / 37
        model.l1.weight.copy_(values)
    model.to(dtype=dtype)

    def fill():
        return pwc.fill_production_weight_cache(
            model, torch.tensor([[0, 1]], dtype=torch.long), ["l1", "l2"],
            render_qnames=["l1"], formats=["FP8_E4M3"],
            levers={"gptq": False, "scale_sweep": False},
            cache_dir=tmp_path / "weights", max_act_rows=8, progress=False,
        )

    cache = fill()
    if resume:
        cache = fill()
    assert not cache.failed
    stored = cache.get("l1", "FP8_E4M3")
    assert stored.dtype == torch.bfloat16
    expected = float((model.l1.weight.detach().to(torch.float32) - stored.to(torch.float32)).pow(2).mean().item())
    record = cache.metadata["render_scores"]["records"]["l1|FP8_E4M3"]
    entry = cost._production_cost_entry(record, score_field="weight_mse")
    assert entry is not None
    assert record["weight_mse"] == expected
    assert entry["weight_mse"] == expected
