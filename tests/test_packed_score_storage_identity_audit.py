"""Batched packed costs must price the BF16 candidate actually stored."""
from __future__ import annotations

import pytest
import torch

from prismaquant import production_weight_cache as pwc
from test_packed_expert_cross_domain_gate import ASSIGNMENT, TinyLM


@pytest.mark.parametrize("mode", ["batched", "per_expert"])
def test_packed_retained_weight_cost_prices_stored_candidate(mode):
    torch.manual_seed(13)
    model = TinyLM().eval().to(torch.bfloat16)
    calib = torch.randint(0, 32, (2, 64))
    cache = pwc.ProductionWeightCache(weights={}, levers={"gptq": True})
    coverage = pwc.fill_packed_expert_cache_entries(
        cache, model, calib, render_assignment=ASSIGNMENT,
        levers={"gptq": True}, profile=None, module_token_budget=4096,
        eval_rows_per_expert=8, render_mode=mode, progress=False,
    )
    assert set(coverage) == set(ASSIGNMENT)
    assert not cache.failed
    for qname in ASSIGNMENT:
        attr = qname.rsplit(".", 1)[1]
        reference = getattr(model.mlp.experts, attr).detach().to(torch.float32)
        stored = cache.get(qname, "NVFP4")
        assert stored.dtype == torch.bfloat16
        per_expert_sums = [
            float((reference[e] - stored[e].to(torch.float32)).pow(2).mean().item())
            * reference[e].numel()
            for e in range(reference.shape[0])
        ]
        expected = sum(per_expert_sums) / reference.numel()
        record = cache.metadata["render_scores"]["records"][qname + "|NVFP4"]
        assert record["weight_mse"] == expected, (qname, record["weight_mse"], expected)
