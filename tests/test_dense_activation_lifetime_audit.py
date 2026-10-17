"""Dense rendering must release completed activation owners before the next unit."""
from __future__ import annotations

import gc
import weakref

import pytest
import torch

from prismaquant import production_weight_cache as pwc
from test_production_weight_cache import _TinyChain


@pytest.mark.parametrize("two_linears", [False, True], ids=["single_return_control", "subsequent_module"])
def test_completed_dense_activation_owners_are_released(tmp_path, monkeypatch, two_linears):
    model = _TinyChain().eval()
    refs = {}
    observations = []
    real_collected = pwc._LinearActivationCollector.collected
    real_render = pwc.render_production_weight

    def observe_collected(collector):
        rows = real_collected(collector)
        refs["collected"] = weakref.ref(rows["l1"])
        refs["reservoir"] = weakref.ref(collector.activations["l1"][0])
        return rows

    def observe_render(weight, fmt, *, qname, **kwargs):
        if qname == "l2":
            gc.collect()
            observations.append({owner: ref() is not None for owner, ref in refs.items()})
        return real_render(weight, fmt, qname=qname, **kwargs)

    monkeypatch.setattr(pwc._LinearActivationCollector, "collected", observe_collected)
    monkeypatch.setattr(pwc, "render_production_weight", observe_render)
    cache = pwc.fill_production_weight_cache(
        model, torch.tensor([[0, 1]], dtype=torch.long), ["l1", "l2"],
        render_qnames=None if two_linears else ["l1"],
        formats=["FP8_E4M3"], levers={"gptq": False, "scale_sweep": False},
        cache_dir=tmp_path / "weights", max_act_rows=8, progress=False,
    )
    assert not cache.failed
    assert set(refs) == {"collected", "reservoir"}
    assert all(ref() is None for ref in refs.values()), "fixture retained activation references after return"
    if two_linears:
        assert observations == [{"collected": False, "reservoir": False}]
    else:
        assert observations == []
