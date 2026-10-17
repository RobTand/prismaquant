"""Dense rendering must release completed activation owners before the next unit."""
from __future__ import annotations

import gc
import weakref

import pytest
import torch

from prismaquant import production_weight_cache as pwc
from test_production_weight_cache import _TinyChain


@pytest.mark.parametrize("mode", [
    "single_return_control", "subsequent_module", "multiple_formats",
    "failed_first_render", "full_resume", "zero_activation_scalar_fallback",
])
def test_completed_dense_activation_owners_are_released(tmp_path, monkeypatch, mode):
    model = _TinyChain().eval()
    if mode == "zero_activation_scalar_fallback":
        with torch.no_grad():
            model.embed.weight.zero_()
    refs = {}
    observations = []
    first_unit_observations = []
    formats = (
        ["FP8_E4M3", "NVFP4"] if mode == "multiple_formats"
        else ["NVFP4"] if mode == "zero_activation_scalar_fallback"
        else ["FP8_E4M3"]
    )
    real_collected = pwc._LinearActivationCollector.collected
    real_render = pwc.render_production_weight

    def fill():
        return pwc.fill_production_weight_cache(
            model, torch.tensor([[0, 1]], dtype=torch.long), ["l1", "l2"],
            render_qnames=["l1"] if mode == "single_return_control" else None,
            formats=formats, levers={"gptq": False, "scale_sweep": False},
            cache_dir=tmp_path / "weights", max_act_rows=8, progress=False,
        )

    if mode == "full_resume":
        assert not fill().failed

    def observe_collected(collector):
        rows = real_collected(collector)
        refs["collected"] = weakref.ref(rows["l1"])
        refs["reservoir"] = weakref.ref(collector.activations["l1"][0])
        return rows

    def observe_render(weight, fmt, *, qname, **kwargs):
        if qname == "l1":
            first_unit_observations.append({owner: ref() is not None for owner, ref in refs.items()})
            if mode == "failed_first_render":
                raise RuntimeError("intentional first-unit render failure")
        if qname == "l2":
            gc.collect()
            observations.append({owner: ref() is not None for owner, ref in refs.items()})
        return real_render(weight, fmt, qname=qname, **kwargs)

    monkeypatch.setattr(pwc._LinearActivationCollector, "collected", observe_collected)
    monkeypatch.setattr(pwc, "render_production_weight", observe_render)
    cache = fill()
    if mode == "full_resume":
        assert not cache.failed
        assert refs == {}
        assert observations == first_unit_observations == []
        return
    if mode == "failed_first_render":
        assert set(cache.failed) == {("l1", "FP8_E4M3")}
    else:
        assert not cache.failed
    assert set(refs) == {"collected", "reservoir"}
    assert all(ref() is None for ref in refs.values()), "fixture retained activation references after return"
    assert first_unit_observations == [{"collected": True, "reservoir": False}] * len(formats)
    if mode != "single_return_control":
        assert observations == [{"collected": False, "reservoir": False}] * len(formats)
    else:
        assert observations == []


def test_renderer_preserves_caller_owned_activation_map():
    model = _TinyChain().eval()
    rows = torch.ones(2, 32)
    before = rows.clone()
    activations = {"l1": rows}
    rendered = pwc.render_production_weight(
        model.l1.weight.detach(), "FP8_E4M3", qname="l1",
        activations=activations, levers={"gptq": False, "scale_sweep": False},
    )
    assert rendered.shape == model.l1.weight.shape
    assert set(activations) == {"l1"}
    assert activations["l1"] is rows
    assert torch.equal(rows, before)
